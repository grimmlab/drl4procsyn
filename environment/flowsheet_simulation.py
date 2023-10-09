import copy

import numpy as np
import scipy.optimize as opt
import itertools

from environment import units


def ensure_positive_flowrates(molar_flowrates):
    return molar_flowrates.clip(min=0)


def compute_molar_fractions(molar_flowrates):
    if sum(molar_flowrates) > 0:
        molar_fractions = molar_flowrates / sum(molar_flowrates)

    else:
        molar_fractions = np.zeros(len(molar_flowrates))

    return molar_fractions


def get_leaving_streams(state_simulation):
    """
    get streams, which leave the process currently
    """
    list_leaving_streams = []
    for line_index, line in enumerate(state_simulation["list_raw_lines"]):
        # check if the stream is an open stream, i.e. leaving the process, by checking if an
        # apparatus is connected to it
        if sum(line["OHE_unit"]) == 0:
            list_leaving_streams.append({"index": line_index, "flowrate": line["total_flowrate"],
                                         "composition": line["molar_fractions"]})

    return list_leaving_streams


def check_mass_balance(tree_structure):
    """
    we assume no reactions between the components
    """
    mass_balance_fulfilled = True

    # get an array with an entry for every existing component
    component_array = np.zeros(tree_structure.config.max_number_of_components)

    # add up feed streams
    feeds_added_up = np.zeros(tree_structure.config.max_number_of_components)
    for _, stream in enumerate(tree_structure.feed_stream_information["list_feed_streams"]):
        feeds_added_up = feeds_added_up + stream

    component_array = component_array + feeds_added_up

    # add up added components, for this go through lines
    for line_ind, line in enumerate(tree_structure.lines):
        if line.apparatus is not None:
            if line.apparatus.apparatus_type_name == "add_solvent":
                inp_flowrates = tree_structure.accumulate_line_input(line_ind)
                out_flowrates = tree_structure.compute_apparatus_output(line.apparatus, inp_flowrates)

                difference = out_flowrates[0] - inp_flowrates
                difference.clip(min=0)

                component_array = component_array + difference

    # get leaving streams
    for i, line in enumerate(tree_structure.lines):
        if line.apparatus is None:
            line_dict = tree_structure.get_line_information(line_index=i)
            component_array = component_array - (line_dict["total_flowrate"] * line_dict["molar_fractions"])

    # eps criterion
    for entry in component_array:
        if np.abs(entry) > 0.01 * sum(feeds_added_up):
            mass_balance_fulfilled = False

    return mass_balance_fulfilled, component_array


class Stream:
    """
    stream inside the flowsheet, serves as connection between lines of the flowsheet matrix
    """
    def __init__(self, molar_flowrates, stream_index, start_index, end_index, recycle_num):
        self.molar_flowrates = molar_flowrates
        self.stream_index = stream_index  # index in list of streams
        self.start_index = start_index  # stream starts in this line, None for feed streams
        self.end_index = end_index  # stream ends in this line
        # if this is a recycle store its "number" here (None otherwise), number means its for
        # example the third recycle stream in this flowsheet -> 2
        self.recycle_num = recycle_num


class MatrixLine:
    """
    line in the flowsheet matrix, contains up to one apparatus, information on its input streams and
    is connected to other lines by multiple streams
    """
    def __init__(self, input_streams_indices, line_index):
        self.input_streams_indices = input_streams_indices
        self.line_index = line_index
        self.apparatus = None  # contains the class of the apparatus in this line
        self.output_streams_indices = []


class Apparatus:
    """
    contains the apparatus type and its specifications, allows to compute its output for a given input
    """
    def __init__(self, apparatus_type_index, apparatus_type_name,
                 specification_discrete, specification_continuous,
                 num_output_streams):
        self.apparatus_type_index = apparatus_type_index
        self.apparatus_type_name = apparatus_type_name
        self.specification_discrete = specification_discrete
        self.specification_continuous = specification_continuous

        # each apparatus type has a fixed number of output streams, which is stored
        # in the config and provided within initialization in a list
        self.num_outputs = num_output_streams


class TreeStructure:
    """
    Contains tree structure for one flowsheet. The flowsheet matrix can be constructed from the tree.
    Each line in the matrix can contain up to one apparatus and its input and output streams are the
    connections to the other lines.

    Alert: this tree structure does not have anything to do with the MCTS of the RL agent.
    """
    def __init__(self, feed_stream_information, config):
        self.config = config
        self.feed_stream_information = feed_stream_information

        # indices of components, which are present in the flowsheet (also in correct order)
        self.current_indices = self.feed_stream_information["indices_components_in_feeds"]
        self.present_in_feed = copy.deepcopy(self.current_indices)

        # limit recycles
        self.limits_recycles = np.zeros(self.config.max_number_of_components)
        for feed in self.feed_stream_information["list_feed_streams"]:
            self.limits_recycles = self.limits_recycles + feed

        self.sum_feeds = sum(self.limits_recycles)
        self.limits_recycles = self.limits_recycles * self.config.limit_recycle_size
        for j in range(len(self.current_indices), self.config.max_number_of_components):
            self.limits_recycles[j] = self.sum_feeds * self.config.limit_recycle_size

        # track solvent added
        self.counter_add_solvent = 0

        # we need a mapping from current indices to the molar fractions vector
        self.mapping = [None] * len(self.config.phase_eq_generator.names_components)
        self.update_mapping()

        # current phase eq, may be updated/changed later
        self.current_phase_eq = {"lle": self.feed_stream_information["lle_for_start"],
                                 "vle": self.feed_stream_information["vle_for_start"],
                                 "indices": self.feed_stream_information["lle_for_start"]["indices_components"]}

        if len(self.current_indices) > self.config.max_number_of_components:
            print("initialization wrong")
            return None

        self.roots = self.feed_stream_information["list_feed_streams"]  # stores the root(s) of the tree structure
        self.lines = []  # list of lines in the flowsheet matrix
        self.streams = []  # multiple streams connect the apparatus / lines
        self.recyclestream_indices = []  # indices of recycle streams (referring to self.streams list)

        # for each given feed stream, add a stream as root of the tree
        for index, feedstream in enumerate(self.roots):
            self.streams.append(Stream(feedstream, index, None, index, None))
            self.lines.append(MatrixLine([index], index))

    def update_mapping(self):
        # if current indices are changed, we need to update the mapping
        for j, ind in enumerate(self.current_indices):
            self.mapping[ind] = j

    # sums up all streams which go into a line
    def accumulate_line_input(self, line_index):
        accumulation = np.zeros(self.config.max_number_of_components)
        for stream_index in self.lines[line_index].input_streams_indices:
            accumulation = accumulation + self.streams[stream_index].molar_flowrates

        return accumulation

    # add new apparatus
    def add_apparatus(self, line_index, apparatus_type_index,
                      specification_continuous, specification_discrete):
        # get apparatus type name
        apparatus_type_name = self.config.units_map_indices_type[apparatus_type_index]
        self.lines[line_index].apparatus = Apparatus(apparatus_type_index, apparatus_type_name,
                                                     specification_discrete, specification_continuous,
                                                     self.config.unit_types[apparatus_type_name]["output_streams"])

        # if we add a solvent, we update the current indices
        if apparatus_type_name == "add_solvent":
            self.counter_add_solvent = self.counter_add_solvent + 1
            solvent_index = apparatus_type_index - self.config.add_solvent_start_index
            if solvent_index not in self.current_indices:
                self.current_indices.append(solvent_index)
                self.update_mapping()

                # update phase eq
                names = [self.config.phase_eq_generator.names_components[ik] for ik in self.current_indices]
                phase_eq_dict = self.config.phase_eq_generator.search_subsystem_phase_eq(names)

                self.current_phase_eq["vle"] = phase_eq_dict["vle"]
                self.current_phase_eq["lle"] = phase_eq_dict["lle"]
                self.current_phase_eq["indices"] = phase_eq_dict["lle"]["indices_components"]

            # for safety
            if len(self.current_indices) > self.config.max_number_of_components:
                return None

        outputs = self.compute_apparatus_output(self.lines[line_index].apparatus,
                                                self.accumulate_line_input(line_index))

        # if the unit is no recycle or mixer, we just add the new lines to our structure
        if (apparatus_type_name != "recycle") and (apparatus_type_name != "mixer"):
            # those apparatus models converge always
            process_convergent = True
            for output in outputs:
                # store new streams
                self.streams.append(Stream(
                    output, len(self.streams), line_index, len(self.lines), None))

                # these are output streams of the chosen line
                self.lines[line_index].output_streams_indices.append(len(self.streams) - 1)

                # store new lines
                self.lines.append(MatrixLine([len(self.streams) - 1], len(self.lines)))

        # unit is recycle or mixer
        else:
            if apparatus_type_name == "mixer":
                process_convergent = True
                self.streams.append(Stream(outputs[0], len(self.streams),
                                           line_index, specification_discrete,
                                           None))

                self.lines[line_index].output_streams_indices.append(len(self.streams) - 1)
                self.lines[specification_discrete].input_streams_indices.append(len(self.streams) - 1)

            else:
                # store the recycle as stream
                self.streams.append(Stream(outputs[0], len(self.streams),
                                           line_index, specification_discrete,
                                           len(self.recyclestream_indices)))

                self.lines[line_index].output_streams_indices.append(len(self.streams) - 1)
                # store that this stream is a recycle in the designated list
                self.recyclestream_indices.append(len(self.streams) - 1)
                self.lines[specification_discrete].input_streams_indices.append(len(self.streams) - 1)

                guesses = []

                # the initial guess for the root finding problem is equal to the flowrates, which we get
                # when we run through the loop one time till the tear streams
                initial_guess = np.zeros(len(self.recyclestream_indices) * self.config.max_number_of_components)
                for counter, rec_stream_index in enumerate(self.recyclestream_indices):
                    initial_guess[counter * self.config.max_number_of_components: (
                        counter + 1) * self.config.max_number_of_components] = \
                        self.streams[rec_stream_index].molar_flowrates

                guesses.append(initial_guess)
                # the second guess are zeros
                guesses.append(np.zeros(len(initial_guess)))

                # random numbers can also serve as initial guess
                for i in range(self.config.random_guesses_root_iteration):
                    guesses.append(np.random.rand(len(initial_guess)))

                process_convergent = False
                # test guesses with root finding algorithm
                for guess in guesses:
                    out_solve = opt.fsolve(self.recycle_loop_root, guess, full_output=True, maxfev=self.config.max_num_root_finding_interactions)
                    fvec_candidate = out_solve[1]["fvec"]  # function evaluation of candidate solution

                    # test if proposed solution is good enough
                    if sum(np.abs(fvec_candidate)) < \
                            self.config.epsilon * len(self.recyclestream_indices) * self.config.max_number_of_components:

                        process_convergent = True
                        break

                # if all guesses failed, we try the wegstein method combined with the initial guess
                # (initial guess = current state of process)
                if not process_convergent:
                    if self.config.use_wegstein:
                        x = initial_guess
                        for step in range(self.config.wegstein_steps):
                            old_value = x
                            new_value = self.wegstein_step(x)
                            if sum(np.abs(old_value - new_value)) < self.config.epsilon * len(
                                    old_value) * self.config.max_number_of_components:
                                process_convergent = True
                                break

                            else:
                                x = (self.config.wegstein_constant * old_value) + (
                                        (1 - self.config.wegstein_constant) * new_value)

                # for safety, check mbil
                if process_convergent:
                    mass_balance_fulfilled, comp_array = check_mass_balance(self)
                    if not mass_balance_fulfilled:
                        process_convergent = False

        return process_convergent

    # resets all streams to None, feeds to initial values and recycles to guessed tear streams
    # then runs through the complete flowsheet and returns the computed values at the tears
    def wegstein_step(self, guess):
        guess = ensure_positive_flowrates(guess)

        # limit the guess to the allowed size
        current_index = 0
        while current_index < len(guess):
            for j in range(self.config.max_number_of_components):
                guess[current_index + j] = np.min([guess[current_index + j], self.limits_recycles[j]])

            current_index = current_index + self.config.max_number_of_components

        computed_tear_streams = np.zeros(len(guess))

        # set feeds to initial values
        # note that feeds cannot be recycled but just mixed somewhere.
        # For the case that a feed stream is mixed somewhere, the reset also fulfills its purpose, as
        # most likely this initial value is already the solution for the "mixing root". For the rare case, that
        # previously another stream was mixed to the feed stream, which is now mixed to somewhere: the framework
        # created already a new stream for the rec operation, which will later get the value of the guess.
        # Therefore, no values of the guess are wasted or omitted.
        for i in range(len(self.roots)):
            self.streams[i].molar_flowrates = self.roots[i]

        # set all other streams to None, except the recycles
        for i in range(len(self.roots), len(self.streams)):
            if self.streams[i].recycle_num is None:
                self.streams[i].molar_flowrates = None

            else:
                # recycle_num is the constant index, where the guess for this exact stream is stored
                relevant_index = self.streams[i].recycle_num * self.config.max_number_of_components
                self.streams[i].molar_flowrates = guess[
                    relevant_index: relevant_index + self.config.max_number_of_components]

        # we just run through all the lines and compute the respective outputs
        # if a recycle is reached, it is compared to the guess
        # there is one exception: when mixers are in the flowsheet, accumulation of the line
        # input may not work (as the mixer output is not computed yet). Therefore, we store
        # those occurrencies and compute them later
        indices_to_compute = []
        for line_index, line in enumerate(self.lines):
            if line.apparatus is not None:
                do_it_later = False
                for stream_index in self.lines[line_index].input_streams_indices:
                    if self.streams[stream_index].molar_flowrates is None:
                        do_it_later = True
                        break

                if do_it_later:
                    indices_to_compute.append(line_index)

                else:
                    computed_tear_streams = self.set_streams_within_wegstein(
                        line_index=line_index,
                        computed_tear_streams=computed_tear_streams
                    )

        # do remaining streams
        while len(indices_to_compute) > 0:
            temp_indices = []
            for _, line_index in enumerate(indices_to_compute):
                do_it_later = False
                for stream_index in self.lines[line_index].input_streams_indices:
                    if self.streams[stream_index].molar_flowrates is None:
                        do_it_later = True
                        temp_indices.append(line_index)
                        break

                if not do_it_later:
                    computed_tear_streams = self.set_streams_within_wegstein(
                        line_index=line_index,
                        computed_tear_streams=computed_tear_streams
                    )

            indices_to_compute = temp_indices

        return computed_tear_streams

    def set_streams_within_wegstein(self, line_index, computed_tear_streams):
        new_output = self.compute_apparatus_output(self.lines[line_index].apparatus,
                                                   self.accumulate_line_input(line_index))
        if self.lines[line_index].apparatus.apparatus_type_name != "recycle":
            for i in range(len(new_output)):
                self.streams[self.lines[line_index].output_streams_indices[i]].molar_flowrates = \
                    new_output[i]

        else:
            # we get the corresponding index for the difference array (inside the stream we stored
            # before this information)
            relevant_index = self.streams[self.lines[line_index].output_streams_indices[0]].recycle_num * \
                             self.config.max_number_of_components

            computed_tear_streams[relevant_index: relevant_index + self.config.max_number_of_components] = \
                new_output[0]

        return computed_tear_streams

    # function for the root-finding algorithm
    def recycle_loop_root(self, guess):
        return guess - self.wegstein_step(guess)

    # get information on certain line of the flowsheet matrix
    def get_line_information(self, line_index):
        line_dict = {}

        # own index
        line_dict["own_line_index"] = line_index

        # if the line contains a feed stream, we mark it
        if line_index < len(self.roots):
            line_dict["is_feed"] = 1

        else:
            line_dict["is_feed"] = 0

        flowrates = np.zeros(self.config.max_number_of_components)
        for stream_index in self.lines[line_index].input_streams_indices:
            flowrates = flowrates + self.streams[stream_index].molar_flowrates

        line_dict["total_flowrate"] = sum(flowrates)
        line_dict["molar_fractions"] = compute_molar_fractions(flowrates)
        line_dict["pseudo_mass_flowrate"] = 0

        if self.config.npv_version == "literature":
            molar_flowrates = line_dict["molar_fractions"] * line_dict["total_flowrate"]
            for j, index in enumerate(self.current_indices):
                name = self.config.phase_eq_generator.names_components[index]
                molar_mass = self.config.dict_pure_component_data[name]["M"]

                line_dict["pseudo_mass_flowrate"] = line_dict["pseudo_mass_flowrate"] + (
                        self.config.scale_pseudo_mass * molar_flowrates[j] * molar_mass)

        # T in Kelvin, p in bar
        line_dict["T"] = self.config.phase_eq_generator.subsystems_temperatures[
            self.feed_stream_information["feed_situation_index"]]
        line_dict["p"] = self.config.phase_eq_generator.subsystems_pressures[
            self.feed_stream_information["feed_situation_index"]]

        # apparatus information
        ohe_unit = np.zeros(self.config.num_units)
        cont_spec = 0
        ohe_cont = 0
        indices_destinations_output_streams = []
        if self.lines[line_index].apparatus is not None:
            ohe_unit[self.lines[line_index].apparatus.apparatus_type_index] = 1

            # unit specification continuous, check first if the spec is continuous (determined
            # in the config by a set range)
            if self.config.ranges_cont_specification[
                    self.lines[line_index].apparatus.apparatus_type_index] is not None:
                ohe_cont = 1
                cont_spec = self.lines[line_index].apparatus.specification_continuous

            # set destinations of output streams
            for stream_index in self.lines[line_index].output_streams_indices:
                indices_destinations_output_streams.append(self.streams[stream_index].end_index)

        line_dict["OHE_unit"] = ohe_unit
        line_dict["cont_spec"] = cont_spec
        line_dict["OHE_cont_spec"] = ohe_cont
        line_dict["indices_destinations_output_streams"] = indices_destinations_output_streams

        return line_dict

    def get_flowsheet_information(self):
        """
        returns a list, each element corresponds to a line of the former flowsheet matrix
        """
        flowsheet_information = []
        for i in range(len(self.lines)):
            flowsheet_information.append(self.get_line_information(i))

        return flowsheet_information

    def compute_apparatus_output(self, apparatus, input_molar_flowrates):
        """
        unit_class and component system contain the correct models from units, before anything
        is calculated there, the flowrates are transformed to the correct subsystem
        """
        # ensure that no negative values are processed (could occur during the use of fsolve)
        input_molar_flowrates = ensure_positive_flowrates(input_molar_flowrates)

        # for a non-empty input stream, there is something to compute
        # the output stream(s) are always provided as list
        # distillation column, split, decanter, add_component, mixer, recycle
        if sum(input_molar_flowrates) > 0:
            if apparatus.apparatus_type_name == "distillation_column":
                # we transform the stream to the current phase_eq
                flowrates_transformed_to_phase_eq = units.transform_stream_fs_to_stream_phase_eq(
                    molar_flowrates_flowsheet=input_molar_flowrates,
                    order_components_flowsheet=self.current_indices,
                    phase_eq_order_components=self.current_phase_eq["indices"])

                output_streams = units.distillation(
                    transformed_feed_flowrates=flowrates_transformed_to_phase_eq,
                    df=apparatus.specification_continuous,
                    column=self.config.distillation_column,
                    current_vle=self.current_phase_eq["vle"]["phase_eq"]
                    )

                # backtransformation
                for index in range(len(output_streams)):
                    output_streams[index] = units.transform_stream_phase_eq_to_stream_fs(
                        molar_flowrates_phase_eq=output_streams[index],
                        phase_eq_order_components=self.current_phase_eq["indices"],
                        order_components_flowsheet=self.current_indices,
                        max_num_components_flowsheet=self.config.max_number_of_components
                    )

            elif apparatus.apparatus_type_name == "split":
                output_streams = units.split(
                    feed_molar_flowrates=input_molar_flowrates,
                    split_ratio=apparatus.specification_continuous)

            elif apparatus.apparatus_type_name == "decanter":
                # we transform the stream to the current phase_eq
                flowrates_transformed_to_phase_eq = units.transform_stream_fs_to_stream_phase_eq(
                    molar_flowrates_flowsheet=input_molar_flowrates,
                    order_components_flowsheet=self.current_indices,
                    phase_eq_order_components=self.current_phase_eq["indices"])

                output_streams = units.decantation(
                    transformed_feed_molar_flowrates=flowrates_transformed_to_phase_eq,
                    current_phase_eq_liq=self.current_phase_eq["lle"]["phase_eq"]
                )

                # backtransformation
                for index in range(len(output_streams)):
                    output_streams[index] = units.transform_stream_phase_eq_to_stream_fs(
                        molar_flowrates_phase_eq=output_streams[index],
                        phase_eq_order_components=self.current_phase_eq["indices"],
                        order_components_flowsheet=self.current_indices,
                        max_num_components_flowsheet=self.config.max_number_of_components
                    )

            elif apparatus.apparatus_type_name == "add_solvent":
                output_streams = units.add_solvent(
                    feed_molar_flowrates=input_molar_flowrates,
                    index_new_component=self.mapping[apparatus.apparatus_type_index\
                                                     - self.config.add_solvent_start_index],
                    ratio_component_to_feed=apparatus.specification_continuous)

            elif apparatus.apparatus_type_name == "mixer":
                output_streams = [input_molar_flowrates]

            elif apparatus.apparatus_type_name == "recycle":
                output_streams = [input_molar_flowrates]

            # if a distillation column or decanter was used, we have to ensure that for two comp
            # flowsheets there was no third comp introduced
            if len(self.current_indices) == 2:
                for index in range(len(output_streams)):
                    output_streams[index][-1] = 0

            for index in range(len(output_streams)):
                output_streams[index] = ensure_positive_flowrates(output_streams[index])

        else:
            # if the input is empty, we just provide as many output streams as needed for this apparatus type
            output_streams = []
            for i in range(apparatus.num_outputs):
                output_streams.append(np.zeros(len(input_molar_flowrates)))

        return output_streams


class FlowsheetSimulation:
    """
    Class for the simulation of a flowsheet.
    Flowsheets will be build up in a tree like structure to be able to simulate recycle loops.
    To initialize the class, the feed stream information dict has to be provided (see env_config
    for description).

    config is from env_config and stores necessary information for the simulation.
    """
    def __init__(self, feed_stream_information, config):
        self.config = config
        self.synthesis_completed = False  # track if process is finished
        self.convergent_process = True  # track if process flowsheet is convergent
        # npv of process is computed, when the flowsheet is finished
        self.net_present_value, self.net_present_value_normed = None, None

        # store all information to rebuild process if needed
        self.blueprint = {"initial_information": copy.deepcopy(feed_stream_information)}
        self.blueprint["move_seq"] = []

        # search for current phase eqs and place it in feed stream info
        phase_eq_dict = self.config.phase_eq_generator.search_subsystem_phase_eq(
            feed_stream_information["comp_order_feeds"]
        )
        feed_stream_information["vle_for_start"] = phase_eq_dict["vle"]
        feed_stream_information["lle_for_start"] = phase_eq_dict["lle"]

        self.sit_index = feed_stream_information["feed_situation_index"]
        self.num_comps_in_feed = len(self.config.phase_eq_generator.feed_situations[self.sit_index][0])

        # one hot encoding, of which add components are allowed
        self.restrictions_for_feasible = np.zeros(len(
            self.config.phase_eq_generator.names_components))

        for i in feed_stream_information["possible_ind_add_comp"]:
            self.restrictions_for_feasible[i] = 1

        # tree structure, especially needed for recycle loops
        self.tree_structure = TreeStructure(feed_stream_information, self.config)

        # to track max number of apparatus
        self.steps = 0

        # initialize flowsheet information
        self.state_simulation = self.get_current_state(first_init=True)
        self.current_net_present_value, self.current_net_present_value_normed = self.compute_npv()

        # update leaving streams
        self.blueprint["leaving_streams"] = get_leaving_streams(state_simulation=self.state_simulation)

        # in some cases, we want an option to just get the 'score' of pure streams leaving the
        # process. therefore apparatus cost etc are neglected (basically the npv added up by positive
        # values of leaving streams (and negative of added solvents).
        self.npv_without_app_cost = None

    def convert_line_dict_for_transformer(self, line_dict, pre_y):
        """
        line_dict is a dict generated by get_line_information, here we convert it to a proper format, which
        can be processed with an embedding later on for the transformer. pre_y contains information on pure
        components, interactions of components. we add mfr and flowrate here. also we return the destinations
        of the output streams and the line index. see syn_game for detailed description of format.
        """
        transformed_dict = {}
        transformed_dict["input_w_stream"] = np.concatenate((line_dict["molar_fractions"],
                                                             line_dict["total_flowrate"],
                                                             line_dict["pseudo_mass_flowrate"],
                                                             pre_y), axis=None)

        transformed_dict["input_unit"] = line_dict["OHE_unit"]

        key = self.config.keys_for_unit_spec_vector[0]
        vector = line_dict[key]

        for key in self.config.keys_for_unit_spec_vector[1:]:
            vector = np.concatenate((vector, line_dict[key]), axis=None)

        transformed_dict["input_w_unit_specification"] = vector

        return {"transformed_dict": transformed_dict,
                "line_index": line_dict["own_line_index"],
                "output_destinations": line_dict["indices_destinations_output_streams"]}

    def get_current_state(self, first_init=False):
        """
        description of state is in syn_game
        """
        state = {}

        # if the current indices in the flowsheet did not change, we just take the vectors from the actual state
        # of course this can not be done if the state is not initialized yet
        update_component_vectors = False
        if not first_init:
            # check if indices did not change
            if len(self.tree_structure.current_indices) != self.state_simulation["len_current_indices"]:
                update_component_vectors = True

        else:
            update_component_vectors = True

        if update_component_vectors:
            pure_component_vectors_list = []
            for pure_index in self.tree_structure.current_indices:
                # first get information on pure components (in correct order), tc, pc, omega for now...
                name = self.config.phase_eq_generator.names_components[pure_index]
                pure_info = self.config.dict_pure_component_data[name]

                # up to now, only critical data is in the state
                pure_component_vectors_list.append(
                    pure_info["critical_data"])

            # if the max number of components is not already reached, we fill the list with zero vectors
            while len(pure_component_vectors_list) < self.config.max_number_of_components:
                pure_component_vectors_list.append(np.zeros(len(pure_component_vectors_list[0])))

            # now get interaction parameter matrix (gamma inf)
            combinations = itertools.combinations(list(range(len(self.tree_structure.current_indices))), 2)
            inter_list = []
            for comb in combinations:
                name_index_1 = self.tree_structure.current_indices[comb[0]]
                name_index_2 = self.tree_structure.current_indices[comb[1]]
                name_1 = self.config.phase_eq_generator.names_components[name_index_1]
                name_2 = self.config.phase_eq_generator.names_components[name_index_2]
                temperature = self.config.phase_eq_generator.subsystems_temperatures[
                    self.blueprint["initial_information"]["feed_situation_index"]]
                interaction = self.config.phase_eq_generator.compute_inf_dilution_act_coeffs(
                    name_1, name_2, temperature)
                inter_list.append(interaction)

            interactions = inter_list[0]
            for interaction in inter_list[1:]:
                interactions = np.concatenate((interactions, interaction), axis=None)

            # the resulting vector is filled with zeros to always have a constant length
            len_full_combinations = len(tuple(
                itertools.combinations(list(range(self.config.max_number_of_components)), 2)))
            if len_full_combinations > len(inter_list):
                for i in range(len_full_combinations - len(inter_list)):
                    interactions = np.concatenate((interactions, np.zeros(2)), axis=None)

            # concatenate all information on the components
            pre_y = pure_component_vectors_list[0]
            for v in pure_component_vectors_list[1:]:
                pre_y = np.concatenate((pre_y, v), axis=None)

            pre_y = np.concatenate((pre_y, interactions), axis=None)
            state["pre_y"] = pre_y

            # information for next time
            state["len_current_indices"] = len(self.tree_structure.current_indices)

        else:
            # take everything from the current state
            state["pre_y"] = self.state_simulation["pre_y"]
            state["len_current_indices"] = self.state_simulation["len_current_indices"]

        # now get information on the current flowsheet
        raw_line_dicts_list = self.tree_structure.get_flowsheet_information()
        state["list_raw_lines"] = raw_line_dicts_list

        # convert it to the format described in syn_game
        state["list_line_information"] = [self.convert_line_dict_for_transformer(
            line, state["pre_y"])["transformed_dict"] for line in raw_line_dicts_list]

        num_lines = len(raw_line_dicts_list)
        connectivity_matrix = np.zeros((num_lines, num_lines), dtype=np.uint8)
        # forward destinations are marked with 1
        for i in range(num_lines):
            for j in raw_line_dicts_list[i]["indices_destinations_output_streams"]:
                connectivity_matrix[i, j] = 1

                # only needed for transformer
                if not self.config.legacy:
                    connectivity_matrix[j, i] = 2

        state["lines_connectivity_matrix"] = connectivity_matrix

        # other information on the flowsheet
        state["blueprint"] = self.blueprint
        state["steps"] = self.steps
        state["flowsheet_syn_done"] = self.synthesis_completed

        return state

    def place_apparatus(self, line_index, apparatus_type_index,
                        specification_continuous, specification_discrete):
        self.steps = self.steps + 1

        if self.steps > self.config.max_steps_for_flowsheet_synthesis:
            print("\nplayed more steps than allowed\n")
            return None

        # this case corresponds to the terminate action of the flowsheet synthesis
        if line_index == len(self.tree_structure.lines):
            self.blueprint["move_seq"].append([line_index])
            self.synthesis_completed = True

        elif self.config.legacy and line_index == self.config.num_lines_flowsheet_matrix:
            self.blueprint["move_seq"].append([line_index])
            self.synthesis_completed = True

        else:
            # add apparatus in tree, get convergence marker in return
            # from the cont spec, we just need the value, the other information
            # is stored in the blueprint, to be able to see later on the chosen
            # branches in the tree
            if specification_continuous is not None:
                cont_para = specification_continuous[0]

            else:
                cont_para = None

            self.convergent_process = self.tree_structure.add_apparatus(
                line_index=line_index, apparatus_type_index=apparatus_type_index,
                specification_continuous=cont_para,
                specification_discrete=specification_discrete)

            self.blueprint["move_seq"].append([line_index,
                                               self.config.units_map_indices_type[apparatus_type_index],
                                               apparatus_type_index,
                                               specification_continuous, specification_discrete])

        if self.convergent_process:
            # update flowsheet information
            self.state_simulation = self.get_current_state()

            # if the process is finished
            if self.synthesis_completed:
                self.net_present_value, self.net_present_value_normed = self.compute_npv()

            # current npv
            self.current_net_present_value, self.current_net_present_value_normed = self.compute_npv()

            # update leaving streams
            self.blueprint["leaving_streams"] = get_leaving_streams(state_simulation=self.state_simulation)

        else:
            self.state_simulation = None
            self.net_present_value, self.net_present_value_normed = None, None
            self.synthesis_completed = False
            self.current_net_present_value, self.current_net_present_value_normed = None, None
            self.blueprint["leaving_streams"] = None

        return self.state_simulation, self.net_present_value, self.net_present_value_normed, \
            self.synthesis_completed, self.convergent_process

    def get_feasible_actions(self, current_level, chosen_stream, chosen_unit):
        # filter feasible actions depending on the flowsheet matrix, current level and chosen feed stream
        # level 0 is for choice of streams or termination action
        if current_level == 0:
            # one action for each possible line in the matrix and an additional action for termination
            # if legacy is True, we have a max number of lines
            if self.config.legacy:
                feasible_actions = np.zeros(self.config.num_lines_flowsheet_matrix + 1)

            else:
                feasible_actions = np.zeros(len(self.state_simulation["list_line_information"]) + 1)

            # first check, if max number of steps is already reached
            if self.steps == self.config.max_steps_for_flowsheet_synthesis - 1:
                # if so, only termination is allowed
                feasible_actions[-1] = 1

            else:
                # check for open streams with flowrate > 0
                for line_index, line in enumerate(self.state_simulation["list_raw_lines"]):
                    if line["total_flowrate"] > 0 and sum(line["OHE_unit"]) == 0:
                        feasible_actions[line_index] = 1

                # last action is termination and always legal / feasible except if specified
                # in the config that the first x steps should be done without termination
                # as we want to get some process. only exception form this is, if there would be no
                # other action left.
                if len(self.blueprint["move_seq"]) >= self.config.termination_possible_from_step or sum(
                        feasible_actions) < 0.5:
                    feasible_actions[-1] = 1

        # at level 1 we choose a unit
        elif current_level == 1:
            # order of units: distillation column, split, decanter, various add_component, mixer,
            # recycle
            feasible_actions = np.zeros(self.config.num_units)
            num_open_streams = 0
            num_closed_non_rec_streams = 0

            # in the previous level it is ensured that each unit fits in,
            # therefore we can declare distillation column, split and decanter feasible
            for key in ["distillation_column", "decanter"]:
                feasible_actions[self.config.units_indices_ranges[key][0]:self.config.units_indices_ranges[key][1]] = \
                    feasible_actions[
                    self.config.units_indices_ranges[key][0]:self.config.units_indices_ranges[key][1]] + 1

            if self.config.allow_split:
                feasible_actions[
                    self.config.units_indices_ranges["split"][0]:self.config.units_indices_ranges["split"][1]] = \
                    feasible_actions[
                    self.config.units_indices_ranges["split"][0]:self.config.units_indices_ranges["split"][1]] + 1

            # add solvent is a special action, it is only allowed to add one extra solvent
            # per process, this is ensured here (also it is not allowed to add feed components)
            if self.config.allow_solvent:
                # first, check if we already added as much solvent as allowed
                if self.tree_structure.counter_add_solvent < self.config.num_add_solvent_actions:
                    if len(self.tree_structure.current_indices) < self.config.max_number_of_components:
                        for i in range(len(self.restrictions_for_feasible)):
                            if self.restrictions_for_feasible[i] == 1:
                                feasible_actions[self.config.add_solvent_start_index+i] = 1

                    else:
                        # in this case, only the already existing solvent can be added
                        for i in self.tree_structure.current_indices:
                            if i not in self.tree_structure.present_in_feed:
                                feasible_actions[self.config.add_solvent_start_index+i] = 1

            # now we check for mixers and recycles
            # count open streams and closed streams, which are no recycle (closed streams, which are
            # no recycle can be the destinations of recycles)
            for line in self.state_simulation["list_raw_lines"]:
                if line["total_flowrate"] > 0 and sum(line["OHE_unit"]) == 0:
                    num_open_streams = num_open_streams + 1

                # count closed streams, which are no recycles themself
                # (these can be the destination of a possible recycle)
                if line["total_flowrate"] > 0 and sum(line["OHE_unit"]) == 1 and line["OHE_unit"][
                    -1] == 0:
                    num_closed_non_rec_streams = num_closed_non_rec_streams + 1

            # a mixer mixes always two open streams (at least we define it this way in this env)
            # therefore, two open streams are at least needed to be able to place a mixer
            if num_open_streams > 1:
                feasible_actions[-2] = 1

            if self.config.allow_recycle:
                # a recycle can only be placed, if at least one open stream remains
                # and there is a possible destination
                if num_open_streams > 1 and num_closed_non_rec_streams > 0:
                    feasible_actions[-1] = 1

        # at level 2, the destination of mixers / recycles are chosen
        elif current_level == 2:
            # now it is important to know, which unit (ie mixer or rec) was chosen
            # chosen_unit is an index (either num.units -2 or -1 (mix, rec))
            if not self.config.allow_recycle and chosen_unit == self.config.num_units - 1:
                # this should never be reached
                return None, None

            # if legacy is True, we have a max number of lines
            if self.config.legacy:
                feasible_actions = np.zeros(self.config.num_lines_flowsheet_matrix)

            else:
                feasible_actions = np.zeros(len(self.state_simulation["list_line_information"]))

            # mixer
            if chosen_unit == self.config.num_units - 2:
                # search open streams (possible destinations of mixers) and
                for line_index, line in enumerate(self.state_simulation["list_raw_lines"]):
                    # open streams
                    if line["total_flowrate"] > 0 and sum(line["OHE_unit"]) == 0:
                        feasible_actions[line_index] = 1

            # rec
            elif chosen_unit == self.config.num_units - 1:
                # search closed streams, which are no recycles themself (possible destinations of recycles)
                for line_index, line in enumerate(self.state_simulation["list_raw_lines"]):
                    # closed, no rec
                    if line["total_flowrate"] > 0 and sum(line["OHE_unit"]) == 1 and line["OHE_unit"][-1] == 0:
                        feasible_actions[line_index] = 1

            else:
                # should never be reached
                return None, None, None

            # the previously chosen stream cannot be a destination
            feasible_actions[chosen_stream] = 0

        # level 2b in reality for cont specifications, always everything is feasible
        elif current_level == 3:
            feasible_actions = np.ones(self.config.num_disc_steps_2b_1)

        elif current_level == 4:
            feasible_actions = np.ones(self.config.num_disc_steps_2b_2)

        if self.config.legacy:
            # corrections, if level is 0 or 2
            if current_level == 0:
                trafo_feasible = np.zeros(self.config.num_lines_flowsheet_matrix + 1)
                trafo_feasible[:len(feasible_actions) - 1] = feasible_actions[:-1]
                trafo_feasible[-1] = feasible_actions[-1]
                feasible_actions = trafo_feasible

            if current_level == 2:
                trafo_feasible = np.zeros(self.config.num_lines_flowsheet_matrix)
                trafo_feasible[:len(feasible_actions)] = feasible_actions
                feasible_actions = trafo_feasible

        return feasible_actions

    def create_line_vector_for_legacy_matrix(self, line_index):
        relevant_dict = self.state_simulation["list_line_information"][line_index]
        keys = list(relevant_dict.keys())
        key = keys[0]
        vector = relevant_dict[key]

        for key in keys[1:]:
            vector = np.concatenate((vector, relevant_dict[key]), axis=None)

        # concatenate destination vector in the end
        destination_vec = np.zeros(self.config.num_lines_flowsheet_matrix)
        destination_vec[:len(self.state_simulation["list_line_information"])] = \
            self.state_simulation["lines_connectivity_matrix"][line_index]

        # destination vector, bool syn complete and line used added to vector
        vector = np.concatenate((vector, destination_vec, self.synthesis_completed, 1), axis=None)

        return vector

    def create_legacy_flowsheet_matrix(self):
        # if one wants to use the old model, the old flowsheet matrix is needed
        matrix = np.zeros((self.config.num_lines_flowsheet_matrix, self.config.line_length_flowsheet_matrix))

        for i, _ in enumerate(self.state_simulation["list_line_information"]):
            matrix[i] = self.create_line_vector_for_legacy_matrix(i)

        return matrix

    def compute_npv(self):
        """
        we have different variants for npv:
            legacy: here we obtained some good results in first tests
            generic: generic npv, advanced version of legacy
            literature: closer to real npvs

        Returns a tuple (float, Optional[float]), where the first entry is the actual NPV, and
        the second entry is a normalized version of it w.r.t. to optimal (albeit non-reachable) results
        (i.e., the value of the pure streams).
        """
        mass_balance_fulfilled, _ = check_mass_balance(self.tree_structure)
        if not mass_balance_fulfilled:
            print("\n\nthis should never happen")
            print(self.blueprint)
            print(self.blueprint["move_seq"], "\n\n")
            return None, None

        # to determine if a stream is empty
        epsilon_for_flowrates = 0.0001

        # for performance ratio, not important for training etc
        sum_n_leaving = 0
        sum_n_solvent_added = 0

        if self.config.npv_version == "legacy":
            solvent_weight = 10
            solvent_added = 0
            solvent_released = 0
            npv = -10 * self.steps
            for line_index, line in enumerate(self.state_simulation["list_raw_lines"]):
                # check if the stream is an open stream, i.e. leaving the process, by checking if an
                # apparatus is connected to it
                if sum(line["OHE_unit"]) == 0:
                    # if the stream is non empty, its win/loss is computed
                    if line["total_flowrate"] > epsilon_for_flowrates:
                        # first check the non-solvents
                        relevant_mfr = line["molar_fractions"][:self.num_comps_in_feed]
                        if np.max(relevant_mfr) > 0.95:
                            weight = 10
                            if np.max(relevant_mfr) > 0.99:
                                weight = 1000

                            npv = npv + (weight * line["total_flowrate"])

                        # in some situations a solvents was maybe used
                        if self.num_comps_in_feed < self.config.max_number_of_components:
                            # if it is a solvent and pure, it is at least some value
                            solvent_max = np.max(line["molar_fractions"][self.num_comps_in_feed:])
                            if solvent_max > 0.99:
                                solvent_released = solvent_released + (solvent_weight * line["total_flowrate"])

                # if a solvent is added, we subtract the solvent input
                elif self.tree_structure.lines[line_index].apparatus.apparatus_type_name == "add_solvent":
                    inp_flowrates = self.tree_structure.accumulate_line_input(line_index)
                    out_flowrates = self.tree_structure.compute_apparatus_output(
                        self.tree_structure.lines[line_index].apparatus, inp_flowrates)

                    difference = out_flowrates[0] - inp_flowrates
                    difference.clip(min=0)

                    solvent_added = solvent_added + (np.sum(difference) * solvent_weight)

            npv = npv - solvent_added + solvent_released
            normed_npv = None

        elif self.config.npv_version == "generic":
            gain_leaving_stream = 0
            cost_units = 0
            cost_solvent_added = 0
            gain_solvent_released = 0
            weight_pure_component = 1000
            weight_solvent = 100
            specification_pure = 0.99
            specification_solvent = 0.99
            capital_costs = {"distillation_column": 10}
            capital_costs["decanter"] = capital_costs["distillation_column"] / 5
            capital_costs["split"] = capital_costs["decanter"] / 10
            capital_costs["add_solvent"] = capital_costs["decanter"] / 10
            capital_costs["mixer"] = capital_costs["decanter"] / 10
            capital_costs["recycle"] = capital_costs["decanter"] / 10

            for line_index, line in enumerate(self.state_simulation["list_raw_lines"]):
                # check if the stream is an open stream, i.e. leaving the process, by checking if an
                # apparatus is connected to it
                if sum(line["OHE_unit"]) == 0:
                    # if the stream is non empty, its win/loss is computed
                    if line["total_flowrate"] > epsilon_for_flowrates:
                        # check non-solvents
                        relevant_mfr = line["molar_fractions"][:self.num_comps_in_feed]
                        if np.max(relevant_mfr) > specification_pure:
                            gain_leaving_stream = gain_leaving_stream + \
                                (weight_pure_component * line["total_flowrate"] * np.max(relevant_mfr))

                            sum_n_leaving = sum_n_leaving + line["total_flowrate"]

                        # in some situations a solvents was maybe used
                        if self.num_comps_in_feed < self.config.max_number_of_components:
                            # if it is a solvent and pure, it is at least some value
                            solvent_max = np.max(line["molar_fractions"][self.num_comps_in_feed:])
                            if solvent_max > specification_solvent:
                                gain_solvent_released = gain_solvent_released + \
                                    (weight_solvent * line["total_flowrate"] * solvent_max)

                                sum_n_leaving = sum_n_leaving + line["total_flowrate"]

                # if an apparatus is connected, we get the capital costs
                else:
                    unitname_connected = self.tree_structure.lines[line_index].apparatus.apparatus_type_name
                    cap_cost_unit = capital_costs[unitname_connected]
                    cost_units = cost_units + cap_cost_unit

                    # if a solvent is added, we subtract the solvent input
                    if unitname_connected == "add_solvent":
                        inp_flowrates = self.tree_structure.accumulate_line_input(line_index)
                        out_flowrates = self.tree_structure.compute_apparatus_output(
                            self.tree_structure.lines[line_index].apparatus, inp_flowrates)

                        difference = out_flowrates[0] - inp_flowrates
                        difference.clip(min=0)

                        cost_solvent_added = cost_solvent_added + (np.sum(difference) * weight_solvent)

                        sum_n_solvent_added = sum_n_solvent_added + difference[-1]

            sum_n_feed = 0
            for feed_stream in self.blueprint["initial_information"]["list_feed_streams"]:
                sum_n_feed = sum_n_feed + np.sum(feed_stream)

            self.performance_ratio = sum_n_leaving / (sum_n_feed + sum_n_solvent_added)

            # add up in npv
            npv = 0
            self.npv_without_app_cost = 0

            # gain leaving streams
            npv = npv + gain_leaving_stream
            self.npv_without_app_cost = self.npv_without_app_cost + gain_leaving_stream

            # solvents
            npv = npv - cost_solvent_added + gain_solvent_released
            self.npv_without_app_cost = self.npv_without_app_cost - cost_solvent_added + gain_solvent_released

            # units
            npv = npv - cost_units
            normed_npv = max(0, npv) / 1000.

        elif self.config.npv_version == "literature":
            # cost function based on shi2015, chen2015, originally for water, pyridine, toluene
            # of course can be used for other examples as well
            # we calculate npv after 10 years
            years = 10
            hr_per_year = 8000

            # capital cost apparatus models, scaling later according to the power rule
            # those values originate for 1000 kmol/hr, 0.1 pyridine, 0.9 water, this roughly corresponds
            # to 25 000 kg/hr. In our state we have molar flowrates in Mmol / hr (in this cost case),
            # therefore, we always start with a feed of 1000 kmol/hr.
            capital_costs = {"distillation_column": 1000000}
            capital_costs["decanter"] = 200000

            # cost for the 'simpler' apparatus models are neglected
            capital_costs["split"] = 0
            capital_costs["add_solvent"] = 0
            capital_costs["mixer"] = 0
            capital_costs["recycle"] = 0

            # to some up cost/gain for solvents
            kg_solvent_added_per_hr = 0
            kg_solvent_released_per_hr = 0

            # prices, specifications
            price_pure_component = 0.5  # €/kg
            price_solvent = 0.05  # €/kg
            price_steam = 0.04  # €/kg
            specification_pure = 0.99
            specification_solvent = 0.99
            gain_leaving_stream = 0
            cost_units = 0

            for line_index, line in enumerate(self.state_simulation["list_raw_lines"]):
                # flowrate in Mmol / hr, convert it to kg / hr
                flowrates_kg = self.convert_mol_flow_to_kg(line["total_flowrate"] * line["molar_fractions"],
                                                           factor_mol=1000000)
                total_flowrate_kg = sum(flowrates_kg)

                # check if the stream is an open stream, i.e. leaving the process, by checking if an
                # apparatus is connected to it
                if sum(line["OHE_unit"]) == 0:
                    # if the stream is non empty, its win/loss is computed
                    if line["total_flowrate"] > epsilon_for_flowrates:
                        mass_fractions = flowrates_kg / total_flowrate_kg

                        # check non-solvents
                        relevant_mass_fr = mass_fractions[:self.num_comps_in_feed]
                        if np.max(relevant_mass_fr) > specification_pure:
                            # we assume 10 years with 8000 hr / a
                            gain_leaving_stream = gain_leaving_stream + \
                                (price_pure_component * total_flowrate_kg * np.max(relevant_mass_fr)\
                                 * years * hr_per_year)

                            sum_n_leaving = sum_n_leaving + line["total_flowrate"]

                        # in some situations a solvents was maybe used
                        if self.num_comps_in_feed < self.config.max_number_of_components:
                            # if it is a solvent and pure, it is at least some value
                            solvent_max = np.max(mass_fractions[self.num_comps_in_feed:])
                            if solvent_max > specification_solvent:
                                kg_solvent_released_per_hr = kg_solvent_released_per_hr + \
                                                             (total_flowrate_kg * solvent_max)

                                sum_n_leaving = sum_n_leaving + line["total_flowrate"]

                # if an apparatus is connected, we get the capital costs
                else:
                    unitname_connected = self.tree_structure.lines[line_index].apparatus.apparatus_type_name
                    cap_cost_unit = capital_costs[unitname_connected]

                    # scaling according to power rule (if a stream is almost empty, we just assume that one still
                    # has to pay the standard capital cost
                    if line["total_flowrate"] > epsilon_for_flowrates:
                        # scale with 25 000 kg / hr
                        cap_cost_unit = cap_cost_unit * np.power(total_flowrate_kg / 25000, 0.6)

                    cost_units = cost_units + cap_cost_unit

                    # if a solvent is added, we subtract the solvent input
                    if unitname_connected == "add_solvent":
                        inp_flowrates = self.tree_structure.accumulate_line_input(line_index)
                        out_flowrates = self.tree_structure.compute_apparatus_output(
                            self.tree_structure.lines[line_index].apparatus, inp_flowrates)

                        difference = out_flowrates[0] - inp_flowrates
                        difference.clip(min=0)

                        sum_n_solvent_added = sum_n_solvent_added + difference[-1]

                        # a solvent can only be in the third entry
                        solvent_molar_flowrates = np.zeros(3)
                        solvent_molar_flowrates[-1] = difference[-1]
                        solvent_kg_total_flowrate = sum(self.convert_mol_flow_to_kg(solvent_molar_flowrates,
                                                                                    factor_mol=1000000))

                        kg_solvent_added_per_hr = kg_solvent_added_per_hr + solvent_kg_total_flowrate

                    # get operating costs for d column
                    elif unitname_connected == "distillation_column":
                        # get outputs of column
                        inp_flowrates = self.tree_structure.accumulate_line_input(line_index)
                        out_flowrates = self.tree_structure.compute_apparatus_output(
                            self.tree_structure.lines[line_index].apparatus, inp_flowrates)

                        distillate_molar_flowrates = out_flowrates[0]

                        # as in previous publications, we assume 2 times recycle/distillation of top product
                        heat_per_hr = 0
                        for name_inde_j, name_inde in enumerate(self.tree_structure.current_indices):
                            name = self.config.phase_eq_generator.names_components[name_inde]
                            factor = self.config.dict_pure_component_data[name]["factor_heat_estimation_J_per_mol"]

                            # factor has unit J / mol, molar flowrate has unit Mmol/hr, heat
                            heat_per_hr = heat_per_hr + (2 * factor * distillate_molar_flowrates[name_inde_j] * 1000000)

                        # we have heat flow in J / hr, we divide this now by water heat of vap
                        # (start at 25 Celsius with liq water, c_p given, then go to 100 Celsius then add dhv)
                        mol_water_to_be_heated_per_hr = heat_per_hr / \
                            self.config.dict_pure_component_data["water"]["factor_heat_estimation_J_per_mol"]

                        # get kg / hr
                        kg_water_to_be_heated_per_hr = mol_water_to_be_heated_per_hr * \
                            self.config.dict_pure_component_data["water"]["M"] / 1000

                        operating_cost_per_hr = kg_water_to_be_heated_per_hr * price_steam

                        # 8000 hr/a, 10 years
                        cost_units = cost_units + (operating_cost_per_hr * years * hr_per_year)

            sum_n_feed = 0
            for feed_stream in self.blueprint["initial_information"]["list_feed_streams"]:
                sum_n_feed = sum_n_feed + np.sum(feed_stream)

            self.performance_ratio = sum_n_leaving / (sum_n_feed + sum_n_solvent_added)

            # add up in npv
            npv = 0
            self.npv_without_app_cost = 0

            # gain leaving streams
            npv = npv + gain_leaving_stream
            self.npv_without_app_cost = self.npv_without_app_cost + gain_leaving_stream

            # solvent npv, we assume 10 years with 8000 hr / a
            npv = npv + ((kg_solvent_released_per_hr - kg_solvent_added_per_hr) * price_solvent * years * hr_per_year)
            self.npv_without_app_cost = self.npv_without_app_cost + \
                ((kg_solvent_released_per_hr - kg_solvent_added_per_hr) * price_solvent * years * hr_per_year)

            # units
            npv = npv - cost_units

            theoretical_max_npv = 0
            # convert it to kg / hr
            for feed_stream in self.blueprint["initial_information"]["list_feed_streams"]:
                flowrates_kg = self.convert_mol_flow_to_kg(feed_stream, factor_mol=1000000)
                theoretical_max_npv = theoretical_max_npv +\
                    price_pure_component * sum(flowrates_kg) * years * hr_per_year

            normed_npv = np.max([npv, 0]) / theoretical_max_npv

            # scaled to M€
            npv = npv / 1000000
            self.npv_without_app_cost = self.npv_without_app_cost / 1000000

        return npv, normed_npv

    def convert_mol_flow_to_kg(self, flowrates_mol, factor_mol):
        flowrates_kg = np.zeros(len(flowrates_mol))
        for j, index in enumerate(self.tree_structure.current_indices):
            name = self.config.phase_eq_generator.names_components[index]
            molar_mass = self.config.dict_pure_component_data[name]["M"]

            # molar mass is in g/mol
            flowrates_kg[j] = molar_mass * flowrates_mol[j] * factor_mol / 1000

        return flowrates_kg

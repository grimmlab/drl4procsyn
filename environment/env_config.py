import numpy as np
import os
import copy

import environment.units as units
import environment.phase_equilibria.phase_eq_handling as phase_eq_generation
import environment.flowsheet_simulation as flowsheet_simulation


class EnvConfig:
    """
    Config class with all parameters needed for the environment.
    """
    def __init__(self):
        # limit for full actions (=placing unit) per feed stream
        self.max_steps_for_flowsheet_synthesis = 10

        # the maximum number of components in a flowsheet (also there will
        # be not more molar fractions than this number in the line information)
        self.max_number_of_components = 3

        # initialize class for phase eq and property data
        # the parameters specify the discretization levels of the lles and vles
        self.systems_allowed = {
            "acetone_chloroform": True,
            "ethanol_water": True,
            "n-butanol_water": True,
            "water_pyridine": True
        }
        self.dicretization_parameter_lle = 5
        self.curvature_parameter_vle = 0.001
        self.phase_eq_generator = phase_eq_generation.PhaseEqHandling(
            directory=os.path.join(os.getcwd(), "environment", "phase_equilibria"),
            systems_allowed=self.systems_allowed)
        self.phase_eq_generator.load_phase_eqs(num_comp_lle=self.max_number_of_components,
                                               disc_para_lle=self.dicretization_parameter_lle,
                                               curvature_parameter=self.curvature_parameter_vle)

        # options: "legacy", "generic", "literature".
        self.npv_version = "generic"
        self.norm_npv = True
        self.scale_pseudo_mass = 0.01

        # limit for size of recycle streams, compared to feed input
        # a recycle entry is allowed to be x times as big as the respective feed entry
        # for solvents the limit is x times the entire feed stream
        self.limit_recycle_size = 25

        # add solvent is allowed as action just a limited number of times
        self.num_add_solvent_actions = 1

        # this dict contains for each name the vector with information on the pure comp
        # and also information on how this component interacts with the other components
        self.dict_pure_component_data = self.phase_eq_generator.load_pure_component_data()

        # this many units have to be placed until termination action is allowed (if no other
        # action would be feasible, termination will be always possible)
        self.termination_possible_from_step = 0

        # keys are the types, items are the number of different subtypes (e.g., more choices for
        # add_solvent).
        self.unit_types = {"distillation_column": {"num": 1, "output_streams": 2, "cont_range": [0, 1],
                                                   "next_level": 3},
                           "split": {"num": 1, "output_streams": 2, "cont_range": [0, 1],
                                     "next_level": 3},
                           "decanter": {"num": 1, "output_streams": 2, "cont_range": None,
                                        "next_level": 0},
                           "add_solvent": {"num": len(self.phase_eq_generator.names_components),
                                           "output_streams": 1, "cont_range": [0, 10],
                                           "next_level": 3},
                           "mixer": {"num": 1, "output_streams": 1, "cont_range": None, "next_level": 2},
                           "recycle": {"num": 1, "output_streams": 1, "cont_range": None, "next_level": 2}}

        # given index, get a unit name
        self.units_map_indices_type = []
        for key in self.unit_types.keys():
            for i in range(self.unit_types[key]["num"]):
                self.units_map_indices_type.append(key)

        # distillation column, split, decanter, add_component, mixer, recycle
        # for distillation, split and add_component a continuous parameter is required
        # add component is a separate apparatus for each component
        self.num_units = len(self.units_map_indices_type)

        # index ranges of units (start inclusive, end exclusive)
        self.units_indices_ranges = {}
        current_key = self.units_map_indices_type[0]
        start_ind = 0
        for i, key in enumerate(self.units_map_indices_type):
            if current_key != key:
                self.units_indices_ranges[current_key] = [start_ind, i]
                current_key = key
                start_ind = i

        # recycle is last
        self.units_indices_ranges[current_key] = [start_ind, self.num_units]

        for i, key in enumerate(self.units_map_indices_type):
            if key == "add_solvent":
                # to determine, which solvent is chosen
                self.add_solvent_start_index = i
                break

        # store cont ranges
        self.ranges_cont_specification = [self.unit_types[key]["cont_range"] for key in
                                          self.units_map_indices_type]

        # allow special actions like solvent / rec / split
        self.allow_split = False
        self.allow_solvent = True
        self.allow_recycle = True

        # fixed order of components in respective situations or shuffle
        self.shuffle_order_of_components = False

        # specifications in the vectors regarding line information for the state matrix
        self.keys_for_unit_spec_vector = ["is_feed", "cont_spec", "OHE_cont_spec"]

        # to get the vector length for the transformer, initialize an example simulation and get the length
        example_feed = self.create_random_problem_instance()
        example_simulation = flowsheet_simulation.FlowsheetSimulation(example_feed, self)

        # in the following we define variables, which describe the length of the inputs for the transformer or
        # describe, which of the other variables in this config can be used for this purpose
        self.len_input_wunit = self.num_units
        self.len_input_wstream = len(example_simulation.state_simulation["list_line_information"][0][
                                         "input_w_stream"])
        self.len_input_wunitspecification = len(example_simulation.state_simulation["list_line_information"][0][
                                                "input_w_unit_specification"])

        # this class is sometimes needed inside the environment
        self.distillation_column = units.distillation_column()

        # from now on continuous parameters are discretized into two levels (2b_1, 2b_2):
        # on the first level, the whole range is discretized into equal intervals
        self.num_disc_steps_2b_1 = 7
        # the second level discretizes each of those intervals once again
        self.num_disc_steps_2b_2 = 7

        # calculate the increments from the ranges per level (those are then used in
        # the game class to get the specific continuous action)
        self.increments_per_unit = [None] * self.num_units

        # if a continuous parameter is needed, a dict is placed for None
        for i, cont_range in enumerate(self.ranges_cont_specification):
            if cont_range is not None:
                increment_dict = {}
                # 2b_1: action (index starting from 0) * increment specifies first part of
                # cont parameter
                increment_2b_1 = (cont_range[1] - cont_range[0]) / self.num_disc_steps_2b_1
                increment_dict["level_2b_1"] = increment_2b_1

                # 2b_2: (action_2b_1 * increment_2b_1) + (action_2b_2 * increment_2b_2)
                # gives full action
                increment_2b_2 = increment_2b_1 / self.num_disc_steps_2b_2
                increment_dict["level_2b_2"] = increment_2b_2

                self.increments_per_unit[i] = increment_dict

        # when a recycle is placed, a root finding problem is solved, as initial values serve the current
        # streams of the process, zeros and randomly sampled numbers. random_guesses_root_iteration defines
        # how often, a random sample is tried as initial guess
        self.random_guesses_root_iteration = 0
        # When recycle is placed, maximum number of iterations which may be used by `fsolve` to find the root.
        # set to zero for default behaviour of fsolve
        self.max_num_root_finding_interactions = 50

        # bool, which tells, if also wegstein iteration should be used to solve recycles
        self.use_wegstein = False
        # x_n+1 = wegstein_constant * x_n + (1 - wegstein_constant) * f(x_n)
        self.wegstein_constant = 0.5
        self.wegstein_steps = 500

        # convergence criteria for loops
        self.epsilon = 0.001

        # for some tests we want the old model and environment (no transformer matrix state)
        # alert: the old model cannot really do anything useful as the information in the state is
        # not complete (e.g., no information on the present components)
        self.legacy = True

        if not self.legacy:
            self.num_actions_per_level = [
                None,  # Level 0, discrete, num of lines in the flowsheet matrix + 1
                self.num_units,  # Level 1, discrete
                None,  # Level 2a, destination of mixer/recycle, as level 0
                self.num_disc_steps_2b_1,  # Level 2b_1, number of discrete intervals for continuous action, part_1
                self.num_disc_steps_2b_2  # Level 2b_2, number of discrete intervals for continuous action, part_2
            ]

            self.num_levels = len(self.num_actions_per_level)

        else:
            self.num_lines_flowsheet_matrix = (self.max_steps_for_flowsheet_synthesis * 2) + 3

            # get vector length
            self.line_length_flowsheet_matrix = len(example_simulation.create_line_vector_for_legacy_matrix(0))

            self.num_actions_per_level = [
                self.num_lines_flowsheet_matrix + 1,  # Level 0, discrete, num of lines in the flowsheet matrix + 1
                self.num_units,  # Level 1, discrete
                self.num_lines_flowsheet_matrix,  # Level 2a, destination of mixer/recycle, as level 0
                self.num_disc_steps_2b_1,  # Level 2b_1, number of discrete intervals for continuous action, part_1
                self.num_disc_steps_2b_2  # Level 2b_2, number of discrete intervals for continuous action, part_2
            ]

            self.num_levels = len(self.num_actions_per_level)

    def create_random_problem_instance(self):
        """
        sample a feed situation of format: [[indices from self.names_components for feed],
        [indices from self.names_components for add_component unit], number of feed streams]
        and return the situation index, feed streams, restrictions for add component unit and
        the names and order of the components in the streams
        """
        feed_streams = []

        # sample a feed situation
        sampled_index = np.random.randint(len(self.phase_eq_generator.feed_situations))
        sampled_situation = copy.deepcopy(self.phase_eq_generator.feed_situations[sampled_index])

        # shuffle order if specified
        if self.shuffle_order_of_components:
            np.random.shuffle(sampled_situation[0])

        # get names in feed streams
        names_in_streams = []
        for i in sampled_situation[0]:
            names_in_streams.append(self.phase_eq_generator.names_components[i])

        for i in range(sampled_situation[-1]):
            sampled_flowrates = np.random.rand(len(sampled_situation[0]))
            # normalize to 1 total flowrate
            sampled_flowrates = sampled_flowrates / (
                    sampled_situation[-1] * sum(sampled_flowrates))

            stream = np.zeros(self.max_number_of_components)
            stream[:len(sampled_flowrates)] = sampled_flowrates
            feed_streams.append(stream)

        return {"feed_situation_index": sampled_index,
                "indices_components_in_feeds": sampled_situation[0],
                "list_feed_streams": feed_streams,
                "possible_ind_add_comp": sampled_situation[1],
                "comp_order_feeds": names_in_streams,
                "lle_for_start": None,
                "vle_for_start": None}

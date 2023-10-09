import os

import numpy as np

import environment.phase_equilibria.thermo_models as thermo_models
import environment.phase_equilibria.point_discretization as point_discretization
import environment.phase_equilibria.lle as lle
import environment.phase_equilibria.property_data as property_data

from environment.units import ternary_vle as ternary_vle


class PhaseEqHandling:
    def __init__(self, directory, systems_allowed):
        """
        param systems_allowed: dict with bools, which indicate, if a system is included
            or not
        """
        # directories above data
        self.directory = directory

        # name conversion for property data
        self.convert_names_dict = {
            "WATER": "water", "ETHANOL": "ethanol", "ACETONE": "acetone", "CHLOROFO": "chloroform",
            "BENZE-01": "benzene", "TOLUE-01": "toluene", "1-BUT-01": "1-butanol",
            "TETRA-01": "tetrahydrofuran", "PYRID-01": "pyridine", "N-BUT-01": "n-butanol"
        }

        # property data
        self.interaction_data = property_data.NRTLParameters(path=os.path.join(
            self.directory, "data", "source_excel", "parameters.xlsx"),
            name_converter_dict=self.convert_names_dict)
        self.antoine_data = property_data.AntoineParameters(path=os.path.join(
            self.directory, "data", "source_excel", "parameters.xlsx"),
            name_converter_dict=self.convert_names_dict)
        self.additional_parameters = property_data.AdditionalParameters(path=os.path.join(
            self.directory, "data", "source_excel", "parameters.xlsx"),
            name_converter_dict=self.convert_names_dict)

        # here one has to append lists of the form:
        # [[names in feed(s)], [names for add comp], num_feeds, temp (K), pressure (bar)]
        # the rest will be created automatically
        self.pre_list_feed_situations = []
        if systems_allowed["acetone_chloroform"]:
            self.pre_list_feed_situations.append([["acetone", "chloroform"],
                                                  ["benzene", "toluene"],
                                                  1, 50 + 273.15, 1.013])

        if systems_allowed["ethanol_water"]:
            self.pre_list_feed_situations.append([["ethanol", "water"],
                                                  ["benzene", "toluene", "tetrahydrofuran"],
                                                  1, 65 + 273.15, 1.013])

        if systems_allowed["n-butanol_water"]:
            self.pre_list_feed_situations.append([["n-butanol", "water"],
                                                  ["acetone", "benzene", "toluene"],
                                                  1, 65 + 273.15, 1.013])

        if systems_allowed["water_pyridine"]:
            self.pre_list_feed_situations.append([["water", "pyridine"],
                                                  ["toluene"],
                                                  1, 65 + 273.15, 1.013])

        # gather components that can occur, the index in this list serves as identifier
        # the list will be sorted alphabetically.
        self.names_components = []
        self.pre_list_subsystems = []
        for pre_list in self.pre_list_feed_situations:
            # get feed components
            for feed_comp in pre_list[0]:
                if feed_comp not in self.names_components:
                    self.names_components.append(feed_comp)

            if len(pre_list[0]) == 3:
                self.pre_list_subsystems.append([pre_list[0], pre_list[-2], pre_list[-1]])

            for add_comp in pre_list[1]:
                if add_comp not in self.names_components:
                    self.names_components.append(add_comp)

                self.pre_list_subsystems.append([pre_list[0] + [add_comp], pre_list[-2], pre_list[-1]])

        self.names_components.sort()
        print("components included in environment: ", self.names_components, "\n")

        # in pre_list_subsystems we have all subsystems, where we need a vle and lle
        # we want to ensure that there are no double
        # we order the components inside the subsystems according to their index in the
        # names list, this will make it easier later to identify them
        self.subsystems_indices = []
        self.subsystems_names = []
        self.subsystems_pressures = []
        self.subsystems_temperatures = []
        for i in range(len(self.pre_list_subsystems)):
            indices = []
            for name in self.pre_list_subsystems[i][0]:
                indices.append(self.names_components.index(name))

            indices.sort()

            # check if this subsystem is already there
            already_there = False
            for existing_subsystem in self.subsystems_indices:
                if sum(np.abs(np.array(existing_subsystem) - np.array(indices))) < 0.5:
                    already_there = True
                    break

            if not already_there:
                self.subsystems_indices.append(indices)

                name = ""
                for n in self.subsystems_indices[-1][:-1]:
                    name = name + self.names_components[n] + "_"

                name = name + self.names_components[self.subsystems_indices[-1][-1]]
                self.subsystems_names.append(name)
                self.subsystems_temperatures.append(self.pre_list_subsystems[i][-2])  # kelvin
                self.subsystems_pressures.append(self.pre_list_subsystems[i][-1])  # bar

        print("considered subsystems: ", len(self.subsystems_names), self.subsystems_names)

        # list of lists, elements of the form: [[indices from self.names_components for feed],
        # [indices from self.names_components for add_component unit], number of feed streams]
        self.feed_situations = []
        for pre_list in self.pre_list_feed_situations:
            self.feed_situations.append(self.create_feed_stream_situation(
                names_for_feed=pre_list[0],
                names_for_add_component=pre_list[1],
                num_feed_streams=pre_list[2]))

        print("feed situations: ", self.feed_situations, "\n")

        # storage for lles, vles
        self.liquid_liquid_equilibria = {}
        self.vapor_liquid_equilibria = {}

    def load_pure_component_data(self):
        """
        return dictionary with component names as keys and each item is a list with
        the critical temperature (K), critical pressure (bar) and acentric factor

        also compute for all components the latent heat of vaporization according to clausius-clapeyron,
        using antoine models
        """
        dict_to_return = {}
        for name in self.names_components:
            # for each name a dict with critical data
            dict_to_return[name] = {}
            dict_to_return[name]["critical_data"] = self.antoine_data.get_crit_parameters(name)

            # heat of vaporization
            antoine_paras = self.antoine_data.get_parameters([name])[0]
            antoine_model = thermo_models.Antoine(antoine_paras)
            boiling_point = self.additional_parameters.parameters[name]["Tb"]

            dict_to_return[name]["M"] = self.additional_parameters.parameters[name]["M"]
            dict_to_return[name]["Tb"] = boiling_point
            dict_to_return[name]["dhv"] = antoine_model.compute_latent_heat_vaporization(
                temperature_K=boiling_point)

            # when we later on estimate the heat in the distillation column, we assume we have to heat
            # a certain liquid feed from 25 Celsius to the respective boiling point of the pure comps
            # proper explanation is given in npv of flowsheet simulation
            difference_to_boiling_point = boiling_point - 298.15
            # if the comp is already in the vapor phase, we assume there is no heat to be provided
            if difference_to_boiling_point < 0:
                dict_to_return[name]["factor_heat_estimation_J_per_mol"] = 0

            else:
                molar_heat_capacity = self.additional_parameters.parameters[name]["c_p"]
                dict_to_return[name]["factor_heat_estimation_J_per_mol"] = \
                    (molar_heat_capacity * difference_to_boiling_point) + dict_to_return[name]["dhv"]

        return dict_to_return

    def compute_inf_dilution_act_coeffs(self, name_1, name_2, temperature):
        """
        for name pair [name_1, name_2], we get an array with gamma_1^inf, gamma_2^inf
        """
        # matrices = [a, b, e, f, alpha]
        matrices, _ = self.interaction_data.get_parameters([name_1, name_2])

        model = thermo_models.NRTL(alpha_matrix=matrices[-1], a_matrix=matrices[0],
                                   b_matrix=matrices[1], e_matrix=matrices[2], f_matrix=matrices[3])

        gamma_1_inf = model.compute_activity_coefficient(molar_fractions=np.array([0, 1]), index=0,
                                                         temperature=temperature)
        gamma_2_inf = model.compute_activity_coefficient(molar_fractions=np.array([1, 0]), index=1,
                                                         temperature=temperature)

        return np.array([gamma_1_inf, gamma_2_inf])

    def create_feed_stream_situation(self, names_for_feed, names_for_add_component, num_feed_streams):
        indices_for_feed = []
        for name in names_for_feed:
            indices_for_feed.append(self.names_components.index(name))

        indices_for_add_component = []
        for name in names_for_add_component:
            indices_for_add_component.append(self.names_components.index(name))

        return [indices_for_feed, indices_for_add_component, num_feed_streams]

    def load_phase_eqs(self, num_comp_lle, disc_para_lle, curvature_parameter):
        """
        load all phase eq, which are specified in subsystems
        """
        load_path = os.path.join(self.directory, "data")
        point_disc = point_discretization.PointDisc(
            num_comp=num_comp_lle, recursion_steps=disc_para_lle, load=True, store=False, path=self.directory)

        for index_subsystem, subsystem_ind in enumerate(self.subsystems_indices):
            name = self.subsystems_names[index_subsystem]

            # order [a, b, e, f, alpha]
            nrtl_paras_subsystem, _ = self.interaction_data.get_parameters([
                self.names_components[i] for i in subsystem_ind])

            ge_model = thermo_models.NRTL(alpha_matrix=nrtl_paras_subsystem[-1],
                                          a_matrix=nrtl_paras_subsystem[0],
                                          b_matrix=nrtl_paras_subsystem[1],
                                          e_matrix=nrtl_paras_subsystem[2],
                                          f_matrix=nrtl_paras_subsystem[3])

            # load lle
            self.liquid_liquid_equilibria[name] = {
                "phase_eq": lle.miscibility_analysis(discretized_system=point_disc,
                gE_model=ge_model, temperature=self.subsystems_temperatures[index_subsystem], construct=False,
                path=os.path.join(load_path, "lle", "disc_" + str(disc_para_lle), name), actors_for_para=None),
                "index_subsystem": index_subsystem,
                "indices_components": subsystem_ind
            }

            # load vle
            self.vapor_liquid_equilibria[name] = {
                "phase_eq": ternary_vle(index=index_subsystem, name=name,
                                        path=os.path.join(load_path, "vle",
                                                          "curv_" + str(curvature_parameter))),
                "index_subsystem": index_subsystem,
                "indices_components": subsystem_ind
            }

    def search_subsystem_phase_eq(self, names):
        """
        given a list of names, search the correct vle and lle. if to less names are given (e.g.
        2 names and we have only ternary phase eq), we search for ternaries, which contain all
        given names.
        """
        indices = [self.names_components.index(name) for name in names]
        indices.sort()

        found_index = None
        for j, subsystem_indices in enumerate(self.subsystems_indices):
            contained = [False] * len(indices)
            for i in range(len(indices)):
                if indices[i] in subsystem_indices:
                    contained[i] = True

            if sum(contained) == len(contained):
                found_index = j
                break

        return {"lle": self.liquid_liquid_equilibria[self.subsystems_names[found_index]],
                "vle": self.vapor_liquid_equilibria[self.subsystems_names[found_index]]}

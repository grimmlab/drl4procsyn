import itertools
import numpy as np
import pandas as pd


class NRTLParameters:
    """
    class for storing NRTL parameters for the form
    tau_ij = a_ij + b_ij/T + e_ij*logT + f_ij*T (T in K)
    """
    def __init__(self, path, name_converter_dict):
        # format name_1--name_2
        self.binary_names = []

        self.a = []
        self.b = []
        self.e = []
        self.f = []
        self.alpha = []
        self.t_borders = []  # in kelvin, [t_lower, t_upper]

        source = pd.read_excel(path, header=[0, 1], sheet_name="nrtl")

        # add component interaction paras with correct names
        for name in source.columns[1:]:
            parameters = source[name][3:].array.to_numpy()
            name_1 = name_converter_dict[name[0]]
            name_2 = name_converter_dict[name[1]]

            # kelvin in t borders
            parameters[-1] = parameters[-1] + 273.15
            parameters[-2] = parameters[-2] + 273.15

            self.add_pair(name_1, name_2, alpha=np.ones(2) * parameters[4],
                          t_borders=parameters[10:],
                          a=parameters[:2], b=parameters[2:4],
                          e=parameters[6:8], f=parameters[8:10])

    def add_pair(self, name_1, name_2, alpha, t_borders, a=np.zeros(2), b=np.zeros(2),
                 e=np.zeros(2), f=np.zeros(2)):
        self.binary_names.append(str(name_1 + "--" + name_2))
        self.a.append(a)
        self.b.append(b)
        self.e.append(e)
        self.f.append(f)
        self.alpha.append(alpha)
        self.t_borders.append(t_borders)

    def get_binary_interaction_para(self, name_1, name_2):
        # names are strings
        for ind, name in enumerate(self.binary_names):
            if str(name_1 + "--" + name_2) == str(name):
                return [self.a[ind], self.b[ind], self.e[ind], self.f[ind], self.alpha[ind],
                        self.t_borders[ind]]

            elif str(name_2 + "--" + name_1) == str(name):
                ret = [self.a[ind], self.b[ind], self.e[ind], self.f[ind], self.alpha[ind]]
                return [np.flip(el) for el in ret] + [self.t_borders[ind]]

        return None

    def get_parameters(self, names):
        # get binary interactions
        a = np.zeros((len(names), len(names)))
        b = np.zeros((len(names), len(names)))
        e = np.zeros((len(names), len(names)))
        f = np.zeros((len(names), len(names)))
        alpha = np.zeros((len(names), len(names)))
        t_lower = -1 * float("inf")
        t_upper = float("inf")

        matrices = [a, b, e, f, alpha]

        todo = itertools.combinations(list(range(len(names))), 2)
        for combination in todo:
            paras = self.get_binary_interaction_para(names[combination[0]], names[combination[1]])
            if paras is None:
                return None, None

            else:
                for j in range(len(matrices)):
                    matrices[j][combination[0]][combination[1]] = paras[j][0]
                    matrices[j][combination[1]][combination[0]] = paras[j][1]

                # update t borders
                if paras[-1][0] > t_lower:
                    t_lower = paras[-1][0]

                if paras[-1][1] < t_upper:
                    t_upper = paras[-1][1]

        return matrices, [t_lower, t_upper]


class AntoineParameters:
    """
    class for Antoine parameters of the form
    ln p^s = A1 + A2 / (T + A3) + A4 * T + A5 * lnT + A6 * T^A7 for A8 < T < A9

    units: (T in K, p^s in bar)

    thermo_models antoine gives back the value in bar and we also insert bar into vle_analysis.
    """
    def __init__(self, path, name_converter_dict):
        self.names = []

        # for each name we add an item, i.e. a np.vector of length 9
        self.parameters = {}
        # same structure as parameters, for every name a vector with T_c (C in Excel, transformed to K),
        # p_c (bar), omega
        self.critical_data = {}

        # antoine parameters
        source = pd.read_excel(path, sheet_name="antoine")

        # add components with correct names
        for name in source.columns[1:]:
            conv_name = name_converter_dict[name]
            parameters = source[name][2:11].array.to_numpy()

            # kelvin in t borders
            parameters[-1] = parameters[-1] + 273.15
            parameters[-2] = parameters[-2] + 273.15

            self.add_parameters(conv_name, parameters)

        # critical data
        source = pd.read_excel(path, sheet_name="pure")

        # add components with correct names
        for name in source.columns[3:]:
            conv_name = name_converter_dict[name[10:]]
            parameters = source[name].array.to_numpy()

            critical_temp_celsius = parameters[27]
            critical_temp_kelvin = critical_temp_celsius + 273.15
            critical_pressure_bar = parameters[21]
            omega = parameters[20]

            self.add_crit_parameters(conv_name, np.array([critical_temp_kelvin,
                                                          critical_pressure_bar,
                                                          omega]))

    def add_component_name(self, name):
        # check if name already exists
        already_there = False
        for i, ex_name in enumerate(self.names):
            if ex_name == name:
                already_there = True
                break

        if not already_there:
            self.names.append(name)

    def add_parameters(self, name, parameter_vector):
        # add name, if not already there
        self.add_component_name(name)

        # add parameters
        self.parameters[name] = parameter_vector

    def add_crit_parameters(self, name, parameter_vector):
        self.critical_data[name] = parameter_vector

    def get_parameters(self, names):
        """
        names is a list of arbitrary length, get the parameters in the correct order.
        """
        return [self.parameters[key] for key in names]

    def get_crit_parameters(self, name):
        return self.critical_data[name]


class AdditionalParameters:
    """
    sources are provided in the Excel file. Parts of those are not from Aspen.
    """
    def __init__(self, path, name_converter_dict):
        self.names = []

        # for each name we add an item, i.e. a dictionary containing the parameters
        self.parameters = {}

        # for non Aspen parameters
        source = pd.read_excel(path, sheet_name="parameters_not_from_aspen")

        # add components with correct names
        for name in source.columns[3:]:
            already_there = self.add_component_name(name)
            if already_there:
                print("\n\n\nSomething is wrong with property data\n\n\n")
                return None

            conv_name = name_converter_dict[name[10:]]
            parameters = source[name][:2].array.to_numpy()

            molar_heat_capacity = parameters[0]  # J / (mol * K)
            molar_mass = parameters[1]  # g / mol

            self.parameters[conv_name] = {"c_p": molar_heat_capacity, "M": molar_mass}

        # for boiling points
        source_2 = pd.read_excel(path, sheet_name="pure")
        for name in source_2.columns[3:]:
            conv_name = name_converter_dict[name[10:]]
            parameters = source_2[name].array.to_numpy()

            boiling_point_celsius = parameters[26]
            boiling_point_kelvin = boiling_point_celsius + 273.15

            self.parameters[conv_name]["Tb"] = boiling_point_kelvin

    def add_component_name(self, name):
        # check if name already exists
        already_there = False
        for i, ex_name in enumerate(self.names):
            if ex_name == name:
                already_there = True
                break

        if not already_there:
            self.names.append(name)

        return already_there

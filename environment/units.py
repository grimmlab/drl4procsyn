import numpy as np
import os
import math


def transform_stream_fs_to_stream_phase_eq(
        molar_flowrates_flowsheet, order_components_flowsheet: list,
        phase_eq_order_components: list):
    """
    molar_flowrates_flowsheet is a np.array
    order_components_flowsheet is a list of indices
    phase_eq_order_components is a list of indices
    """
    target_flowrates = np.zeros(len(phase_eq_order_components))

    # for safety
    for i, index in enumerate(order_components_flowsheet):
        if index not in phase_eq_order_components:
            print("\nnot legal transformation")
            print(molar_flowrates_flowsheet)
            print(order_components_flowsheet)
            print(phase_eq_order_components)
            print("\n\n")
            return None

    for i in range(len(target_flowrates)):
        index_to_search = phase_eq_order_components[i]
        if index_to_search in order_components_flowsheet:
            index_in_fs = order_components_flowsheet.index(index_to_search)
            target_flowrates[i] = molar_flowrates_flowsheet[index_in_fs]

    return target_flowrates


def transform_stream_phase_eq_to_stream_fs(
        molar_flowrates_phase_eq, phase_eq_order_components: list,
        order_components_flowsheet: list, max_num_components_flowsheet):
    target_flowrates = np.zeros(max_num_components_flowsheet)

    for i in range(len(order_components_flowsheet)):
        index_to_search = order_components_flowsheet[i]
        index_in_phase_eq = phase_eq_order_components.index(index_to_search)
        target_flowrates[i] = molar_flowrates_phase_eq[index_in_phase_eq]

    return target_flowrates


def add_solvent(feed_molar_flowrates, index_new_component, ratio_component_to_feed):
    """
    to a given feed, mix in a specified ratio a specified component
    """
    flowrate_new_comp = ratio_component_to_feed * sum(feed_molar_flowrates)
    new_feed = np.zeros(len(feed_molar_flowrates))
    new_feed[index_new_component] = new_feed[index_new_component] + flowrate_new_comp
    new_feed = new_feed + feed_molar_flowrates

    return [new_feed]


def split(feed_molar_flowrates, split_ratio):
    """
    to a given feed, the upper output is split_ratio * feed
    """
    if split_ratio < 0 or split_ratio > 1:
        print("\nillegal split ratio")
        print("feed", feed_molar_flowrates)
        print("ratio", split_ratio, "\n")
        return None

    return [split_ratio * feed_molar_flowrates, (1 - split_ratio) * feed_molar_flowrates]


def decantation(transformed_feed_molar_flowrates, current_phase_eq_liq):
    """
    function gets feed and corresponding phase_eq_liq class and
    computes a possible liquid phase split within that class.

    always two streams are returned. in the case of no 2 phase split or if the feed just
    consists of a pure component, we just return the feed splitted into two equal streams.

    we ensure positive flowrates and mass balance here
    """
    # check if the feed stream is a pure component
    component_present = [False] * len(transformed_feed_molar_flowrates)
    for i in range(len(transformed_feed_molar_flowrates)):
        if transformed_feed_molar_flowrates[i] > 0.0001:
            component_present[i] = True

    if sum(component_present) < 2:
        # in this case we just split with a ratio of 0.5
        phase_1_flowrate = 0.5 * transformed_feed_molar_flowrates
        phase_2_flowrate = 0.5 * transformed_feed_molar_flowrates

    else:
        phases_flowrates, _ = current_phase_eq_liq.find_phase_split(
            feed_molar_flowrates=transformed_feed_molar_flowrates,
            relevant_simplex=None, discretized_system=current_phase_eq_liq.discretized_system,
            miscibility_gap_simplices=current_phase_eq_liq.miscibility_gap_simplices,
            num_comp=len(transformed_feed_molar_flowrates))

        if len(phases_flowrates) == 1:
            # in this case we just split with a ratio of 0.5
            phase_1_flowrate = 0.5 * transformed_feed_molar_flowrates
            phase_2_flowrate = 0.5 * transformed_feed_molar_flowrates

        else:
            if len(phases_flowrates) == 2:
                phase_1_flowrate = phases_flowrates[0]
                phase_2_flowrate = phases_flowrates[1]

            else:
                print("\n\nsimplex with more than 2 phases")
                print(transformed_feed_molar_flowrates, phases_flowrates)
                print("this should not happen in those systems\n\n")
                return None

    # ensure no negative flowrates
    phase_1_flowrate = np.clip(phase_1_flowrate, 0, None)
    phase_2_flowrate = np.clip(phase_2_flowrate, 0, None)

    # ensure mass balance
    if sum(np.abs(transformed_feed_molar_flowrates - phase_1_flowrate - phase_2_flowrate)) > 0.001 * sum(
            transformed_feed_molar_flowrates) and (
            sum(transformed_feed_molar_flowrates) > len(transformed_feed_molar_flowrates) * 0.001) and (
            sum(np.abs(transformed_feed_molar_flowrates - phase_1_flowrate - phase_2_flowrate)) > 0.001):
        print("\n\nmass balance decantation incorrect\n\n")
        print("feed", transformed_feed_molar_flowrates)
        print("p1", phase_1_flowrate)
        print("p2", phase_2_flowrate)
        return None

    return [phase_1_flowrate, phase_2_flowrate]


def distillation(transformed_feed_flowrates, df, column, current_vle):
    if np.max(transformed_feed_flowrates) < 0.001:
        outputs = [df * transformed_feed_flowrates, (1 - df) * transformed_feed_flowrates]

    else:
        outputs = column.compute_output(transformed_feed_flowrates, df, current_vle)

    # ensure positive flowrates
    outputs[0] = np.clip(outputs[0], 0, None)
    outputs[1] = np.clip(outputs[1], 0, None)

    # ensure mbil
    if (sum(np.abs(transformed_feed_flowrates - outputs[0] - outputs[1])) > 0.001 * sum(
            transformed_feed_flowrates)) and (
            sum(transformed_feed_flowrates) > len(transformed_feed_flowrates) * 0.001) and (
            sum(np.abs(transformed_feed_flowrates - outputs[0] - outputs[1])) > 0.001):
        print("\n\nmass balance distillation incorrect\n\n")
        print("df", df)
        print("feed", transformed_feed_flowrates)
        print("top", outputs[0])
        print("bottom", outputs[1])
        return None

    return outputs


def compute_molefractions(molar_flowrates):
    if sum(molar_flowrates) > 0:
        return molar_flowrates / sum(molar_flowrates)

    else:
        return molar_flowrates


def euclidean_distance(point_1, point_2):
    return np.sqrt(sum(np.square(point_1 - point_2)))


def truncate(number, decimals=6):
    """
    Returns a value truncated to a specific number of decimal places.

    The function truncate will be applied to the inputs of the column.
    """
    if not isinstance(decimals, int):
        raise TypeError("decimal places must be an integer.")
    elif decimals < 0:
        raise ValueError("decimal places has to be 0 or more.")
    elif decimals == 0:
        return math.trunc(number)

    factor = 10.0 ** decimals
    return math.trunc(number * factor) / factor


class singular_point:
    def __init__(self, molar_fractions, cart_coords, boiling_point, role, index):
        """
        class to store singular points in the ternary
        """
        self.molar_fractions = molar_fractions
        self.cart_coords = cart_coords
        self.boiling_point = boiling_point
        self.role = role  # -1 low boiler, 0 saddle, 1 high boiler
        self.index = index  # index inside a complete singular_points list (initialized later)


class distillation_region:
    def __init__(self, boundary_1, boundary_2, index_high_boiler, index_low_boiler, s_p_indices_contained):
        """
        boundaries are given in cartesian coordinates, going from heavy boiler to low boiler
        """
        self.boundaries = [boundary_1, boundary_2]
        self.index_high_boiler = index_high_boiler
        self.index_low_boiler = index_low_boiler
        self.s_p_indices_contained = s_p_indices_contained


class ternary_vle:
    def __init__(self, index, name, path):
        """
        Stores the distillation regions, singular points and boiling points
        of the considered example.
        """
        self.index = index  # refers to outer list, where all ternaries are stored
        self.name = name
        self.standard_path = os.path.join(path, name)

        # indices of components
        self.comp_indices = np.load(os.path.join(self.standard_path, "sorted_indices.npy"))
        self.num_comp = len(self.comp_indices)

        # get pressure
        self.pressure = np.load(os.path.join(self.standard_path, "pressure.npy"))

        # transformation matrices
        matrices = np.load(os.path.join(self.standard_path, "coord_trafo_matrices.npy"))
        self.matrix_mfr_to_cart = matrices[0]
        self.matrix_cart_to_mfr = matrices[1]

        # number of distillation regions
        self.num_dis_regions = np.load(os.path.join(self.standard_path, "num_dis_reg.npy"))

        # get distillation regions
        self.distillation_regions = []
        for i in range(self.num_dis_regions):
            high_low_indices = np.load(os.path.join(self.standard_path, "dis_reg_" + str(i) + "_high_low.npy"))
            high_index = round(high_low_indices[0])
            low_index = round(high_low_indices[1])

            # boundaries and indices of singular points contained
            boundaries = []
            s_p_indices = []
            for j in range(2):
                bound_matrix = np.load(os.path.join(
                    self.standard_path, "dis_reg_" + str(i) + "_bound_" + str(j) + ".npy"))
                boundary = []
                for k in range(len(bound_matrix)):
                    boundary.append(bound_matrix[k])

                boundaries.append(boundary)

                index_pairs = np.load(os.path.join(
                    self.standard_path, "dis_reg_" + str(i) + "_bound_indices_" + str(j) + ".npy"))

                for k in range(len(index_pairs)):
                    for u in range(2):
                        candidate = round(index_pairs[k][u])
                        if candidate not in s_p_indices:
                            s_p_indices.append(candidate)

            self.distillation_regions.append(distillation_region(boundaries[0], boundaries[1], high_index, low_index,
                                                                 s_p_indices))

        # get singular points
        self.singular_points = []
        s_p_mfr = np.load(os.path.join(self.standard_path, "molar_fractions_s_p.npy"))
        s_p_cart_coords = np.load(os.path.join(self.standard_path, "cart_coords_s_p.npy"))
        s_p_bps = np.load(os.path.join(self.standard_path, "boiling_point_s_p.npy"))
        s_p_roles = np.load(os.path.join(self.standard_path, "role_s_p.npy"))

        for i in range(len(s_p_bps)):
            self.singular_points.append(singular_point(s_p_mfr[i], s_p_cart_coords[i], s_p_bps[i],
                                                       round(s_p_roles[i]), i))

    def transform_molar_fr_to_cartesian(self, molar_fractions):
        """
        A * lambda = (1, p), we cut off the first entry
        """

        return np.matmul(self.matrix_mfr_to_cart, molar_fractions)[1:]

    def transform_cartesian_to_molar_fr(self, cartesian_point):
        """
        lambda = A_inv * (1, p)
        """
        vector = np.empty(self.num_comp)
        vector[0] = 1
        vector[1:] = cartesian_point

        return np.matmul(self.matrix_cart_to_mfr, vector)


class distillation_column:
    def __init__(self):
        # below this boundary, a value will be considered as equal to 0
        # (important for intersections etc)
        self.zero_epsilon = 0.00001
        # epsilon cirteria for the DF condition
        self.DF_epsilon = 0.005
        # initial discretization of line element in shift_solution
        self.init_steps_per_boundary = 20
        # basis for increase of self.init_steps_per_boundary, good to have here
        # an even number, as better discretizations include former solutions then
        self.basis_increase = 4
        # number of different stepsizes, which are tested, if we cannot find a
        # good shift_solution
        self.num_of_increases = 1

    # it has to be ensured in the flowsheet simulation that no negative values are inserted here
    # distillation_regions, singular_points, boiling_points_singular_points are
    # provided in cartesian coordinates
    def compute_output(self, molar_flowrates_feed, DF, tern_sys: ternary_vle):
        # just to shorten names
        distillation_regions = tern_sys.distillation_regions
        singular_points = tern_sys.singular_points
        boiling_points_singular_points = [p.boiling_point for p in singular_points]

        DF = truncate(DF)
        molar_flowrates_feed = np.array([truncate(x) for x in molar_flowrates_feed])

        # raise errors for negative input
        if np.min(molar_flowrates_feed) < -1 * self.zero_epsilon:
            print('there is something wrong with:', molar_flowrates_feed,
                  DF, tern_sys.pressure)
            exit

        # close to zero input in flowrates is skipped
        elif sum(np.abs(molar_flowrates_feed)) < self.zero_epsilon:
            return [molar_flowrates_feed * DF, molar_flowrates_feed * (1 - DF)]

        molar_fractions_feed = compute_molefractions(molar_flowrates_feed)
        feed_cartesian = tern_sys.transform_molar_fr_to_cartesian(molar_fractions_feed)

        # if DF is close to 0 or 1, there is nothing to do
        if DF > 1 - self.zero_epsilon:
            molar_flowrates_top = molar_flowrates_feed
            molar_flowrates_bottom = np.zeros(3)

        elif DF < self.zero_epsilon:
            molar_flowrates_top = np.zeros(3)
            molar_flowrates_bottom = molar_flowrates_feed

        else:
            # if the feed is close to a singular point (also the pure
            # components are considered as singular points), it is
            # just split according to DF
            distances_to_singular_points = [
                euclidean_distance(feed_cartesian, point.cart_coords) for point in singular_points
            ]

            # capture all feeds close to singular points
            if np.min(distances_to_singular_points) < 0.01:
                molar_flowrates_top = DF * molar_flowrates_feed
                molar_flowrates_bottom = molar_flowrates_feed - molar_flowrates_top

            # check if the feed only contains two components (in this case
            # the output can be computed quite easily)
            elif np.min(molar_fractions_feed) < self.zero_epsilon:
                molar_flowrates_top, molar_flowrates_bottom = self.binary_distillation(
                    molar_flowrates_feed, DF, singular_points, boiling_points_singular_points, tern_sys)

            else:
                molar_flowrates_top, molar_flowrates_bottom = self.ternary_distillation(molar_flowrates_feed, DF,
                    distillation_regions, singular_points, tern_sys)

        # we want to ensure mass balance and that no negative flowrates are
        # returned (small negative values could always occur due to numerical
        # issues)
        # as either top or bottom are always computed as difference of feed
        # and the other, there shouldn't be negative values in both of them
        # (as one of them was computed by intersections etc)
        if np.min(molar_flowrates_top) < -1 * self.zero_epsilon and np.min(
                molar_flowrates_bottom) < -1 * self.zero_epsilon:
            print('there is something wrong with:', molar_flowrates_feed,
                  DF, tern_sys.pressure)
            exit

        # otherwise, just one might have slightly negative entries
        # this is corrected here,
        if np.min(molar_flowrates_top) < -1 * self.zero_epsilon:
            molar_flowrates_bottom = molar_flowrates_bottom + np.clip(
                molar_flowrates_top, -float('inf'), 0)
            molar_flowrates_top = molar_flowrates_feed - molar_flowrates_bottom

        if np.min(molar_flowrates_bottom) < -1 * self.zero_epsilon:
            molar_flowrates_top = molar_flowrates_top + np.clip(
                molar_flowrates_bottom, -float('inf'), 0)
            molar_flowrates_bottom = molar_flowrates_feed - molar_flowrates_top

        # second check, if everything is ok
        if np.min(molar_flowrates_top) < -1 * self.zero_epsilon or np.min(
                molar_flowrates_bottom) < -1 * self.zero_epsilon:
            print('there is something wrong with:', molar_flowrates_feed,
                  DF, tern_sys.pressure)
            exit

        if not self.mass_balance(molar_flowrates_feed, [molar_flowrates_top, molar_flowrates_bottom]):
            print('there is something wrong with:', molar_flowrates_feed,
                  DF, tern_sys.pressure)
            exit

        # minor negative values are just clipped of
        else:
            molar_flowrates_top = np.clip(molar_flowrates_top, 0, float('inf'))
            molar_flowrates_bottom = np.clip(molar_flowrates_bottom,
                                             0, float('inf'))

        return [molar_flowrates_top, molar_flowrates_bottom]

    def binary_distillation(self, molar_flowrates_feed, DF, singular_points, boiling_points_singular_points,
                            tern_sys: ternary_vle):
        molar_fractions_feed = compute_molefractions(molar_flowrates_feed)
        feed_cartesian = tern_sys.transform_molar_fr_to_cartesian(molar_fractions_feed)

        relevant_singular_points = []
        relevant_boiling_points = []

        profile_feed = np.ones(3)
        for i in range(3):
            if molar_fractions_feed[i] < self.zero_epsilon:
                profile_feed[i] = 0

            else:
                relevant_singular_points.append(singular_points[i])
                relevant_boiling_points.append(boiling_points_singular_points[i])

        for i in range(3, len(singular_points)):
            profile_signular_point = np.ones(3)
            for j in range(3):
                if singular_points[i].molar_fractions[j] < self.zero_epsilon:
                    profile_signular_point[j] = 0

            if sum(np.abs(profile_feed - profile_signular_point)) < self.zero_epsilon:
                relevant_singular_points.append(singular_points[i])
                relevant_boiling_points.append(boiling_points_singular_points[i])

        # all the relevant singular points and the feed are on one line
        # we need to find the two points, which enclose the feed
        distances_to_feed = [euclidean_distance(feed_cartesian,
                                                point.cart_coords) for point in relevant_singular_points]

        # the first point is just given by the minimum distance
        if len(distances_to_feed) == 0:
            print('there is something wrong with:', molar_flowrates_feed,
                  DF, tern_sys.pressure)
            exit

        index_first_enclosing_point = np.argmin(distances_to_feed)
        first_enclosing_point = relevant_singular_points[index_first_enclosing_point]
        first_enclosing_point_boiling_point = relevant_boiling_points[
            index_first_enclosing_point]

        # for the second enclosing point, we also have to ensure that it is on
        # the other side of the feed
        sorted_indices_increasing = np.argsort(distances_to_feed)
        second_enclosing_point = None

        for sorted_index in sorted_indices_increasing[1:]:
            candidate_point = relevant_singular_points[sorted_index]

            # this ensures that the second point is on the other side
            if euclidean_distance(candidate_point.cart_coords,
                                  first_enclosing_point.cart_coords) >= distances_to_feed[sorted_index]:
                second_enclosing_point = relevant_singular_points[sorted_index]
                second_enclosing_point_boiling_point = relevant_boiling_points[sorted_index]
                break

        if second_enclosing_point is None:
            print('there is something wrong with:', molar_flowrates_feed,
                  DF, tern_sys.pressure)
            return None

        # determine local high/low boiler
        if first_enclosing_point_boiling_point > second_enclosing_point_boiling_point:
            local_high_boiler = first_enclosing_point
            local_low_boiler = second_enclosing_point

        else:
            local_high_boiler = second_enclosing_point
            local_low_boiler = first_enclosing_point

        molar_flowrates_top, molar_flowrates_bottom = self.compute_maximum_split_on_given_line(
            molar_flowrates_feed, feed_cartesian, DF, local_low_boiler, local_high_boiler, tern_sys)

        return molar_flowrates_top, molar_flowrates_bottom

    def ternary_distillation(self, molar_flowrates_feed, DF, distillation_regions,
                             singular_points, tern_sys: ternary_vle):
        """
        reminder:
        material balance: top product, bottom product and feed are on straight
        line.

        infinite reflux and infinite column height: top product and bottom
        product are on the same distillation line and the column profile must
        contain a singular point (pure component or azeotrope). this means the
        respective distillation line must contain a singular point.
        """
        # case 1: top or bottom product are a singular point, which is no saddle point
        # the possibilites, which are created here, should fulfill exactly the
        # DF condition
        first_case_possibilities = self.one_product_singular(molar_flowrates_feed,
                                                             DF, distillation_regions, singular_points, tern_sys)

        # all solutions from the first case should always fulfill the DF condition
        real_solutions = first_case_possibilities

        # if we found a solution in the first step, we stop the search
        if len(real_solutions) == 0:
            solution_found = False

            # case 2: top and bottom product are not a singular point, which is no
            # saddle point.
            # this part is time consuming, as we loop over all boundaries and it
            # depends mostly on steps_per_line_element, which is the discretization
            # as one can find a good solution often with a small number of steps, we
            # only increase it, if necessary
            for exponent in range(self.num_of_increases):
                # we increase the steps up to 2 times
                steps_per_line_element = self.init_steps_per_boundary * (
                        self.basis_increase ** exponent)

                # it is important to take an even number for increase to have the same solutions as before in
                second_case_possibilities = self.shift_solution(molar_flowrates_feed, DF, distillation_regions,
                                                                steps_per_line_element, tern_sys)

                # keeps track if a dummy was in the candidates
                dummy_used = False

                # we add all second_case candidates, which fulfill the condition
                # at this point, we can already convert them to flowrates
                for candidate in second_case_possibilities:
                    if candidate[-1] is None:
                        dummy_used = True
                        # in this case the feed is on a distillation boundary (where
                        # we just compute the maximum distance solution on this line
                        # element. If the feed is even the edge of edge of a line
                        # element of the boundary, we just transform the column to
                        # a split,
                        solution_found = True

                    elif np.abs(candidate[-1] - DF) < self.DF_epsilon:
                        solution_found = True
                        cand_top_fractions = tern_sys.transform_cartesian_to_molar_fr(candidate[0])

                        # we have to use the computed DF, to ensure the other conditions
                        cand_top_flowrates = candidate[-1] * sum(
                            molar_flowrates_feed) * cand_top_fractions
                        cand_bottom_flowrates = molar_flowrates_feed - cand_top_flowrates

                        # ensure positive flowrates
                        if np.min(cand_bottom_flowrates) > -1 * self.zero_epsilon:
                            # floor small negative values
                            cand_bottom_flowrates = np.clip(cand_bottom_flowrates,
                                                            0, float('inf'))

                            real_solutions.append([cand_top_flowrates,
                                                   cand_bottom_flowrates])

                if solution_found:
                    break

        # in this case we just take the approximation, which is contained in
        # second_case_possibilities or if there was a dummy, we just treat
        # the column like a split with DF
        if len(real_solutions) == 0:
            # the feed was on the boundary, we just split it according to DF
            if dummy_used:
                cand_top_flowrates = DF * molar_flowrates_feed
                cand_bottom_flowrates = molar_flowrates_feed - cand_top_flowrates

            # we could not find a good enough solution, so we just take the
            # best we could find
            else:
                cand_top_fractions = tern_sys.transform_cartesian_to_molar_fr(candidate[0])

                # we have to use the computed DF, to ensure the other conditions
                cand_top_flowrates = candidate[-1] * sum(
                    molar_flowrates_feed) * cand_top_fractions
                cand_bottom_flowrates = molar_flowrates_feed - cand_top_flowrates

                # ensure mass balance (more important than being close to real DF)
                cand_bottom_flowrates = np.clip(cand_bottom_flowrates, 0, float('inf'))
                cand_top_flowrates = molar_flowrates_feed - cand_bottom_flowrates
                cand_top_flowrates = np.clip(cand_top_flowrates, 0, float('inf'))

            real_solutions.append([cand_top_flowrates, cand_bottom_flowrates])

        # we take the solution which has the best split according to weighted entropy
        if len(real_solutions) > 1:
            entropies = []
            for solution in real_solutions:
                weighted_entropy = 0

                # compute entropy for each output stream and weight it with the
                # total flowrate
                for j in range(2):
                    weighted_entropy = weighted_entropy + (
                            sum(solution[j]) * self.entropy(
                        compute_molefractions(solution[j])))

                entropies.append(weighted_entropy)

            best_solution = real_solutions[np.argmin(entropies)]

        else:
            best_solution = real_solutions[0]

        molar_flowrates_top = best_solution[0]
        molar_flowrates_bottom = best_solution[1]

        return molar_flowrates_top, molar_flowrates_bottom

    def shift_solution(self, molar_flowrates_feed, DF, distillation_regions, steps_per_line_element,
                       tern_sys: ternary_vle):
        """
        For each distillation boundary do:
            start at low boiler and go in small steps towards high boiler across
            the distillation line

            for each point:
                compute all intersections of line through point and feed with the
                given distillation boundary

                look if intersection satisfies DF condition, store best candidate
                (even if it does not fulfill DF condition)

        find the best candidate solution(s) and return those

        if no candidate fulfills the DF condition, the closest candidate is returned
        otherwise only candidates, which fulfill the DF condition are returned
        """
        candidates = []
        feed_cartesian = tern_sys.transform_molar_fr_to_cartesian(
            compute_molefractions(molar_flowrates_feed))

        for region in distillation_regions:
            for boundary in region.boundaries:
                candidates = candidates + self.loop_over_distillation_line(
                    feed_cartesian, DF, boundary, steps_per_line_element)

        candidate_solutions = []
        # add all candidates, which fulfill the DF condition
        for candidate in candidates:
            # is None just adds the dummy candidates, if necessary
            if candidate[-1] is None or np.abs(candidate[-1] - DF) < self.DF_epsilon:
                candidate_solutions.append(candidate)

        # find closest candidate
        if len(candidate_solutions) == 0:
            differences = [np.abs(x[-1] - DF) for x in candidates]
            candidate_solutions.append(candidates[np.argmin(differences)])

        return candidate_solutions

    def loop_over_distillation_line(self, feed_cartesian, DF, distillation_line,
                                    steps_per_line_element):
        """
        receives a distillation line as a list, ordered from heavy to low boiler

        returns a list of candidate_solutions
        each element is a list [top_product_cartesian,
                                bottom_product_cartesian,
                                df for this configuration]
        """
        candidate_solutions = []
        current_best_difference = float('inf')

        for index, line_element in enumerate(distillation_line[:-1]):
            # + 1 to get also the end of the line element
            for step in range(steps_per_line_element + 1):
                start_point = line_element + (
                        float(step / steps_per_line_element) * (
                        distillation_line[index + 1] - line_element))

                # check for all intersections with distillation line
                for j in range(len(distillation_line) - 1):
                    # it does not make sense to search on the element, which
                    # contains start_point
                    if j != index:
                        intersection_point, t, u = self.intersection_line_segments(
                            start_point, feed_cartesian, distillation_line[j],
                            distillation_line[j + 1])

                        # check DF condition
                        # case 1, j < index, intersection point is bottom product
                        if intersection_point is None or t < 1 or u < 0 or u > 1:
                            continue
                        # the u conditions ensure that the intersection is on
                        # the distillation line. the t condition ensures that
                        # the feed is between start_point and intersection
                        if j < index:
                            top_product_cartesian = start_point
                            bottom_product_cartesian = intersection_point

                        # case 2, j > index, intersection point is top product
                        else:
                            top_product_cartesian = intersection_point
                            bottom_product_cartesian = start_point

                        # we can only compute a DF, if intersection is not the start_point
                        if euclidean_distance(top_product_cartesian, bottom_product_cartesian) > 0:
                            DF_computed = euclidean_distance(feed_cartesian,
                                                             bottom_product_cartesian) / euclidean_distance(
                                top_product_cartesian, bottom_product_cartesian)

                            # if there are no candidates yet or a candidate fulfills
                            # the DF criteria it is appended
                            if len(candidate_solutions) == 0 or np.abs(
                                    DF_computed - DF) < self.DF_epsilon:
                                candidate_solutions.append([top_product_cartesian,
                                                            bottom_product_cartesian,
                                                            DF_computed])

                                current_best_difference = np.abs(DF_computed - DF)

                            # there is a candidate, but it does not fulfill the DF criteria
                            # if the current combination is closer, it replaces the old one
                            elif np.abs(DF_computed - DF) < current_best_difference:
                                candidate_solutions.append([top_product_cartesian,
                                                            bottom_product_cartesian, DF_computed])

                                current_best_difference = np.abs(DF_computed - DF)

                    # except for the case, when the feed is also part of this element
                    else:
                        # check if feed is on this element
                        length_line_element = euclidean_distance(
                            distillation_line[index], distillation_line[index + 1])
                        distance_feed_linestart = euclidean_distance(
                            feed_cartesian, distillation_line[index])
                        distance_feed_lineend = euclidean_distance(
                            feed_cartesian, distillation_line[index + 1])

                        # the difference of these lengths is almost equal,
                        # we consider the feed to be on this line
                        if np.abs(length_line_element -
                                  distance_feed_linestart -
                                  distance_feed_lineend) < self.zero_epsilon:

                            # in this case, we just add a dummy solution to our
                            # candidates, corresponding to a split ratio, which is
                            # close to DF (it has to be slightly different to not
                            # overrule possible real solutions)
                            # this also is a solution for the case, when the feed
                            # is the start or end of the line_element
                            # the best difference is not updated, as we still
                            # want to see if we find something better
                            candidate_solutions.append([feed_cartesian,
                                                        feed_cartesian, None])

                            # if the feed is the start or end of the line element
                            # there is nothing more to do, we consider the column
                            # as split with DF as ratio
                            if (distance_feed_linestart > np.sqrt(self.zero_epsilon)) and (
                                    distance_feed_lineend > np.sqrt(self.zero_epsilon)):
                                # there are infinite solutions on the line
                                # element, we just compute the one with the
                                # maximum distance between top and bottom product

                                # boundary goes from heavy to light, therefore
                                # the end is the light boiler
                                local_low_boiler = distillation_line[index + 1]
                                local_high_boiler = distillation_line[index]

                                # first case, low boiler is "pure"
                                if euclidean_distance(feed_cartesian,
                                                      local_high_boiler) / euclidean_distance(
                                    local_high_boiler, local_low_boiler) > DF:
                                    # distance to the unknown bottom point
                                    distance_feed_tobottom = (distance_feed_lineend
                                                              / (1 - DF)) - distance_feed_lineend

                                    # the bottom point is somewhere between
                                    # the feed and the local high boiler
                                    bottom_product = feed_cartesian + (
                                            (distance_feed_tobottom / distance_feed_linestart) * (
                                            local_high_boiler - feed_cartesian))

                                    computed_df = distance_feed_tobottom / euclidean_distance(
                                        local_low_boiler, bottom_product)

                                    candidate_solutions.append([local_low_boiler,
                                                                bottom_product, computed_df])

                                # second case, high boiler is "pure",
                                # just the other way around
                                else:
                                    # distance to the unknown top point
                                    distance_feed_totop = (distance_feed_linestart / (
                                        DF)) - distance_feed_linestart

                                    top_product = feed_cartesian + (
                                            (distance_feed_totop / distance_feed_lineend) * (
                                            local_low_boiler - feed_cartesian))

                                    computed_df = distance_feed_linestart / euclidean_distance(
                                        local_high_boiler, top_product)

                                    candidate_solutions.append([top_product,
                                                                local_high_boiler, computed_df])

        return candidate_solutions

    def one_product_singular(self, molar_flowrates_feed, DF, distillation_regions,
                             singular_points, tern_sys: ternary_vle):
        """
        top or bottom product are a singular point, which is no saddle point

        we loop over all singular points, which are no saddles:
            draw a line from the singular point through the feed so that DF condition
            is fulfilled and get second product.

            check, if the singular point and the second product are in the same
            region (we consider singular points to be in every region they are
            connected to).

        return a list with all feasible combinations
        """
        feasible_possibilities = []
        for reg_ind, reg in enumerate(distillation_regions):
            for s_p_ind, s_p in enumerate(singular_points):
                # if s_p is no saddle and in the respective region
                if s_p.role != 0 and s_p_ind in reg.s_p_indices_contained:
                    if s_p.role == 1:
                        singular_is_heavy = True

                    else:
                        singular_is_heavy = False

                    # compute second product according DF condition
                    if singular_is_heavy:
                        singular_point_flowrates = (1 - DF) * sum(
                            molar_flowrates_feed) * s_p.molar_fractions

                    else:
                        singular_point_flowrates = DF * sum(
                            molar_flowrates_feed) * s_p.molar_fractions

                    second_product = molar_flowrates_feed - singular_point_flowrates

                    # check if feasible
                    if np.min(second_product) > -1 * self.zero_epsilon:
                        # floor small negative values
                        second_product = np.clip(second_product, 0, None)

                        cartesian_second_product = tern_sys.transform_molar_fr_to_cartesian(
                            compute_molefractions(second_product))

                        # check if it is in the region corresponding to region_index
                        if self.jordan_method(cartesian_second_product, reg):
                            if singular_is_heavy:
                                feasible_possibilities.append([
                                    second_product, singular_point_flowrates])

                            else:
                                feasible_possibilities.append([
                                    singular_point_flowrates, second_product])

        return feasible_possibilities

    def intersection_line_segments(self, point1_line1, point2_line1,
                                   point1_line2, point2_line2):
        intersection_point = None
        t = -1
        u = -1

        x1 = point1_line1[0]
        x2 = point2_line1[0]
        x3 = point1_line2[0]
        x4 = point2_line2[0]

        y1 = point1_line1[1]
        y2 = point2_line1[1]
        y3 = point1_line2[1]
        y4 = point2_line2[1]

        denominator = ((x1 - x2) * (y3 - y4)) - ((y1 - y2) * (x3 - x4))

        if np.abs(denominator) > self.zero_epsilon:
            t = (((x1 - x3) * (y3 - y4)) - ((y1 - y3) * (x3 - x4))) / denominator
            u = (((x1 - x3) * (y1 - y2)) - ((y1 - y3) * (x1 - x2))) / denominator
            intersection_point = np.array([x1 + (t * (x2 - x1)), y1 + (t * (y2 - y1))])

        return intersection_point, t, u

    def jordan_method(self, point, region):
        """
        check if a point is in a distillation region, jordan method

        for outside rays through the point check how many intersections there
        are with the polygon. If the number is odd, the point is inside
        """
        # we do that for three choices of outside points and take the result,
        # which is produced at least 2 times (as it could be that for some
        # outside points)
        outside_points = [-1 * np.ones(2), np.array([2, 5]), np.array([0, -4])]
        result = []

        for outside_point in outside_points:
            count = 0
            for j in range(2):
                for boundary_element_index in range(len(region.boundaries[j]) - 1):
                    intersection_point, t, u = self.intersection_line_segments(
                        point, outside_point,
                        region.boundaries[j][boundary_element_index],
                        region.boundaries[j][boundary_element_index + 1])

                    if t >= 0 and u >= 0 and t <= 1 and u <= 1:
                        count = count + 1

            if int(count) % 2 == 1:
                result.append(True)

            else:
                result.append(False)

        if sum(result) > 1.5:
            is_in_region = True

        else:
            is_in_region = False

        return is_in_region

    def mass_balance(self, feed, output_list):
        if sum(np.abs(feed - output_list[0] - output_list[1])) > self.zero_epsilon:
            return False

        else:
            return True

    @staticmethod
    def compute_maximum_split_on_given_line(molar_flowrates_feed, feed_cartesian, DF, local_low_boiler,
                                            local_high_boiler, tern_sys: ternary_vle):
        """
        this function is useful, if the line through feed, distillate and bottom
        is already known (e.g. binary_distillation)
        """
        # one of the local high/low boilers will be a "pure" product somewhere
        # by this and DF the other one can be determined
        if euclidean_distance(feed_cartesian, local_high_boiler.cart_coords) / euclidean_distance(
                local_high_boiler.cart_coords, local_low_boiler.cart_coords) > DF:
            molar_flowrates_top = sum(molar_flowrates_feed) * DF * tern_sys.transform_cartesian_to_molar_fr(
                local_low_boiler.cart_coords)
            molar_flowrates_bottom = molar_flowrates_feed - molar_flowrates_top

        else:
            molar_flowrates_bottom = sum(molar_flowrates_feed) * (
                    1 - DF) * tern_sys.transform_cartesian_to_molar_fr(local_high_boiler.cart_coords)
            molar_flowrates_top = molar_flowrates_feed - molar_flowrates_bottom

        return molar_flowrates_top, molar_flowrates_bottom

    @staticmethod
    def entropy(molar_fractions):
        summed_term = 0
        for x in molar_fractions:
            if x > 0:
                summed_term = summed_term + (x * np.log(x))

        return -1 * summed_term

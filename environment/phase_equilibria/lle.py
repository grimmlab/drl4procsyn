import numpy as np
import scipy.spatial as spat
import os
import time
import itertools
import copy
import ray
import math


class miscibility_gap_simplex:
    def __init__(self, points_coords_cart, points_mfr, points_ind, matrix, matrix_inv):
        """
        class to store a simplex, which contains points which split into multiple liquid phases

        all important attributes are stored in those simplices to be able to define a unit opartion only
        with those, without the simplex discretization module etc.

        these simplices always have dimension n (N=n+1 vertices) and define some phase split.
        for this, we classify the edges into homogeneous (between neighboring points) and
        heterogeneous. neighboring points are two points, which are contained in one subsimplex
        from the discretization
        """
        self.index = None
        self.points_coordinates_cart = points_coords_cart
        self.points_molar_fractions = points_mfr
        self.points_indices = points_ind  # point indices from the simplex discretization
        self.matrix = matrix  # matrix * lambda = (1, p), lambda barycentric coordinates
        self.matrix_inv = matrix_inv

        # symmetric N x N matrix, entry[i][j] == 1 means that the edge from point[i] to point [j]
        # is heterogeneous, matrix is filled afterwards (default is a homogeneous simplex without
        # an occurring phase split).
        self.edge_classification = np.zeros((len(self.points_indices), len(self.points_indices)))

        # for each phase, we append an block == [indices], where indices refers to the point indices
        # which form the basis of the respective phase. the order here is the fixed order of the phases
        self.phase_blocks = []  # the indices in here range from 0 to N-1

    def get_middle(self):
        middle = np.zeros(len(self.points_molar_fractions[0]))
        for mfr in self.points_molar_fractions:
            middle = middle + mfr

        return middle / len(self.points_molar_fractions)


class miscibility_analysis:
    def __init__(self, discretized_system, gE_model, temperature, construct, path, actors_for_para):
        """
        For a detailed description see goettl2023 (convex envelope method).

        the whole component system can be visualized as a simplex and we need a subdivision of this simplex
        as input along with an gE model (to compute delta_g_mix) and temperature (in Kelvin).

        if actors_for_para > 0, d_g_mix values and compare simplices will be run parallelized
        """
        # SimplexDisc class (subsimplices with constant volume but not constant edge length)
        self.discretized_system = discretized_system
        self.gE_model = gE_model  # e.g. NRTL class, UNQUAC class
        self.temperature = temperature  # unit K
        self.num_comp = gE_model.num_comp
        self.epsilon = 0.0001  # for comparisons
        self.actors_for_para = actors_for_para

        if construct:
            # for time stats
            self.time_for_d_g_mix = None
            self.time_for_conv_hull = None
            self.time_for_comparisons = None

            # num phase stats
            self.num_phase_stats = np.zeros(self.num_comp)

            if self.actors_for_para > 0:
                ray.init(num_cpus=70)
                self.convex_hull_simplices = self.get_hull_parallel()
                self.miscibility_gap_simplices = self.compare_simplices_parallel()
                ray.shutdown()

            else:
                self.convex_hull_simplices = self.get_hull()
                self.miscibility_gap_simplices = self.compare_simplices()

        else:
            self.load_phase_eq_liquid(path)

    def compare_simplices(self):
        """
        at this stage we already have a convex hull of our graph which consists of the cartesian coordinates
        and the last entry corresponds to min(0, delta_g_mix).

        now we check for all simplices in the convex hull, if those connect neighboring points in the original
        setting (without delta_g_mix values).
        """
        # storage for all misc gap simplices
        miscibility_gap_simplices = []
        expected_length = np.sqrt(2) * self.discretized_system.stepsize
        expected_simplex_volume = self.discretized_system.volume_simplex(
            self.discretized_system.vertices_outer_simplex) / (
                int(2 ** self.discretized_system.recursion_steps) ** (self.num_comp - 1))

        print("\ncompare simplices\n")
        start = time.time()
        for index, simplex_points_indices in enumerate(self.convex_hull_simplices):
            delta_g_s = [self.graph[i][-1] for i in simplex_points_indices]
            real_point_indices = [self.graph_points_real_indices[i] for i in simplex_points_indices]
            vertices_mfr = [self.discretized_system.points_mfr[i] for i in real_point_indices]
            vertices_cartesian = [self.discretized_system.points_cart[i] for i in real_point_indices]
            simplex_volume = self.discretized_system.volume_simplex(vertices_cartesian)

            found_simplex = self.check_simplex(self.num_comp, delta_g_s, vertices_mfr, vertices_cartesian,
                                               simplex_volume, expected_simplex_volume, self.discretized_system,
                                               expected_length, real_point_indices, self.gE_model, self.temperature)

            if found_simplex is not None:
                num_phases = len(found_simplex.phase_blocks)
                self.num_phase_stats[num_phases-1] = self.num_phase_stats[num_phases-1] + 1
                miscibility_gap_simplices.append(found_simplex)

        print("\ncomparisons complete\n")
        self.time_for_comparisons = time.time() - start

        return miscibility_gap_simplices

    def compare_simplices_parallel(self):
        # storage for all misc gap simplices
        expected_length = np.sqrt(2) * self.discretized_system.stepsize
        expected_simplex_volume = self.discretized_system.volume_simplex(
            self.discretized_system.vertices_outer_simplex) / (
                                          int(2 ** self.discretized_system.recursion_steps) ** (self.num_comp - 1))
        separation = self.get_index_separation(len(self.convex_hull_simplices), self.actors_for_para)

        print("\ncompare simplices parallel\n")
        start = time.time()
        para_res_ids = []
        for i in range(len(separation) - 1):
            index_start = separation[i]
            index_end = separation[i + 1]
            simplices_to_check = self.convex_hull_simplices[index_start:index_end]

            para_res_ids.append(self.check_simplex_list.remote(self, index_start, simplices_to_check,
                                                               expected_simplex_volume, expected_length))

        results = ray.get(para_res_ids)
        print("\ncomparisons complete\n")
        self.time_for_comparisons = time.time() - start

        miscibility_gap_simplices = []
        for res in results:
            miscibility_gap_simplices = miscibility_gap_simplices + res[0]
            self.num_phase_stats = self.num_phase_stats + res[1]

        return miscibility_gap_simplices

    def get_hull(self):
        """
        the simplex discretization provides us with points inside the component system, for those we compute
        delta_g_mix and get the convex hull of this graph afterwards
        """
        # first we get all values for delta g mix
        self.values_delta_g_mix = np.empty(len(self.discretized_system.points_mfr))
        self.graph = []
        self.graph_points_real_indices = []  # a mapping for the relevant points as we ignore d_g_mix > 0
        print("\nget delta g mix graph\n")
        start = time.time()
        for i, point_mfr in enumerate(self.discretized_system.points_mfr):
            # print("delta_g_mix graph ", 100 * i / len(self.discretized_system.points_mfr), time.time() - start)
            self.values_delta_g_mix[i] = self.compute_delta_g_mix(point_mfr, self.gE_model, self.temperature)

            # only negative values matter for this method as positive values for delta_g_mix never
            # lead to a stable state in our case
            self.values_delta_g_mix[i] = np.min([0, self.values_delta_g_mix[i]])

            # we only care for negative values and pure components
            if self.values_delta_g_mix[i] < -1 * self.epsilon or np.max(self.discretized_system.points_mfr) > 1 - self.epsilon:
                # set the value in our graph
                graphvalue = np.zeros(self.num_comp)
                graphvalue[:-1] = self.discretized_system.points_cart[i]
                graphvalue[-1] = self.values_delta_g_mix[i]
                self.graph.append(graphvalue)
                self.graph_points_real_indices.append(i)

        self.time_for_d_g_mix = time.time() - start
        print("\ntime used for d_g_mix", self.time_for_d_g_mix, "\n")

        print("get real hull")
        start = time.time()
        hull = spat.ConvexHull(self.graph).simplices
        self.time_for_conv_hull = time.time() - start

        return hull

    def get_hull_parallel(self):
        separation = self.get_index_separation(len(self.discretized_system.points_mfr), self.actors_for_para)

        print("\nget delta g mix graph parallel\n")
        start = time.time()
        para_res_ids = []
        for i in range(len(separation) - 1):
            index_start = separation[i]
            index_end = separation[i + 1]
            molar_fractions_list = self.discretized_system.points_mfr[index_start:index_end]
            cart_coords_list = self.discretized_system.points_cart[index_start:index_end]
            para_res_ids.append(self.c_d_g_mix_list.remote(self, index_start, molar_fractions_list, cart_coords_list))

        results = ray.get(para_res_ids)
        self.graph = []
        self.graph_points_real_indices = []
        for res in results:
            self.graph = self.graph + res[0]
            self.graph_points_real_indices = self.graph_points_real_indices + res[1]

        self.time_for_d_g_mix = time.time() - start
        print("\ntime used for dg_mix", self.time_for_d_g_mix, "\n")

        print("get real hull")
        start = time.time()

        current_point_set = self.graph
        current_index_set = list(range(len(self.graph)))
        stages = [self.actors_for_para]
        downgrade = 1.5
        while math.floor(stages[-1] / downgrade) > 1:
            stages.append(math.floor(stages[-1] / downgrade))

        for sub_hull_stage in stages:
            print("\nsub_step", sub_hull_stage, len(current_point_set))
            separation = self.get_index_separation(len(current_point_set), sub_hull_stage)
            para_res_ids = []
            for i in range(len(separation) - 1):
                index_start = separation[i]
                index_end = separation[i + 1]
                points_for_hull = current_point_set[index_start:index_end]
                para_res_ids.append(self.sub_hull.remote(self, points_for_hull, index_start, index_end))

            results = ray.get(para_res_ids)

            # correct indices
            new_indices = []
            for res_ind, res in enumerate(results):
                corresponding_indices_total_set = current_index_set[res[1][0]:res[1][1]]
                corresponding_indices = [corresponding_indices_total_set[k] for k in res[0]]

                new_indices = new_indices + corresponding_indices

            new_points = [self.graph[k] for k in new_indices]

            current_point_set = new_points
            current_index_set = new_indices

        print("\nlast len", len(current_point_set), "\n")
        # now conv hull of the last merge
        hull_constructor = spat.ConvexHull(current_point_set)
        pre_hull = hull_constructor.simplices
        hull = []
        for j, el in enumerate(pre_hull):
            new_el = []
            for i in range(len(el)):
                new_el.append(current_index_set[el[i]])

            hull.append(new_el)

        self.time_for_conv_hull = time.time() - start
        print("\ntime for hull", self.time_for_conv_hull, len(hull_constructor.vertices), "\n")

        return hull

    @ray.remote
    def sub_hull(self, points_for_hull, index_start, index_end):
        return [spat.ConvexHull(points_for_hull).vertices, [index_start, index_end]]

    def store_phase_eq_liquid(self, name, comp_indices, path=None):
        """
        component indices are for the names of the files and folders etc
        """
        # initialize folders, if not existing
        if path is None:
            standard_path = os.path.join(os.getcwd(), "data", "lle")
            standard_path = os.path.join(standard_path, name)

        else:
            standard_path = path

        if not os.path.isdir(standard_path):
            os.mkdir(standard_path)

        comp_indices.sort()
        sorted_ind = np.array(comp_indices)

        # store array with sorted indices
        np.save(os.path.join(standard_path, "sorted_indices.npy"), sorted_ind)

        with open(os.path.join(standard_path, "time_stats_detailed.txt"), "w+") as file:
            file.write("time d_g_mix: " + str(self.time_for_d_g_mix) + "\n")
            file.write("time conv hull: " + str(self.time_for_conv_hull) + "\n")
            file.write("time comparisons: " + str(self.time_for_comparisons) + "\n")

        with open(os.path.join(standard_path, "num_phase_stats.txt"), "w+") as file:
            file.write("num phases stat: " + str(self.num_phase_stats) + "\n\n")

        # store misc gap simplices with all attributes
        misc_gap_simpl_p_coords_cart = np.empty((len(self.miscibility_gap_simplices), self.num_comp, self.num_comp - 1))
        misc_gap_simpl_p_mfrs = np.empty((len(self.miscibility_gap_simplices), self.num_comp, self.num_comp))
        misc_gap_simpl_p_ind = np.empty((len(self.miscibility_gap_simplices), self.num_comp))
        misc_gap_simpl_mat = np.empty((len(self.miscibility_gap_simplices),
                                       len(self.discretized_system.matrix_mfr_to_cart),
                                       len(self.discretized_system.matrix_mfr_to_cart)))
        misc_gap_simpl_mat_inv = np.empty((len(self.miscibility_gap_simplices),
                                       len(self.discretized_system.matrix_mfr_to_cart),
                                       len(self.discretized_system.matrix_mfr_to_cart)))
        edge_classifications = np.empty((len(self.miscibility_gap_simplices), self.num_comp, self.num_comp))
        phase_blocks = np.empty((len(self.miscibility_gap_simplices), (self.num_comp * 2) + 1))

        for i, simplex in enumerate(self.miscibility_gap_simplices):
            for j in range(self.num_comp):
                misc_gap_simpl_p_coords_cart[i][j] = simplex.points_coordinates_cart[j]
                misc_gap_simpl_p_mfrs[i][j] = simplex.points_molar_fractions[j]

            misc_gap_simpl_p_ind[i] = simplex.points_indices
            misc_gap_simpl_mat[i] = simplex.matrix
            misc_gap_simpl_mat_inv[i] = simplex.matrix_inv
            edge_classifications[i] = simplex.edge_classification
            # we store each cluster and add a -1 to mark its end
            current_ind = 0
            for cluster in simplex.phase_blocks:
                for j in cluster:
                    phase_blocks[i][current_ind] = j
                    current_ind = current_ind + 1

                phase_blocks[i][current_ind] = -1
                current_ind = current_ind + 1

            # mark end by multiple -1
            phase_blocks[i][current_ind:] = -1

        np.save(os.path.join(standard_path, "simpl_p_coords_cart.npy"), misc_gap_simpl_p_coords_cart)
        np.save(os.path.join(standard_path, "simpl_p_mfrs.npy"), misc_gap_simpl_p_mfrs)
        np.save(os.path.join(standard_path, "simpl_p_ind.npy"), misc_gap_simpl_p_ind)
        np.save(os.path.join(standard_path, "simpl_mat.npy"), misc_gap_simpl_mat)
        np.save(os.path.join(standard_path, "simpl_mat_inv.npy"), misc_gap_simpl_mat_inv)
        np.save(os.path.join(standard_path, "edge_classifications.npy"), edge_classifications)
        np.save(os.path.join(standard_path, "phase_blocks.npy"), phase_blocks)
        np.save(os.path.join(standard_path, "temperature.npy"), self.temperature)

        return standard_path

    def load_phase_eq_liquid(self, path):
        # load necessary data from given path
        # init misc gap simplices
        misc_gap_simpl_p_coords_cart = np.load(os.path.join(path, "simpl_p_coords_cart.npy"))
        misc_gap_simpl_p_mfrs = np.load(os.path.join(path, "simpl_p_mfrs.npy"))

        # for the index arrays, we need integers
        misc_gap_simpl_p_ind = np.load(os.path.join(path, "simpl_p_ind.npy")).astype(int)

        misc_gap_simpl_mat = np.load(os.path.join(path, "simpl_mat.npy"))
        misc_gap_simpl_mat_inv = np.load(os.path.join(path, "simpl_mat_inv.npy"))
        edge_classifications = np.load(os.path.join(path, "edge_classifications.npy"))

        # this one has to be transformed into clusters with ints as indices
        phase_blocks_source = np.load(os.path.join(path, "phase_blocks.npy"))

        self.miscibility_gap_simplices = []
        for i in range(len(misc_gap_simpl_p_ind)):
            misc_gap_simplex = miscibility_gap_simplex(misc_gap_simpl_p_coords_cart[i],
                misc_gap_simpl_p_mfrs[i], misc_gap_simpl_p_ind[i], misc_gap_simpl_mat[i], misc_gap_simpl_mat_inv[i])
            misc_gap_simplex.edge_classification = edge_classifications[i]

            phase_blocks = []
            block = []
            for j in range((self.num_comp * 2) + 1):
                # end of a cluster is marked
                if phase_blocks_source[i][j] < 0:
                    phase_blocks.append(block)
                    # check if we are at the end
                    if phase_blocks_source[i][j+1] < 0:
                        break

                    else:
                        block = []

                else:
                    block.append(int(phase_blocks_source[i][j]))

            misc_gap_simplex.phase_blocks = phase_blocks
            misc_gap_simplex.index = i
            self.miscibility_gap_simplices.append(misc_gap_simplex)

    @ray.remote
    def c_d_g_mix_list(self, index_start, molar_fractions_list, cart_coords_list):
        epsilon = 0.0001
        graph = []
        graph_points_real_indices = []

        for j, mfr in enumerate(molar_fractions_list):
            d_g_mix = np.min([0, miscibility_analysis.compute_delta_g_mix(mfr, self.gE_model, self.temperature)])

            # we only care for negative values and pure components
            if d_g_mix < -1 * epsilon or np.max(
                    mfr) > 1 - epsilon:
                # set the value in our graph
                graphvalue = np.zeros(len(mfr))
                graphvalue[:-1] = cart_coords_list[j]
                graphvalue[-1] = d_g_mix
                graph.append(graphvalue)
                graph_points_real_indices.append(j + index_start)

        return [graph, graph_points_real_indices, index_start]

    @ray.remote
    def check_simplex_list(self, index_start, simplices_to_check, expected_simplex_volume, expected_length):
        simplices_to_return = []
        num_phase_stats = np.zeros(self.num_comp)

        for simplex_points_indices in simplices_to_check:
            delta_g_s = [self.graph[i][-1] for i in simplex_points_indices]
            real_point_indices = [self.graph_points_real_indices[i] for i in simplex_points_indices]
            vertices_mfr = [self.discretized_system.points_mfr[i] for i in real_point_indices]

            vertices_cartesian = [self.discretized_system.points_cart[i] for i in real_point_indices]
            simplex_volume = self.discretized_system.volume_simplex(vertices_cartesian)

            found_simplex = miscibility_analysis.check_simplex(self.num_comp, delta_g_s, vertices_mfr,
                                                               vertices_cartesian,
                                                               simplex_volume, expected_simplex_volume,
                                                               self.discretized_system,
                                                               expected_length, real_point_indices, self.gE_model,
                                                               self.temperature)

            if found_simplex is not None:
                num_phases = len(found_simplex.phase_blocks)
                num_phase_stats[num_phases - 1] = num_phase_stats[num_phases - 1] + 1
                simplices_to_return.append(found_simplex)

        return [simplices_to_return, num_phase_stats, index_start]

    @staticmethod
    def check_simplex(num_comp, delta_g_s, vertices_mfr, vertices_cartesian, simplex_volume, expected_simplex_volume,
                      discretized_system, expected_length, real_point_indices, gE_model, temperature):
        """
        check if a simplex of the convex hull defines a misc gap simplex
        delta_g_s is contains the d_g_mix values of the points of the simplex
        """
        epsilon = 0.0001
        simplex_to_return = None
        # as described by Ryll2009, we don't have to care about subsimplices, where deltag_g_mix is 0
        # for all vertices. this way we automatically filter the "roof" (consisting of the pure components
        # of our convex hull).
        if np.min(delta_g_s) < -1 * epsilon:
            # we don't care about simplices with area equal to 0
            if np.abs(simplex_volume) > epsilon * expected_simplex_volume:
                matrix, matrx_inv = discretized_system.get_basis_change(vertices_cartesian)

                if np.abs(simplex_volume - expected_simplex_volume) / expected_simplex_volume > epsilon:
                    distance_matrix = np.zeros((num_comp, num_comp))
                    for i in range(num_comp):
                        for j in range(i+1, num_comp):
                            distance_matrix[i][j] = discretized_system.euclidean_distance(vertices_mfr[i],
                                                                                          vertices_mfr[j])

                            distance_matrix[j][i] = distance_matrix[i][j]

                    # if this is true we have a relevant simplex. we initialize a misc gap simplex
                    # with the necessary information
                    if np.abs(np.max(distance_matrix) - expected_length) / expected_length > epsilon:
                        # store this simplex
                        candidate_simplex = miscibility_gap_simplex(vertices_cartesian, vertices_mfr,
                                                                    real_point_indices, matrix, matrx_inv)

                        # we want to check if we can model the phase split in this simplex, for this, we have
                        # to determine for each edge if it is homo- or heterogeneous
                        for i in range(num_comp):
                            for j in range(i+1, num_comp):
                                # if the edge is too long, it is heterogeneous
                                if np.abs(distance_matrix[i][j] - expected_length) / expected_length > epsilon:
                                    # symmetric matrix
                                    candidate_simplex.edge_classification[i][j] = 1
                                    candidate_simplex.edge_classification[j][i] = 1

                        if candidate_simplex.edge_classification.sum() < 0.01:
                            return None

                        # we check for all vertices, if they are only connected to heterogeneous
                        # edges (which means they represent a phase) or if there are also homogeneous
                        # edges (and collect those to check if they form a lower dim simplex)
                        homogeneous_edges = []
                        for i in range(num_comp):
                            # if all connections are heterogeneous, we have a phase
                            if sum(candidate_simplex.edge_classification[i]) > 0.99 * (num_comp - 1):
                                candidate_simplex.phase_blocks.append([i])

                            else:
                                for j in range(i+1, num_comp):
                                    if candidate_simplex.edge_classification[i][j] < 0.01:
                                        homogeneous_edges.append([i, j])

                        # now we cluster all homogeneous edges (two edges are in the same cluster,
                        # if they have one index in common).
                        clusters = []
                        cluster = []
                        # if we add new edges to this cluster, we store a copy here, so that we check also for
                        # those, if there are connections left in homogeneous edges
                        todo = []
                        while len(homogeneous_edges) > 0 or len(todo) > 0:
                            # we always compare the remaining homogeneous edges with a current edge
                            if len(todo) == 0:
                                # if to_to is empty, a new cluster was started
                                current_edge = homogeneous_edges[0]
                                homogeneous_edges.remove(homogeneous_edges[0])
                                cluster.append(current_edge)

                            else:
                                current_edge = todo[0]
                                todo.remove(todo[0])

                            # here we store the found connections
                            to_remove = []
                            for i, edge in enumerate(homogeneous_edges):
                                # each edge occurs only once in homogeneous edges (due to the construction)
                                if current_edge[0] in edge or current_edge[1] in edge:
                                    to_remove.append(i)

                            # add to cluster and to to_do
                            for i in to_remove:
                                cluster.append(homogeneous_edges[i])
                                todo.append(homogeneous_edges[i])

                            # remove from homogeneous edges
                            for i in reversed(to_remove):
                                homogeneous_edges.remove(homogeneous_edges[i])

                            # if we did not find any new edges for the cluster and do not have anything in
                            # the to_do list left, we need a new cluster
                            if len(to_remove) == 0 and len(todo) == 0:
                                clusters.append(cluster)
                                cluster = []

                        # for each cluster, we check now, if it is exactly a low-dimensional simplex (not less and
                        # more, as then we cannot model the phase split linearly and we omit this misc gap simplex)
                        omit_candidate_simplex = False
                        for i, cluster in enumerate(clusters):
                            # it is enough to check if the cluster is a k simplex, where k+1 is the number of points
                            # in the cluster. attention: up to now we just stored the edges of the cluster, so we have
                            # to get the unique points first
                            point_ind_list = []
                            for edge in cluster:
                                if edge[0] not in point_ind_list:
                                    point_ind_list.append(edge[0])

                                if edge[1] not in point_ind_list:
                                    point_ind_list.append(edge[1])

                            # now we just check for every point index, if there are exactly k edges containing this
                            # index in the cluster (which means we would have a simplex
                            for point_index in point_ind_list:
                                edge_count = 0
                                for edge in cluster:
                                    if point_index in edge:
                                        edge_count = edge_count + 1

                                # if this condition is not fulfilled only once, we can stop
                                if edge_count != len(point_ind_list) - 1:
                                    omit_candidate_simplex = True
                                    break

                            # we add the point list, which specifies the phase
                            candidate_simplex.phase_blocks.append(point_ind_list)

                        # we try to reduce the candidate simplex
                        reduced_simplex, stat_std = miscibility_analysis.reduce_misc_gap_simplex(candidate_simplex,
                                                                                 gE_model, temperature,
                                                                                 discretized_system)

                        # if no reduction is possible, add the candidate, if this does not harm the isolated
                        # phase condition
                        if reduced_simplex is None:
                            if not omit_candidate_simplex:
                                vert_num = 0
                                for cl in candidate_simplex.phase_blocks:
                                    for vert in cl:
                                        vert_num = vert_num + 1

                                # just for safety
                                if vert_num != num_comp:
                                    return 1

                                simplex_to_return = candidate_simplex

                        # the reduced simplex fulfills the isolated phase block condition and we add this one
                        # (if existing)
                        else:
                            simplex_to_return = reduced_simplex

        return simplex_to_return

    @staticmethod
    def reduce_misc_gap_simplex(simplex, gE_model, temperature, discretized_system):
        """
        misc gap simplex is given, we check if some of the heterogeneous edges can also be homogeneous
        (this misclassification can happen due to sparse discretization...)

        rec_steps discretization and tolerance_steps just specify, which heterogeneous edges can also be
        seen as homogeneous
        """
        num_comp = len(simplex.points_indices)

        # first check if the heterogeneous edges are shorter than a predefined length
        hetero_index_pairs = []
        hetero_lengths = []
        for i in range(num_comp):
            for j in range(i + 1, num_comp):
                if simplex.edge_classification[i][j] == 1:
                    hetero_index_pairs.append([i, j])
                    hetero_lengths.append(miscibility_analysis.distance_for_reduce(simplex.points_molar_fractions[i],
                                                                                   simplex.points_molar_fractions[j]))

        candidate_indices = []
        # we also check if there are edges, which are quite short (and therefore most likely we need a reduction)
        max_len = np.max(hetero_lengths)
        must_reduce = False
        for i, pair in enumerate(hetero_index_pairs):
            if hetero_lengths[i] < 0.6 * max_len:
                must_reduce = True

            candidate_indices.append(i)

        index_subsets = []
        for i in candidate_indices:
            index_subsets.append([i])

        # if all indices are candidate indices, we just look at real subsets, as it just
        # seems quite unlikely that it is not at least some phase split there, but if there are
        # less candidate indices than hetero edges, we just look at all possible (sub)sets, as it
        # may be possible to reduce them all
        relevant_length = np.min([len(hetero_index_pairs) - 1, len(candidate_indices)])
        for i in range(2, relevant_length + 1):
            for el in itertools.combinations(candidate_indices, i):
                index_subsets.append(list(el))

        candidate_simplices = []
        # for each combination of candidate indices, check if this would be a legal misc gap simplex
        # if the edges from the combination are set to homogeneous
        for comb_ind, combination in enumerate(index_subsets):
            candidate_simplex = copy.deepcopy(simplex)
            # reset phase blocks
            candidate_simplex.phase_blocks = []

            # set the edges from the combination to homogeneous
            for index in combination:
                pair = hetero_index_pairs[index]
                candidate_simplex.edge_classification[pair[0]][pair[1]] = 0
                candidate_simplex.edge_classification[pair[1]][pair[0]] = 0

            # we check for all vertices, if they are only connected to heterogeneous
            # edges (which means they represent a phase) or if there are also homogeneous
            # edges (and collect those to check if they form a lower dim simplex)
            homogeneous_edges = []
            for i in range(num_comp):
                # if all connections are heterogeneous, we have a phase
                if sum(candidate_simplex.edge_classification[i]) > 0.99 * (num_comp - 1):
                    candidate_simplex.phase_blocks.append([i])

                else:
                    for j in range(i + 1, num_comp):
                        if candidate_simplex.edge_classification[i][j] < 0.01:
                            homogeneous_edges.append([i, j])

            # now we cluster all homogeneous edges (two edges are in the same cluster,
            # if they have one index in common).
            clusters = []
            cluster = []
            # if we add new edges to this cluster, we store a copy here, so that we check also for
            # those, if there are connections left in homogeneous edges
            todo = []
            while len(homogeneous_edges) > 0 or len(todo) > 0:
                # we always compare the remaining homogeneous edges with a current edge
                if len(todo) == 0:
                    # if to_do is empty, a new cluster was started
                    current_edge = homogeneous_edges[0]
                    homogeneous_edges.remove(homogeneous_edges[0])
                    cluster.append(current_edge)

                else:
                    current_edge = todo[0]
                    todo.remove(todo[0])

                # here we store the found connections
                to_remove = []
                for i, edge in enumerate(homogeneous_edges):
                    # each edge occurs only once in homogeneous edges (due to the construction)
                    if current_edge[0] in edge or current_edge[1] in edge:
                        to_remove.append(i)

                # add to cluster and to to_do
                for i in to_remove:
                    cluster.append(homogeneous_edges[i])
                    todo.append(homogeneous_edges[i])

                # remove from homogeneous edges
                for i in reversed(to_remove):
                    homogeneous_edges.remove(homogeneous_edges[i])

                # if we did not find any new edges for the cluster and do not have anything in
                # the to_do list left, we need a new cluster
                if len(to_remove) == 0 and len(todo) == 0:
                    clusters.append(cluster)
                    cluster = []

            # for each cluster, we check now, if it is exactly a low-dimensional simplex (not less and
            # more, as then we cannot model the phase split linearly and we omit this misc gap simplex)
            omit_candidate_simplex = False
            for i, cluster in enumerate(clusters):
                # it is enough to check if the cluster is a k simplex, where k+1 is the number of points
                # in the cluster. attention: up to now we just stored the edges of the cluster, so we have
                # to get the unique points first
                point_ind_list = []
                for edge in cluster:
                    if edge[0] not in point_ind_list:
                        point_ind_list.append(edge[0])

                    if edge[1] not in point_ind_list:
                        point_ind_list.append(edge[1])

                # now we just check for every point index, if there are exactly k edges containing this
                # index in the cluster (which means we would have a simplex
                for point_index in point_ind_list:
                    edge_count = 0
                    for edge in cluster:
                        if point_index in edge:
                            edge_count = edge_count + 1

                    # if this condition is not fulfilled only once, we can stop
                    if edge_count != len(point_ind_list) - 1:
                        omit_candidate_simplex = True
                        break

                # we add the point list, which specifies the phase
                candidate_simplex.phase_blocks.append(point_ind_list)

            # if this is a split, which we can model, we add the simplex
            if not omit_candidate_simplex:
                vert_num = 0
                for cl in candidate_simplex.phase_blocks:
                    for vert in cl:
                        vert_num = vert_num + 1

                # just for safety
                if vert_num != num_comp:
                    return 1

                candidate_simplices.append([comb_ind, combination, candidate_simplex])

        simplex_to_return = None
        # get std for normal simplex, if we have isoact, don't change it...
        std_border = 0.05

        _, std = miscibility_analysis.act_mean_std_analysis(simplex, gE_model, temperature, num_comp,
                                                            discretized_system, miscibility_gap_simplices=None)
        stat_to_ret = np.max(std)

        # we have to reduce if there is a short edge
        if len(candidate_simplices) > 0 and (must_reduce or stat_to_ret > std_border):
            # if there are more candidates, we first search for the simplex with the least phase blocks
            # if this is not unique, we return the simplex, with the largest minimal hetero edge length
            cand_num_phases = np.zeros(len(candidate_simplices))
            min_hetero_edge_len = np.zeros(len(candidate_simplices))
            for ind, lis in enumerate(candidate_simplices):
                cand_num_phases[ind] = len(lis[-1].phase_blocks)
                current_min = float('inf')
                for i in range(num_comp):
                    for j in range(i + 1, num_comp):
                        if lis[-1].edge_classification[i][j] == 1:
                            edge_len = miscibility_analysis.distance_for_reduce(lis[-1].points_molar_fractions[i],
                                                                                lis[-1].points_molar_fractions[j])

                            if edge_len < current_min:
                                current_min = edge_len

                min_hetero_edge_len[ind] = current_min

            min_phases = min(cand_num_phases)
            counter = 0
            cands_with_min_phases = []
            min_phases_min_hetero_lens = []
            for ind, c in enumerate(candidate_simplices):
                if np.abs(min_phases - cand_num_phases[ind]) < 0.1:
                    counter = counter + 1
                    cands_with_min_phases.append(c)
                    min_phases_min_hetero_lens.append(min_hetero_edge_len[ind])

            second_reduced_cands = []
            if len(cands_with_min_phases) > 1:
                max_min_hetero_len = np.max(min_phases_min_hetero_lens)
                for ind, c in enumerate(cands_with_min_phases):
                    if np.abs(min_phases_min_hetero_lens[ind] - max_min_hetero_len) < 0.0001:
                        second_reduced_cands.append(c)

                if len(second_reduced_cands) <= 1:
                    simplex_to_return = second_reduced_cands[0][-1]

                else:
                    # if this still is not enough, we check for the highest average len
                    # in the hetero edges
                    averages = np.zeros(len(second_reduced_cands))
                    for c_ind, c in enumerate(second_reduced_cands):
                        counter = 0
                        for iw in range(len(c[-1].edge_classification)):
                            for jw in range(iw + 1, len(c[-1].edge_classification)):
                                if c[-1].edge_classification[iw][jw] == 1:
                                    counter = counter + 1
                                    averages[c_ind] = averages[c_ind] + miscibility_analysis.distance_for_reduce(
                                        c[-1].points_molar_fractions[iw], c[-1].points_molar_fractions[jw])

                        averages[c_ind] = averages[c_ind] / counter

                    simplex_to_return = second_reduced_cands[np.argmax(averages)][-1]

            else:
                simplex_to_return = cands_with_min_phases[0][-1]

        return simplex_to_return, stat_to_ret

    @staticmethod
    def act_mean_std_analysis(simplex, gE_model, temperature, num_comp, discretized_system, miscibility_gap_simplices):
        """
        for a misc gap simplex, get the middle as feed and analyze the occurring split
        for isoactivity

        if some components are not present in some split flowrates, we omit those for analysis
        """
        # get middle feed
        feed_middle = simplex.get_middle()
        phases_flowrates, _ = miscibility_analysis.find_phase_split(feed_middle, simplex, discretized_system,
                                                                    miscibility_gap_simplices, num_comp)

        # conversion to molar fractions
        phases_mfr = [fr / sum(fr) for fr in phases_flowrates]

        # get activity coefficients and activities
        act, act_x_mfr = miscibility_analysis.isoactivity(gE_model, temperature, phases_mfr)

        border_to_be_present = 0.003

        # get mean in every index
        means = np.zeros(num_comp)
        counters = np.zeros(num_comp)
        for i, mfr in enumerate(phases_mfr):
            for j in range(num_comp):
                # check if component i is present
                if mfr[j] > border_to_be_present:
                    counters[j] = counters[j] + 1
                    means[j] = means[j] + act_x_mfr[i][j]

        for j in range(num_comp):
            if counters[j] > 0:
                means[j] = means[j] / counters[j]

        # get std in every index
        stds = np.zeros(num_comp)
        counters = np.zeros(num_comp)
        for i, mfr in enumerate(phases_mfr):
            for j in range(num_comp):
                # check if component i is present
                if mfr[j] > border_to_be_present:
                    stds[j] = stds[j] + np.square(act_x_mfr[i][j] - means[j])
                    counters[j] = counters[j] + 1

        for j in range(num_comp):
            if counters[j] > 0:
                stds[j] = stds[j] / counters[j]
                stds[j] = np.sqrt(stds[j])

        return means, stds

    @staticmethod
    def find_phase_split(feed_molar_flowrates, relevant_simplex, discretized_system, miscibility_gap_simplices,
                         num_comp):
        """
        returned is a list with arrays containing the respective molar flowrates of the phases
        """
        # ensure non negative flowrates
        feed_molar_flowrates = np.clip(feed_molar_flowrates, 0, None)
        if sum(feed_molar_flowrates) > 0:
            feed_molar_fractions = feed_molar_flowrates / sum(feed_molar_flowrates)
            # check if it is contained in a simplex with a phase split
            feed_cartesian = discretized_system.transform_molar_fr_to_cartesian(feed_molar_fractions)

            # if the relevant simplex is not already given, try to find it
            if relevant_simplex is None:
                in_gap = False
                for simplex_ind, simplex in enumerate(miscibility_gap_simplices):
                    if miscibility_analysis.point_in_simplex_via_bary(simplex, feed_cartesian):
                        in_gap = True
                        relevant_simplex = miscibility_gap_simplices[simplex_ind]
                        break

                if not in_gap:
                    return [feed_molar_flowrates], relevant_simplex

            # we get the barycentric coordinates of our feed with respect to the relevant simplex
            ext_feed_cartesian = np.ones(num_comp)
            ext_feed_cartesian[1:] = feed_cartesian
            bary_feed_rel_simplex = np.matmul(relevant_simplex.matrix_inv, ext_feed_cartesian)

            # special case: the feed is already exactly lying on one phase block, in this case
            # we just return the feed for this phase and zeros for the rest
            # to check this, for each phase block, we add up the respective barycentric coords
            # of the feed wrt the relevant simplex
            summed_bary_rel_simplex = np.zeros(len(relevant_simplex.phase_blocks))
            for i, pb in enumerate(relevant_simplex.phase_blocks):
                for j in pb:
                    summed_bary_rel_simplex[i] = summed_bary_rel_simplex[i] + bary_feed_rel_simplex[j]

            # if the sum in one phase block is almost 1, we have no real split
            if np.abs(1 - np.max(summed_bary_rel_simplex)) < 0.001:
                splits_flowrates = []
                for i in range(len(relevant_simplex.phase_blocks)):
                    if i == np.argmax(summed_bary_rel_simplex):
                        splits_flowrates.append(feed_molar_flowrates)

                    else:
                        splits_flowrates.append(np.zeros(len(feed_molar_flowrates)))

            else:
                # the split ratios are the sum of the barycentric coordinates of the points, which
                # belong to the respective phase block (proof will be given in some publication)
                num_phases = len(relevant_simplex.phase_blocks)
                split_ratios = np.zeros(num_phases)

                # this way we get the split ratios and with those, we can compute the mfrs of the
                # phases. if we have those, we can get the flowrates.
                mfr_phases = []
                for i, block in enumerate(relevant_simplex.phase_blocks):
                    split_ratios[i] = sum([bary_feed_rel_simplex[j] for j in block])
                    phase_cart = np.zeros(num_comp - 1)
                    for j in range(len(block)):
                        phase_cart = phase_cart + ((bary_feed_rel_simplex[block[j]] / split_ratios[i]) *
                                                   relevant_simplex.points_coordinates_cart[block[j]])

                    mfr_phases.append(discretized_system.transform_cartesian_to_molar_fr(phase_cart))

                splits_flowrates = miscibility_analysis.get_split_flowrates(feed_molar_flowrates, mfr_phases,
                                                                            split_ratios)

            return splits_flowrates, relevant_simplex

        else:
            # we just return the (empty feed)
            return [feed_molar_flowrates], relevant_simplex

    @staticmethod
    def get_split_flowrates(molar_flowrates_feed, phases_mfr, split_ratio):
        """
        phases_mfr contains the molar fractions of the phases, split ratio is a vector with nonnegative
        entries summing up to 1, function returns the flowrates of the ordered phases
        """
        epsilon = 0.0001
        split_flowrates = []
        for i in range(len(split_ratio) - 1):
            flowrates_phase = np.zeros(len(molar_flowrates_feed))
            total_flowrate_phase = sum(molar_flowrates_feed) * split_ratio[i]
            for j in range(len(molar_flowrates_feed)):
                flowrates_phase[j] = total_flowrate_phase * phases_mfr[i][j]

            split_flowrates.append(flowrates_phase)

        # set the last phase
        last_phase_flowrates = molar_flowrates_feed
        for fr in split_flowrates:
            last_phase_flowrates = last_phase_flowrates - fr

        # check for safety
        if np.min(last_phase_flowrates) < -1 * epsilon * sum(molar_flowrates_feed):
            print(molar_flowrates_feed, split_flowrates, last_phase_flowrates)
            return None

        split_flowrates.append(last_phase_flowrates)

        return split_flowrates

    @staticmethod
    def point_in_simplex_via_bary(simplex, point_cartesian):
        """
        for a simplex class and cartesian coordinates of a point.

        we just get the barycentric coordinates of the point (we assume
        that the required matrices exist already in the simplex).
        """
        epsilon = 0.0001
        bary_coords = np.matmul(simplex.matrix_inv, np.array([1] + list(point_cartesian)))

        # this should never happen
        if np.abs(sum(bary_coords) - 1) > epsilon:
            return None

        # check if all coords are in the interval [0, 1]
        if np.min(bary_coords) < -1 * epsilon or np.max(bary_coords) > 1 + epsilon:
            return False

        else:
            return True

    @staticmethod
    def isoactivity(ge_model, temperature, mfr_list):
        activities = np.zeros((len(mfr_list), len(mfr_list[0])))
        act_x_mfr = np.zeros((len(mfr_list), len(mfr_list[0])))
        for i, mfr in enumerate(mfr_list):
            for j in range(len(mfr)):
                if 0.999 > mfr[j] > 0.001:
                    activities[i][j] = ge_model.compute_activity_coefficient(mfr, j, temperature)
                    act_x_mfr[i][j] = activities[i][j] * mfr[j]

        return activities, act_x_mfr

    @staticmethod
    def get_index_separation(len_list, num_actors):
        """
        given the len of an array and a number of actors, return start and end indices to
        separate the workload
        """
        indices = [0]
        part = int(len_list / num_actors)

        while len(indices) < num_actors + 1:
            indices.append(indices[-1] + part)

        indices[-1] = len_list

        return indices

    @staticmethod
    def compute_delta_g_mix(molar_fractions, gE_model, temperature):
        """
        R = 8.314 J/(K mol)
        a_j = x_j * gamma_j
        mu_j = g_j^pure + R * T * ln a_j
        g = sum_j x_j * mu_j
        g_mix = g - sum_j x_j * g_j^pure

        g_mix = R * T * sum_j ln x_j * gamma_j
        see ryll2012 and rowlinson1982 chapter 4
        """
        epsilon = 0.0001
        delta_g_mix = 0
        for j in range(len(molar_fractions)):
            # exclude pure components and also prevent log(0)
            if 1 - epsilon > molar_fractions[j] > epsilon:
                activity_coefficient = gE_model.compute_activity_coefficient(molar_fractions, j, temperature)
                delta_g_mix = delta_g_mix + (molar_fractions[j] * np.log(molar_fractions[j] * activity_coefficient))

        delta_g_mix = delta_g_mix * temperature * 8.314

        return delta_g_mix

    @staticmethod
    def distance_for_reduce(p_1_mfr, p_2_mfr):
        return np.sqrt(sum(np.square(p_1_mfr - p_2_mfr)))

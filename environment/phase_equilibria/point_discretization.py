import numpy as np
import itertools
import os


class PointDisc:
    def __init__(self, num_comp, recursion_steps, load, store, path=None):
        """
        num_comp: the number of components and should be >= 2
        recursion steps: defines the level of discretization and should be > 1
        load: if this discretization already has been stored, we can load it this way
        store: if load is False, the discretization will be constructed and stored if this is True

        a surrounding simplex for the component system will be defined (each pure component is a vertice).
        contrary to the simplex discretization we just get the points here to be faster.
        """
        self.num_comp = num_comp  # number of components
        self.n = self.num_comp - 1  # we need a n simplex in R^n to store the surrounding simplex
        self.recursion_steps = recursion_steps
        self.epsilon = 0.0001  # for comparisons

        # path variable for load and store
        if load or store:
            if path is None:
                self.std_path = os.path.join(os.getcwd(), "data", str(self.num_comp) + "_" + str(self.recursion_steps))

            else:
                self.std_path = os.path.join(path, "data", str(self.num_comp) + "_" + str(self.recursion_steps))

            self.filename = os.path.join(self.std_path, str(self.num_comp) + "_" + str(self.recursion_steps))

            # ensure that the path exists
            if not os.path.isdir(self.std_path):
                os.mkdir(self.std_path)

        self.vertices_outer_simplex = self.construct_outer_simplex()

        # to get the barycentric coordinates lambda for a point p in R^n, we use the matrix A, where the first row
        # contains only ones and the columns below are given by the vertices of the outer simplex, A * lambda = (1, p)
        self.matrix_mfr_to_cart, self.matrix_cart_to_mfr = self.get_basis_change(self.vertices_outer_simplex)

        self.points_mfr = []
        self.points_cart = []

        self.stepsize = 1 / int(2 ** self.recursion_steps)

        # load discretization if specified
        if load:
            self.points_mfr = np.load(self.filename + "_molar_fr_p" + ".npy")
            self.points_cart = np.load(self.filename + "_cart_coords_p" + ".npy")

        else:
            # if we do not load anything, we construct the discretization
            # add pure components as first points
            for v in self.vertices_outer_simplex:
                self.points_cart.append(v)
                mfr = self.transform_cartesian_to_molar_fr(v)
                self.points_mfr.append(mfr)

            self.get_points(base=int(2 ** self.recursion_steps))

            if store:
                np.save(self.filename + "_molar_fr_p", self.points_mfr)
                np.save(self.filename + "_cart_coords_p", self.points_cart)

    def get_points(self, base):
        """
        just loop over all combinations with itertools
        """
        todo = list(itertools.combinations_with_replacement(list(range(base + 1)), self.num_comp - 1))
        stepsize = 1 / base

        index = 0
        while len(todo) > 0:
            index = index + 1
            combination = todo.pop()
            # if sum is zero, then it is the last pure
            if 0 < sum(combination) <= base:
                # pures are already added
                if max(combination) != base:
                    perm = itertools.permutations(combination)
                    for perm_el in set(perm):
                        self.points_mfr.append(np.array(list(perm_el) + [base - sum(perm_el)]) * stepsize)
                        self.points_cart.append(self.transform_molar_fr_to_cartesian(self.points_mfr[-1]))

    def construct_outer_simplex(self):
        # first, construct a regular n simplex in [0,1]^n
        # (explained in the common literature or wiki...)
        vertices_outer_simplex = []
        for i in range(self.n):
            basis_vector = np.zeros(self.n)
            basis_vector[i] = 1 / np.sqrt(2)
            vertices_outer_simplex.append(basis_vector)

        # the last point
        vertices_outer_simplex.append(np.ones(self.n) * (1 + np.sqrt(self.n + 1)) / (self.n * np.sqrt(2)))

        if self.num_comp == 3:
            # rotation with psi
            psi = 2 * np.pi * 285 / 360
            rotation_matrix = np.array([[np.cos(psi), -1 * np.sin(psi)], [np.sin(psi), np.cos(psi)]])
            for i in range(len(vertices_outer_simplex)):
                vertices_outer_simplex[i] = np.matmul(rotation_matrix, vertices_outer_simplex[i])

        return vertices_outer_simplex

    def transform_molar_fr_to_cartesian(self, molar_fractions):
        """
        A * lambda = (1, p), we cut off the first entry
        """

        return np.matmul(self.matrix_mfr_to_cart, molar_fractions)[1:]

    def transform_cartesian_to_molar_fr(self, cartesian_point):
        """
        lambda = A_inv * (1, p)
        """
        vector = np.empty(self.n + 1)
        vector[0] = 1
        vector[1:] = cartesian_point

        return np.matmul(self.matrix_cart_to_mfr, vector)

    @staticmethod
    def euclidean_distance(p1, p2):
        return np.sqrt(sum(np.square(p1 - p2)))

    @staticmethod
    def get_basis_change(vertices_cartesian):
        """
        for given vertices in cartesian coordinates of a simplex, get matrix
        A (first row only ones, columns below are given by vertices) and A_inv.
        A * lambda = (1, p), lambda barycentric coordinates, p cartesian
        """
        matrix = np.empty((len(vertices_cartesian), len(vertices_cartesian)))
        matrix[0] = np.ones(len(vertices_cartesian))
        for i in range(1, len(vertices_cartesian)):
            for j in range(len(vertices_cartesian)):
                matrix[i][j] = vertices_cartesian[j][i - 1]

        return matrix, np.linalg.inv(matrix)

    @staticmethod
    def volume_simplex(vertices):
        """
        for a n-simplex in R^n
        """
        return np.abs(np.linalg.det(vertices[1:] - vertices[0])) / np.math.factorial(len(vertices) - 1)

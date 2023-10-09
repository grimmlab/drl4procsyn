import numpy as np
import matplotlib.pyplot as plt

import environment.phase_equilibria.point_discretization as point_discretization


class Plotter:
    def __init__(self, num_comp):
        self.num_comp = num_comp

        # rec steps does not really matter, we just need the functions from the class
        self.discretization = point_discretization.PointDisc(num_comp=num_comp, recursion_steps=7,
                                                             load=False, store=False)

        # to be able to plot the simplex for the whole component system, we just construct it
        # in the same way as in the simplex discretization
        self.vertices_outer_simplex = self.discretization.construct_outer_simplex()

        # the transformation matrices may be useful at some point
        self.matrix_mfr_to_cart, self.matrix_cart_to_mfr = self.discretization.get_basis_change(
            self.vertices_outer_simplex)

        if self.num_comp == 4:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111, projection='3d')

    def plot_vle(self, regions, color_regions, sing_points, color_sing_points):
        if self.num_comp == 3:
            plt.gca().set_aspect('equal', adjustable='box')
            for region in regions:
                self.plot_distillation_region(region, color_regions)

            for s_p in sing_points:
                self.plot_singular_point(s_p, color_sing_points)

    def plot_distillation_region(self, region, color):
        """
        region is of class distillation_region
        """
        for j in range(2):
            for i in range(len(region.boundaries[j]) - 1):
                region_plot = plt.plot(
                    [region.boundaries[j][i][0], region.boundaries[j][i + 1][0]],
                    [region.boundaries[j][i][1], region.boundaries[j][i + 1][1]],
                    color=color
                )

                self.add_arrow(region_plot[0])

    def plot_singular_point(self, point, color):
        plt.plot(point.cart_coords[0], point.cart_coords[1], marker="o", color=color)

    def plot_delta_g_mix(self, graph, color):
        """
        we get a graph, which contains in the first entries the cartesian coords and in the last entry delta g mix
        """
        if self.num_comp == 2:
            graph_y = np.array([i[-1] for i in graph])
            graph_x = np.array([self.transform_cartesian_to_molar_fr(i[:-1])[0] for i in graph])

            # sort points (maybe not done during simplex disc)
            real_y = np.array([graph_y[i] for i in np.argsort(graph_x)])
            real_x = np.array([graph_x[i] for i in np.argsort(graph_x)])

            plt.plot(real_x, real_y, color=color)

    def plot_outer_simplex(self, color):
        if self.num_comp == 3:
            plt.gca().set_aspect('equal', adjustable='box')
            points = [self.vertices_outer_simplex[j] for j in range(self.num_comp)]
            for j in range(self.num_comp):
                plt.plot([points[j - 1][0], points[j][0]], [points[j - 1][1], points[j][1]], color=color, linewidth=1)

        elif self.num_comp == 4:
            points = [self.vertices_outer_simplex[j] for j in range(self.num_comp)]
            for i in range(self.num_comp):
                for j in range(i+1, self.num_comp):
                    self.ax.plot([points[i][0], points[j][0]], [points[i][1], points[j][1]],
                                 zs=[points[i][2], points[j][2]], color=color)

    def plot_misc_gap_simplex(self, simplex, color, size=None):
        """
        simplex is class miscibility_gap_simplex
        """
        if self.num_comp == 2:
            # we just plot the points
            for i in range(self.num_comp):
                plt.plot(simplex.points_molar_fractions[i][0], 0, marker="o", markersize=size, color=color)

        elif self.num_comp == 3:
            for i in range(self.num_comp):
                for j in range(i+1, self.num_comp):
                    plt.plot([simplex.points_coordinates_cart[i][0], simplex.points_coordinates_cart[j][0]],
                             [simplex.points_coordinates_cart[i][1], simplex.points_coordinates_cart[j][1]],
                             color=color, linewidth=1)

        elif self.num_comp == 4:
            for i in range(self.num_comp):
                for j in range(i+1, self.num_comp):
                    self.ax.plot([simplex.points_coordinates_cart[i][0], simplex.points_coordinates_cart[j][0]],
                                 [simplex.points_coordinates_cart[i][1], simplex.points_coordinates_cart[j][1]],
                                 zs=[simplex.points_coordinates_cart[i][2], simplex.points_coordinates_cart[j][2]],
                                 color=color)

    def plot_split(self, feed_flowrates, flowrates_phases, color_feed, color_phases, color_lines):
        """
        feed_flowrates is array, flowrates phases a list of arrays
        """
        # only possible if feed is non empty
        if sum(feed_flowrates) > 0:
            # get molar fractions first
            feed_mfr = feed_flowrates / sum(feed_flowrates)
            feed_cart = self.transform_molar_fr_to_cartesian(feed_mfr)
            cart_phases = []
            for i, fr in enumerate(flowrates_phases):
                if sum(fr) > 0:
                    mfr = fr / sum(fr)
                    cart = self.transform_molar_fr_to_cartesian(mfr)
                    cart_phases.append(cart)

                else:
                    cart_phases.append(None)
                    print("\nphase ", i, " is empty\n")

            if self.num_comp == 2:
                # we just plot a line (molar fractions), put dots for the phases and feed
                plt.xlim(0, 1)
                plt.yticks([])
                for cart in cart_phases:
                    if cart is not None:
                        mfr = self.transform_cartesian_to_molar_fr(cart)
                        plt.plot(mfr[0], 0, marker="o", markersize=10, color=color_phases)
                        plt.plot([feed_mfr[0], mfr[0]], [0, 0], color=color_lines)

                plt.plot(feed_mfr[0], 0, marker="o", markersize=10, color=color_feed)

            elif self.num_comp == 3:
                # plot points, plot lines between feed and phases
                for cart in cart_phases:
                    if cart is not None:
                        plt.plot(cart[0], cart[1], marker="o", markersize=10, color=color_phases)
                        plt.plot([feed_cart[0], cart[0]], [feed_cart[1], cart[1]], color=color_lines)

                plt.plot(feed_cart[0], feed_cart[1], marker="o", markersize=10, color=color_feed)

            elif self.num_comp == 4:
                # plot points, plot lines between feed and phases
                for cart in cart_phases:
                    if cart is not None:
                        self.ax.plot(cart[0], cart[1], cart[2], marker="o", markersize=10, color=color_phases)
                        self.ax.plot([feed_cart[0], cart[0]], [feed_cart[1], cart[1]], zs=[feed_cart[2], cart[2]],
                                     color=color_lines)

                self.ax.plot(feed_cart[0], feed_cart[1], feed_cart[2], marker="o", markersize=10, color=color_feed)

        else:
            print("\nfeed empty\n")

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

    @staticmethod
    def add_arrow(line, size=30):
        """
        Plot an arrow inside a ternary (used for distillation regions).
        """
        color = line.get_color()

        xdata = line.get_xdata()
        ydata = line.get_ydata()

        start_ind = 0
        end_ind = 1

        line.axes.annotate('',
                           xytext=(xdata[start_ind], ydata[start_ind]),
                           xy=(0.5 * (xdata[start_ind] + xdata[end_ind]),
                               0.5 * (ydata[start_ind] + ydata[end_ind])),
                           arrowprops=dict(arrowstyle="->", color=color),
                           size=size
                           )

    @staticmethod
    def save_plot(path):
        plt.savefig(path)
        plt.close()

    @staticmethod
    def show_plot():
        plt.show()
        plt.close()

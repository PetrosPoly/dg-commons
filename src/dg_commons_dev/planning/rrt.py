import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon, LineString
from typing import List, Callable, Optional, Tuple
from shapely.geometry.base import BaseGeometry
from dataclasses import dataclass
from dg_commons_dev.planning.rrt_utils.sampling import uniform_sampling
from dg_commons_dev.planning.rrt_utils.steering import straight_to
from dg_commons_dev.planning.rrt_utils.nearest_neighbor import distance_cost, naive
from dg_commons_dev.planning.rrt_utils.sampling import RectangularBoundaries, BaseBoundaries
from dg_commons_dev.planning.rrt_utils.utils import Node, move_vehicle
from dg_commons.sim.models.vehicle import VehicleGeometry


@dataclass
class RRTParams:
    path_resolution: float = 0.5
    """ Resolution of points on path """
    max_iter: int = 500
    """ Maximal number of iterations """
    goal_sample_rate: float = 5
    """ Rate at which, on average, the goal position is sampled in % """
    sampling_fct: Callable[[BaseBoundaries, Node, float], Node] = uniform_sampling
    """ 
    Sampling function: takes sampling boundaries, goal node, goal sampling rate and returns a sampled node
    """
    steering_fct: Callable[[Node, Node, float, float, float], Node] = straight_to
    """ 
    Steering function: takes two nodes (start and goal); Maximum distance; Resolution of path; Max curvature
    and returns the new node fulfilling the passed requirements
    """
    distance_meas: Callable[[Node, Node], float] = distance_cost
    """ 
    Formulation of a distance between two nodes, in general not symmetric: from first node to second
    """
    max_distance: float = 3
    """ 
    Maximum distance between a node and its nearest neighbor wrt distance_meas
    """
    max_distance_to_goal: float = 3
    """ Max distance to goal """
    nearest_neighbor_search: Callable[[Node, List[Node], Callable[[Node, Node], float]], int] = naive
    """ 
    Method for nearest neighbor search. Searches for the nearest neighbor to a node through a list of nodes wrt distance
    function.
    """
    vehicle_geom: VehicleGeometry = VehicleGeometry.default_car()
    """ vehicle geometry """
    enlargement_factor: Tuple[float, float] = (1.5, 1.5)
    """ Proportional length and width min distance to keep from obstacles"""


class RRT:
    """
    Class for RRT planning
    """

    def __init__(self,
                 start: Node,
                 goal: Node,
                 obstacle_list: List[BaseGeometry],
                 sampling_bounds: BaseBoundaries,
                 params: RRTParams = RRTParams()
                 ):
        """
        Parameters set up
        @param start: Starting node
        @param goal: Goal node
        @param obstacle_list: List of shapely objects representing obstacles
        @param sampling_bounds: Boundaries in the samples space
        @param params: RRT parameters
        """
        self.start: Node = start
        self.end: Node = goal
        self.node_list: List[Node] = []
        self.boundaries: BaseBoundaries = sampling_bounds
        self.obstacle_list = obstacle_list

        self.expand_dis: float = params.max_distance
        self.max_dist_to_goal: float = params.max_distance_to_goal
        self.path_resolution: float = params.path_resolution
        self.goal_sample_rate: float = params.goal_sample_rate
        self.max_iter: int = params.max_iter

        self.steering_fct: Callable[[Node, Node, float, float, float], Node] = params.steering_fct
        self.sampling_fct: Callable[[BaseBoundaries, Node, float], Node] = params.sampling_fct
        self.distance_meas: Callable[[Node, Node], float] = params.distance_meas
        self.nearest: Callable[[Node, List[Node], Callable[[Node, Node], float]], int] = params.nearest_neighbor_search

        self.path: Optional[List[Node]] = None
        self.vg: VehicleGeometry = params.vehicle_geom
        self.enl_f: Tuple[float, float] = params.enlargement_factor

    def planning(self, search_until_max_iter: bool = False) -> Optional[List[Node]]:
        """
        RRT Dubin path planning
        @param search_until_max_iter: flag for whether to search until max_iter
        @return: sequence of nodes corresponding to path found or None if no path was found
        """
        # TODO: implement search until max iteration

        self.node_list = [self.start]
        for i in range(self.max_iter):
            rnd_node = self.sampling_fct(self.boundaries, self.end, self.goal_sample_rate)
            nearest_ind = self.nearest(rnd_node, self.node_list, self.distance_meas)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steering_fct(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                self.node_list.append(new_node)

            if self.distance_meas(self.node_list[-1], self.end) <= self.max_dist_to_goal:
                final_node = self.steering_fct(self.node_list[-1], self.end, self.expand_dis)
                if self.check_collision(final_node, self.obstacle_list):
                    return self.generate_final_course(len(self.node_list) - 1)

        return None

    def generate_final_course(self, goal_index: int) -> List[Node]:
        """
        Generate list of nodes composing the path
        @param goal_index: index of the goal node
        @return: the generated list
        """
        node = self.node_list[goal_index]
        path = [self.end]
        while node.parent:
            for (ix, iy) in zip(reversed(node.path_x), reversed(node.path_y)):
                temp_node: Node = Node(ix, iy)
                path.append(temp_node)
            node = node.parent
        path.append(self.start)
        path.reverse()
        self.path = path

        return path

    def plot_results(self, create_animation: bool = False) -> None:
        """
        Tool to plot and save the results
        """
        assert self.path is not None
        plt.clf()
        for node in self.node_list:
            if node.parent:
                plt.plot(node.path_x, node.path_y, "-g")

        for ob in self.obstacle_list:
            if isinstance(ob, Polygon):
                x, y = ob.exterior.xy
                plt.plot(x, y)
            if isinstance(ob, LineString):
                plt.plot(*ob.xy)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis("equal")
        plt.axis([-2, 15, -2, 15])
        plt.grid(True)

        plt.plot([node.x for node in self.path], [node.y for node in self.path], '-r')
        plt.grid(True)
        plt.gca().set_aspect('equal', adjustable='box')

        enlargement_f: float = 0.1
        min_enlargement: float = 0
        x_lim = self.boundaries.x_bounds()
        delta = enlargement_f * (x_lim[1] - x_lim[0]) + min_enlargement
        x_lim = (x_lim[0] - delta, x_lim[1] + delta)
        plt.xlim(x_lim)

        y_lim = self.boundaries.y_bounds()
        delta = enlargement_f * (y_lim[1] - y_lim[0]) + min_enlargement
        y_lim = (y_lim[0] - delta, y_lim[1] + delta)
        plt.ylim(y_lim)
        plt.savefig("test")

        if create_animation:

            from matplotlib.animation import FuncAnimation
            plt.style.use('seaborn-pastel')

            fig = plt.figure()
            ax = plt.axes(xlim=x_lim, ylim=y_lim)
            lines = []
            line, = ax.plot([], [], lw=3)
            lines.append(line)
            line, = ax.plot([], [], lw=3)
            lines.append(line)
            line, = ax.plot([], [], lw=3)
            lines.append(line)
            for _ in self.obstacle_list:
                line, = ax.plot([], [], lw=3)
                lines.append(line)

            def init():
                lines[0].set_data([], [])
                lines[1].set_data([], [])
                lines[2].set_data([node.x for node in self.path], [node.y for node in self.path])
                for i, obs in enumerate(self.obstacle_list):
                    idx = 3 + i
                    if isinstance(obs, Polygon):
                        x, y = obs.exterior.xy
                        lines[idx].set_data(x, y)
                    if isinstance(obs, LineString):
                        lines[idx].set_data(*obs.xy)
                return lines[0], lines[1], lines[2], lines[3], lines[4],

            def animate(i):
                node = self.path[i]
                node_next = self.path[i + 1]

                p1 = (node.x, node.y)
                p2 = (node_next.x, node_next.y)
                vehicle = move_vehicle(self.vg, p1, p2)
                x, y = vehicle.exterior.xy
                lines[0].set_data(x, y)

                vehicle = move_vehicle(self.vg, p1, p2, self.enl_f)
                x, y = vehicle.exterior.xy
                lines[1].set_data(x, y)
                return lines[0], lines[1], lines[2], lines[3], lines[4],

            anim = FuncAnimation(fig, animate, init_func=init,
                                 frames=len(self.path) - 1, interval=20, blit=True)

            anim.save('car_moving.gif', writer='imagemagick')

    def check_collision(self, node: Node, obstacle_list: List[BaseGeometry]) -> bool:
        """
        Check if the connection between the passed node and its parent causes a collision
        @param node: the node considered
        @param obstacle_list: list of shapely obstacles
        @return: whether it is causing a collision
        """
        if node is None:
            return False
        return_val: bool = True

        assert len(node.path_x) == len(node.path_y)
        n = len(node.path_x) - 1
        for i in range(n):
            p = (node.path_x[i], node.path_y[i])
            p_next = (node.path_x[i + 1], node.path_y[i + 1])
            line = move_vehicle(self.vg, p, p_next, self.enl_f)
            for obs in obstacle_list:
                if line.intersects(obs):
                    return_val = False

        return return_val


def main(gx=6.0, gy=10.0):
    print("start " + __file__)

    # ====Search Path with RRT====
    obstacleList = [Polygon(((5, 4), (5, 6), (10, 6), (10, 4), (5, 4))), LineString([Point(0, 8), Point(5, 8)])]
    # Set Initial parameters

    bounds: BaseBoundaries = RectangularBoundaries((-2, 15, -2, 15))
    rrt = RRT(
        start=Node(0, 0),
        goal=Node(gx, gy),
        sampling_bounds=bounds,
        obstacle_list=obstacleList,
        # play_area=[0, 10, 0, 14]
        )
    path = rrt.planning()
    rrt.plot_results(create_animation=True)


if __name__ == '__main__':
    main()

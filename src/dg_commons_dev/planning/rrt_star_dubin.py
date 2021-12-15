import matplotlib.pyplot as plt
import numpy as np
from dg_commons_dev.planning.rrt_utils.dubins_path_planning import dubins_path_planning, plot_arrow
from dg_commons_dev.planning.rrt import RRT
from dg_commons_dev.planning.rrt_star import RRTStar, RRTStarParams
from shapely.geometry import Point, Polygon, LineString
from dg_commons_dev.planning.rrt_utils.utils import Node
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple
from dg_commons_dev.planning.rrt_utils.sampling import uniform_sampling, BaseBoundaries, RectangularBoundaries
from dg_commons_dev.planning.rrt_utils.steering import dubin_curves
from dg_commons_dev.planning.rrt_utils.nearest_neighbor import \
    distance_cost, naive, distance_angle_cost, dubin_distance_cost
from dg_commons_dev.planning.rrt_utils.sampling import BaseBoundaries
from shapely.geometry.base import BaseGeometry
import math


@dataclass
class RRTStarDubinParams(RRTStarParams):
    path_resolution: float = 0.1
    """ Resolution of points on path """
    max_iter: int = 1000
    """ Maximal number of iterations """
    goal_sample_rate: float = 10
    """ Rate at which, on average, the goal position is sampled in % """
    sampling_fct: Callable[[BaseBoundaries, Node, float], Node] = uniform_sampling
    """ 
    Sampling function: takes sampling boundaries, goal node, goal sampling rate and returns a sampled node
    """
    steering_fct: Callable[[Node, Node, float, float, float], Node] = dubin_curves
    """ 
    Steering function: takes two nodes (start and goal); Maximum distance; Resolution of path; Max curvature
    and returns the new node fulfilling the passed requirements
    """
    distance_meas: Callable[[Node, Node, ...], float] = dubin_distance_cost
    """ 
    Formulation of a distance between two nodes, in general not symmetric: from second node to first
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
    connect_circle_dist: float = 50.0
    """ Radius of near neighbors is proportional to this one """
    max_curvature: float = 0.2
    """ Maximal curvature in Dubin curve """
    enlargement_factor: Tuple[float, float] = (1.5, 1.5)
    """ Proportional length and width min distance to keep from obstacles"""


class RRTStarDubins(RRT):
    """
    Class for RRT planning with Dubins path
    """

    def __init__(self,
                 start: Node,
                 goal: Node,
                 obstacle_list: List[BaseGeometry],
                 sampling_bounds: BaseBoundaries,
                 params: RRTStarDubinParams = RRTStarDubinParams()
                 ):
        """
        Parameters set up
        @param start: Starting node
        @param goal: Goal node
        @param obstacle_list: List of shapely objects representing obstacles
        @param sampling_bounds: Boundaries in the samples space
        @param params: RRT Dubin parameters
        """
        super().__init__(start, goal, obstacle_list, sampling_bounds, params)
        self.curvature: float = params.max_curvature
        self.connect_circle_dist: float = params.connect_circle_dist

    def planning(self, search_until_max_iter: bool = False):
        """
        RRT star Dubin path planning
        @param search_until_max_iter: flag for whether to perform every iteration
        @return: list of path positions or None if the path was not found
        """

        self.node_list = [self.start]
        for i in range(self.max_iter):
            print("Iter:", i, ", number of nodes:", len(self.node_list))
            rnd = self.sampling_fct(self.boundaries, self.end, self.goal_sample_rate)
            nearest_ind = self.nearest(rnd, self.node_list, self.distance_meas, self.curvature)
            new_node = self.steering_fct(self.node_list[nearest_ind], rnd, self.expand_dis,
                                         self.path_resolution, self.curvature)

            if self.check_collision(new_node, self.obstacle_list):
                near_indexes = self.find_near_nodes(new_node)
                new_node = self.choose_parent(new_node, near_indexes)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_indexes)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)
        else:
            print("Cannot find path")

        return None

    def rewire(self, new_node: Node, near_inds: List[int]) -> None:
        """
        For each node in near_inds, this will check if it is cheaper to
        arrive to them from new_node.
        In such a case, this will re-assign the parent of the nodes in
        near_inds to new_node.
        @param new_node: Node randomly added which can be joined to the tree
        @param near_inds: A list of indices of the self.new_node which contains nodes within a circle of a given radius
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steering_fct(new_node, near_node, self.expand_dis, self.path_resolution, self.curvature)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node.x = edge_node.x
                near_node.y = edge_node.y
                near_node.cost = edge_node.cost
                near_node.path_x = edge_node.path_x
                near_node.path_y = edge_node.path_y
                near_node.parent = edge_node.parent
                self.propagate_cost_to_leaves(new_node)

    def choose_parent(self, new_node: Node, near_inds: List[int]) -> Optional[Node]:
        """
        Computes the cheapest point to new_node contained in the list
        near_inds and set such a node as the parent of new_node.
        @param new_node: Randomly generated node with a path from its neared point.
        @param near_inds: Indices of the nodes that are near new_node
        @return: New node with the chosen parent. None if it was not possible to compute
        """
        if not near_inds:
            return None

        costs = []
        for i in near_inds:
            near_node = self.node_list[i]
            t_node = self.steering_fct(near_node, new_node, self.expand_dis, self.path_resolution, self.curvature)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        min_ind = near_inds[costs.index(min_cost)]
        new_node = self.steering_fct(self.node_list[min_ind], new_node, self.expand_dis, self.path_resolution,
                                     self.curvature)
        new_node.cost = min_cost

        return new_node

    def find_near_nodes(self, new_node: Node) -> List[int]:
        """
        Defines a circle centered on new_node
        Returns all nodes of the three that are inside this ball
        @param new_node: New randomly generated node, without collisions between its nearest node
        @return: List with the indices of the nodes inside the ball of radius r
        """
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))
        r = min(r, self.expand_dis)

        dist_list = [self.distance_meas(node, new_node, self.curvature) for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r**2]
        return near_inds

    def generate_final_course(self, goal_index: int) -> List[Node]:
        """
        Generate list of nodes composing the path
        @param goal_index: index of the goal node
        @return: the generated list
        """
        path = []

        node = self.end
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
        Tool for plotting results
        """
        super().plot_results(create_animation)
        plot_arrow(
            self.start.x, self.start.y, self.start.yaw)
        plot_arrow(
            self.end.x, self.end.y, self.end.yaw)

    def calc_new_cost(self, from_node: Node, to_node: Node) -> float:
        """
        Compute new cost to go of to_node considering previous node plus edge cost
        @param from_node:
        @param to_node:
        @return:
        """
        _, _, _, _, course_length, cost = dubins_path_planning(
            from_node.x, from_node.y, from_node.yaw,
            to_node.x, to_node.y, to_node.yaw, self.curvature)

        return from_node.cost + cost

    def search_best_goal_node(self) -> Optional[int]:
        """
        Return index of the last node of the path to the goal. From that, the whole path can be constructed.
        @return: The index
        """
        final_goal_indexes = []
        for (i, node) in enumerate(self.node_list):
            cost = self.distance_meas(self.end, node, self.curvature)
            if cost <= self.max_dist_to_goal:
                final_goal_indexes.append(i)

        if not final_goal_indexes:
            return None

        min_cost = min([self.node_list[i].cost for i in final_goal_indexes])
        for i in final_goal_indexes:
            if self.node_list[i].cost == min_cost:
                if not (self.node_list[i].x == self.end.x and
                        self.node_list[i].y == self.end.y and self.node_list[i].yaw == self.end.yaw):
                    self.end = self.steering_fct(self.node_list[i], self.end, self.expand_dis,
                                                 self.path_resolution, self.curvature)
                    if not self.check_collision(self.end, self.obstacle_list):
                        self.node_list.remove(self.node_list[i])
                        return None
                return i

        return None

    def propagate_cost_to_leaves(self, parent_node: Node) -> None:
        """
        Update all costs affected by the structure change
        @param parent_node: Node, whose cost has changed
        """
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)


def main():
    print("Start " + __file__)
    # ====Search Path with RRT====
    obstacleList = [Polygon(((5, 4), (5, 6), (10, 6), (10, 4), (5, 4))), LineString([Point(0, 8), Point(5, 8)])]

    # Set Initial parameters
    start = Node(0.0, 0.0, np.deg2rad(0.0))
    goal = Node(10.0, 10.0, np.deg2rad(0.0))
    rand_area = RectangularBoundaries((-2, 15, -2, 15))

    rrt_dubins = RRTStarDubins(start, goal, obstacleList, rand_area)
    path = rrt_dubins.planning(False)

    # Draw final path
    rrt_dubins.plot_results(create_animation=True)


if __name__ == '__main__':
    main()

from __future__ import annotations
from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes, draw_networkx_labels, all_simple_paths
import networkx as nx
from typing import List, Optional, Tuple, Mapping, Set, Dict, Any, Callable, Union
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.common.file_reader import CommonRoadFileReader
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, LineCollection
from dg_commons.sim import SimObservations, SimTime
from dg_commons import PlayerName
from copy import copy, deepcopy
from shapely.geometry import shape, Polygon, LinearRing
from shapely.strtree import STRtree
from shapely.affinity import scale, rotate
import math
from commonroad.visualization.mp_renderer import MPRenderer

resource_id_factor = 100

def interpolate2d(fraction: float, points: [[float, float], [float, float]]) -> [float, float]:
    """
    Simple interpolation between two points in 2D.
    """
    assert 0.0 <= fraction <= 1.0, "You can only interpolate with a fraction between 0.0 and 1.0"
    x_new = fraction * (points[1][0] - points[0][0]) + points[0][0]
    y_new = fraction * (points[1][1] - points[0][1]) + points[0][1]
    return x_new, y_new


def interpolate1d(fraction: float, points: [float, float]) -> float:
    """
    Simple interpolation between two points in 1D.
    """
    assert 0.0 <= fraction <= 1.0, "You can only interpolate with a fraction between 0.0 and 1.0"
    x_new = fraction * (points[1] - points[0]) + points[0]
    return x_new


# fixme: find input and return types
def get_intermediate_points(previous_beta: float, beta: float, distance_vector: np.ndarray,
                            points_left: np.ndarray, points_right: np.ndarray) -> np.ndarray:
    delta = 1.0 / float(len(distance_vector))
    remainder = beta % delta
    remainder = remainder / delta
    divisor = beta // delta
    previous_divisor = previous_beta // delta
    intermediate_points = []
    if divisor == previous_divisor:
        pass
    elif divisor > previous_divisor:
        # account for case where there may be more than one pair of points
        for point_idx in range(int(divisor - previous_divisor)):
            point = (points_left[divisor + point_idx], points_right[divisor + point_idx],)
            intermediate_points.append(point)

    else:
        raise ValueError("divisor can only be greater or equal than previous_divisor.")

    intermediate_points = np.asarray(intermediate_points)
    return intermediate_points


def get_beta_in_distance_vector(pos: float, points: np.ndarray):
    """
    Compute the progression factor beta in the "distance space".
    :param pos: query point for which to find beta in the distance vector
    :param points: Distance vector
    """
    delta = 1.0 / float(len(points))
    # handle special cases at beginning or end of distance vector
    if pos <= points[0]:
        beta = 0.0
    elif pos >= points[-1]:
        beta = 1.0
    # case when point is inside distance vector
    else:
        idx_before = np.argmin(points < pos) - 1
        idx_after = np.argmax(points > pos)
        fraction = (pos - points[idx_before]) / (points[idx_after] - points[idx_before])
        beta = fraction * delta + idx_before * delta
    return beta


def get_point_from_beta_2d(beta: float, points: np.ndarray):
    """
    Get point from a vector of points, with beta defined as the progression in the distance vector.
    :param beta: progression in distance vector
    :param points: vector of points
    """
    delta = 1.0 / float(len(points))
    remainder = beta % delta
    remainder = remainder / delta
    divisor = beta // delta
    # handle special cases
    if beta == 1.0:
        # point = points[int(divisor)-1]
        point = points[-1]
        pos = point[0], point[1]
    elif beta <= delta:
        pos = interpolate2d(fraction=remainder,
                            points=[points[0], points[1]])
    elif beta + delta >= 1.0:
        pos = interpolate2d(fraction=remainder,
                            points=[points[-2], points[-1]])
    # standard case
    else:
        pos = interpolate2d(fraction=remainder,
                            points=[points[int(divisor)], points[int(divisor + 1)]])
    return pos


def make_polygons_from_lanelet(point_list: List[tuple[tuple[float, float]]]):
    """
    Make list of polygons from a list of points.
    For every tuple, first element is left vertex and second element is right vertex
    """
    polygons = []

    for idx, points in enumerate(point_list[:-1]):
        polygon_vertices = points + point_list[idx + 1]
        polygon_vertices = [vert for vert in polygon_vertices]
        current_poly = Polygon(polygon_vertices)  # check mutability
        polygons.append(current_poly)

    return polygons


# currently not used
def get_lanelet_center_coordinates(lanelet_network: LaneletNetwork, lanelet_id: int) -> np.ndarray:
    center = lanelet_network.find_lanelet_by_id(lanelet_id).polygon.center
    return center


# fixme: temporary workaround to plot with digraph as input. This can be used to avoid deepcopying entire
# dynamicgraph but only Digraph at each simulation timestep.
def get_collections_networkx_temp(road_graph: DiGraph(), ax: Axes = None) -> Tuple[PathCollection, LineCollection,
                                                                                   Mapping[int, plt.Text]]:
    """
    Get collections for plotting a graph on top of a scenario

    :param ax: Axes on which to draw the Artists
    """
    nodes = road_graph.nodes
    edges = road_graph.edges
    cents = []
    for node in nodes.data():
        cents.append(node[-1]['polygon'].center)

    centers = dict(zip(nodes.keys(), cents))

    # set default edge and node colors
    edge_colors = ['k'] * len(road_graph.edges)
    node_colors = ['#1f78b4'] * len(road_graph.nodes)

    # set special node and edge colors depending on node and edge attributes
    for node in nodes.data():
        node_idx = list(nodes).index(node[0])
        if node[1]['goal_of_interest']:
            node_colors[node_idx] = 'magenta'
        if node[1]['ego_occupied_resource']:
            node_colors[node_idx] = 'r'
        if node[1]['occupied_by_agent']:
            node_colors[node_idx] = 'cyan'
        if node[1]['goal']:
            node_colors[node_idx] = 'limegreen'
        if node[1]['start']:
            node_colors[node_idx] = 'gold'

    # color edges possibly occupied by ego
    for edge in edges.data():
        edge_idx = list(edges).index((edge[0], edge[1]))
        if edge[2]['ego_occupied_edge']:
            edge_colors[edge_idx] = 'r'

    # return collections and set zorders
    # the functions draw_* already plots on axes
    nodes_plot = draw_networkx_nodes(G=road_graph, ax=ax, pos=centers, node_size=200, node_color=node_colors)
    nodes_plot.set_zorder(50)

    edges_plot = draw_networkx_edges(G=road_graph, ax=ax, pos=centers, edge_color=edge_colors)
    for edge in range(len(edges_plot)):
        edges_plot[edge].set_zorder(49)

    labels_plot = draw_networkx_labels(G=road_graph, ax=ax, pos=centers)
    for label in labels_plot.keys():
        labels_plot[label].set_zorder(51)

    return nodes_plot, edges_plot, labels_plot


def get_weight_from_lanelets(lanelet_network: LaneletNetwork, id_lanelet_1: int, id_lanelet_2: int) -> float:
    """
    Adapted from Commonroad Route Planner.
    Calculate weights for edges on graph.
    For successor: calculate average length of the involved lanes.
    For right/left adjacent lanes: calculate average road width.
    """
    if id_lanelet_2 in lanelet_network.find_lanelet_by_id(id_lanelet_1).successor:
        length_1 = lanelet_network.find_lanelet_by_id(id_lanelet_1).distance[-1]
        length_2 = lanelet_network.find_lanelet_by_id(id_lanelet_2).distance[-1]
        return (length_1 + length_2) / 2.0
    # rough approximation by only calculating width on first point of polyline
    elif id_lanelet_2 == lanelet_network.find_lanelet_by_id(id_lanelet_1).adj_left \
            or id_lanelet_2 == lanelet_network.find_lanelet_by_id(id_lanelet_1).adj_right:
        width_1 = np.linalg.norm(lanelet_network.find_lanelet_by_id(id_lanelet_1).left_vertices[0]
                                 - lanelet_network.find_lanelet_by_id(id_lanelet_1).right_vertices[0])
        width_2 = np.linalg.norm(lanelet_network.find_lanelet_by_id(id_lanelet_2).left_vertices[0]
                                 - lanelet_network.find_lanelet_by_id(id_lanelet_2).right_vertices[0])
        return (width_1 + width_2) / 2.0
    else:
        raise ValueError("You are trying to assign a weight but no edge exists.")


def zero(a: Any):
    return 0.0


class PredDict:
    """
    Class to work with dictionaries.
    """

    def __init__(self, players: List[PlayerName], goals: List[List[int]], entry: Union[bool, float] = True):
        self.data = {}
        self.give_structure_dict(players=players, goals=goals, entry=entry)  # gives structure to data dictionary

    @staticmethod
    def from_dict(skeleton: Dict[PlayerName, Dict[int, bool]], entry: Union[bool, float] = True) -> PredDict:
        players = []
        goals = []
        for player, player_dict in skeleton.items():
            if player.lower() == 'ego':
                continue
            players.append(player)
            player_goals = []
            for goal, goal_data in player_dict.items():
                player_goals.append(goal)
            goals.append(player_goals)
        return PredDict(players=players, goals=goals, entry=entry)

    def give_structure_dict(self, players: List[PlayerName], goals: List[List[int]],
                            entry: Union[bool, float] = True) -> None:
        for index, player in enumerate(players):
            self.add_player_to_dict(player=player, goals=goals[index], data=len(goals[index]) * [entry])
        return

    def add_player_to_dict(self, player: PlayerName, goals: List[Optional[int]], data: List[Any]) -> None:
        assert len(goals) == len(data), 'Goals and data need to have the same number of elements.'
        temp = dict.fromkeys(goals, None)
        for i, (goal, value) in enumerate(temp.items()):
            temp[goal] = data[i]

        self.data[player] = copy(temp)
        temp.clear()
        return

    def add_datapoint_to_dict(self, player: PlayerName, goal: int, data: Any) -> None:
        self.data[player][goal] = data
        return

    # check out utils_toolz valmap
    def valfun(self, func: Callable) -> None:
        for player, goals in self.data.items():
            for goal in goals:
                self.data[player][goal] = func(self.data[player][goal])
        return

    def set_to_zero(self) -> None:
        self.valfun(func=zero)

    """def initialize_prior(self, distribution: str) -> None:
        for player, goals in self.data.items():
            if distribution == "Uniform":
                goals_list = list(goals.keys())
                uniform = np.ones((1, len(goals_list))) * 1.0 / float(len(goals_list))
                self.add_player_to_dict(player=player, goals=goals_list, data=uniform[0].tolist())
            elif distribution != "Uniform":
                raise NotImplementedError
        self.normalize()
        return"""

    # question: check this works
    # fixme: division by 0 ignored. Should handle here or somewhere else?
    def normalize(self) -> None:
        """
        normalize according to func
        """
        for player, player_dict in self.data.items():
            if sum(player_dict.values()) == 0.0 or sum(player_dict.values()) == 0:  # just for debugging
                print("Division by zero encountered. Fixme.")
            norm_factor = 1.0 / sum(player_dict.values())
            for goal in player_dict.items():
                a = goal
                b = self.data[player][goal[0]]
                self.data[player][goal[0]] = self.data[player][goal[0]] * norm_factor
        return

    def __add__(self, other) -> None:
        """
        Element-wise sum for PredDict.
        """
        if self.data.keys() != other.data.keys():
            raise TypeError('Keys of summing elements are not matching')
        for player, goals in self.data.items():
            if goals.keys() != other.data[player].keys():
                raise TypeError('Goals for player ' + str(player) + ' are not matching in both elements.')
        for player, goals in self.data.items():
            for goal in goals:
                self.data[player][goal] += other.data[player][goal]
        return

    def __mul__(self, other) -> None:
        """
        Element-wise multiplication. Can be either between two PredDict or between
        a PredDict and a scalar.
        """
        if isinstance(other, PredDict):
            if self.data.keys() != other.data.keys():
                raise TypeError('Keys of multiplying elements are not matching')
            for player, goals in self.data.items():
                if goals.keys() != other.data[player].keys():
                    raise TypeError('Goals for player ' + str(player) + ' are not matching in both elements.')
            for player, goals in self.data.items():
                for goal in goals:
                    self.data[player][goal] *= other.data[player][goal]
        elif isinstance(other, float):
            for player, goals in self.data.items():
                for goal in goals:
                    a = self.data[player][goal]
                    self.data[player][goal] = self.data[player][goal] * other
                    # fixme: why is [0] needed (...[player][goal] is a List, but where does that come from?
        else:
            raise TypeError('You can only multiply by another PredictionDictionary or by a scalar')

        return

    def __sub__(self, other) -> None:
        other.__mul__(-1.0)
        return self.__add__(other)


class Prediction:
    """
        Class to handle probabilities, costs and rewards on DynamicGraphs.
    """

    def __init__(self, goals_dict: Dict[PlayerName, Dict[int, bool]]):
        # prediction parameters
        self.params: PredictionParams = PredictionParams(goals_dict=goals_dict)

        # dictionary containing information about reachability of each goal by each agent
        self.reachability_dict: PredDict = PredDict.from_dict(skeleton=goals_dict)

        # probabilities
        # dictionary containing probability of each goal for each agent
        self.prob_dict: PredDict = PredDict.from_dict(skeleton=goals_dict, entry=0.0)

        # rewards
        # dictionary containing optimal rewards from current position to goal
        self.suboptimal_reward: PredDict = PredDict.from_dict(skeleton=goals_dict)
        # dictionary containing optimal rewards from initial position to goal
        self.optimal_reward: PredDict = PredDict.from_dict(skeleton=goals_dict)


class PredictionParams:
    """
    Class for storing prediction parameters
    """

    def __init__(self, goals_dict: Dict[PlayerName, Dict[int, bool]], beta: float = 1.0, distribution: str = "Uniform"):
        self.distribution = distribution
        self.beta = beta
        self.priors: PredDict = PredDict.from_dict(skeleton=goals_dict)
        self._initialize_prior()

    def _initialize_prior(self) -> None:
        for player, goals in self.priors.data.items():
            if self.distribution == "Uniform":
                goals_list = list(goals.keys())
                uniform = np.ones((1, len(goals_list))) * 1.0 / float(len(goals_list))
                self.priors.add_player_to_dict(player=player, goals=goals_list, data=uniform[0].tolist())
            elif self.distribution != "Uniform":
                raise NotImplementedError
        self.priors.normalize()
        return


class RoadGraph:
    """
    Class to represent Commonroad lane networks as directed graphs.
    Forbidden lanelets (for cars) are not added to the network.
    Forbidden lanelets can also be manually.
    """
    # forbidden_lanelet_types = ["busLane", "bicycleLane", "sidewalk"]
    uncrossable_line_markings = ["solid", "broad_solid"]  # no vehicle can cross this

    def __init__(self, lanelet_network: LaneletNetwork, excluded_lanelets: Optional[List[int]] = None):
        """
        Create a digraph from a Commonroad lanelet network.
        :param lanelet_network: Commonroad lanelet network
        :param excluded_lanelets: lanelets that should not be added to the graph
        """
        self.road_graph: DiGraph = DiGraph()
        if excluded_lanelets is None:
            excluded_lanelets = list()
        self.excluded_lanelets = excluded_lanelets
        self._init_road_graph(lanelet_network)

    def _init_road_graph(self, lanelet_network: LaneletNetwork):
        """
        Construct road graph from road network. All lanelets are added, including the lanelet type.
        Lanelets that are in "excluded_lanelets" will be omitted.
        Edges are constructed between a lanelet and its successor, its right adjacent, and left adjacent.
        If a lane between adjacent lanelets is uncrossable, edge is omitted.
        Length of lane is considered and added as weight.
        """
        # add all nodes
        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelet
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            # add permissible lanelet to graph
            polygon = lanelet_network.find_lanelet_by_id(lanelet.lanelet_id).polygon
            # center = get_lanelet_center_coordinates(lanelet_network, lanelet.lanelet_id)

            self.road_graph.add_node(lanelet.lanelet_id,
                                     lanelet_type=lanelet.lanelet_type,
                                     polygon=polygon)

            # add edge for all succeeding lanelets
            for id_successor in lanelet.successor:
                # skip excluded lanelet (may be a successor of an allowed lanelet)
                if id_successor in self.excluded_lanelets:
                    continue
                weight = get_weight_from_lanelets(lanelet_network=lanelet_network,
                                                  id_lanelet_1=lanelet.lanelet_id,
                                                  id_lanelet_2=id_successor)
                self.road_graph.add_edge(lanelet.lanelet_id, id_successor, weight=weight)

            # add edge for adjacent right lanelet (if existing)
            if lanelet.adj_right_same_direction and lanelet.adj_right is not None:

                # skip excluded lanelet (may be adj right of an allowed lanelet)
                if lanelet.adj_right in self.excluded_lanelets \
                        or lanelet.line_marking_right_vertices.value \
                        in self.uncrossable_line_markings:
                    continue

                weight = get_weight_from_lanelets(lanelet_network=lanelet_network,
                                                  id_lanelet_1=lanelet.lanelet_id,
                                                  id_lanelet_2=lanelet.adj_right)
                self.road_graph.add_edge(lanelet.lanelet_id, lanelet.adj_right, weight=weight)

            # add edge for adjacent left lanelets (if existing)
            if lanelet.adj_left_same_direction and lanelet.adj_left is not None:

                # skip excluded lanelet (may be adj left of an allowed lanelet)
                if lanelet.adj_left in self.excluded_lanelets \
                        or lanelet.line_marking_left_vertices.value \
                        in self.uncrossable_line_markings:
                    continue

                weight = get_weight_from_lanelets(lanelet_network=lanelet_network,
                                                  id_lanelet_1=lanelet.lanelet_id,
                                                  id_lanelet_2=lanelet.adj_left)
                self.road_graph.add_edge(lanelet.lanelet_id, lanelet.adj_left, weight=weight)

        self.set_default_attributes()

        return

    def set_default_attributes(self) -> None:
        # set default attributes for nodes
        for node in list(self.road_graph.nodes):
            self.set_node_attribute(attribute='start', value=False, node=node)
            self.set_node_attribute(attribute='goal', value=False, node=node)
            self.set_node_attribute(attribute='ego_occupied_resource', value=False, node=node)
            self.set_node_attribute(attribute='occupied_by_agent', value=False, node=node)
            self.set_node_attribute(attribute='goal_of_interest', value=False, node=node)

        # set default attributes for edges
        for edge in list(self.road_graph.edges):
            self.set_edge_attribute(attribute='ego_occupied_edge', value=False, edge=edge)
        return

    def plot_graph(self, file_path: str = None) -> None:
        fig, ax = plt.subplots(figsize=(60, 60))
        _, _, _ = self.get_collections_networkx(ax=ax)
        if file_path is not None:
            plt.savefig(file_path)
        plt.close()
        return

    def get_collections_networkx(self, ax: Axes = None) -> Tuple[PathCollection, LineCollection,
                                                                 Mapping[int, plt.Text]]:
        """
        Get collections for plotting a graph on top of a scenario

        :param ax: Axes on which to draw the Artists
        """
        nodes = self.road_graph.nodes
        edges = self.road_graph.edges
        cents = []
        for node in nodes.data():
            cents.append(node[-1]['polygon'].center)

        centers = dict(zip(nodes.keys(), cents))

        # set default edge and node colors
        edge_colors = ['k'] * len(self.road_graph.edges)
        node_colors = ['#1f78b4'] * len(self.road_graph.nodes)

        # set special node and edge colors depending on node and edge attributes
        for node in nodes.data():
            node_idx = list(nodes).index(node[0])
            if node[1]['goal_of_interest']:
                node_colors[node_idx] = 'magenta'
            if node[1]['ego_occupied_resource']:
                node_colors[node_idx] = 'r'
            if node[1]['occupied_by_agent']:
                node_colors[node_idx] = 'cyan'
            if node[1]['goal']:
                node_colors[node_idx] = 'limegreen'
            if node[1]['start']:
                node_colors[node_idx] = 'gold'

        # color edges possibly occupied by ego
        for edge in edges.data():
            edge_idx = list(edges).index((edge[0], edge[1]))
            if edge[2]['ego_occupied_edge']:
                edge_colors[edge_idx] = 'r'

        # return collections and set zorders
        # the functions draw_* already plots on axes
        nodes_plot = draw_networkx_nodes(G=self.road_graph, ax=ax, pos=centers, node_size=200, node_color=node_colors)
        nodes_plot.set_zorder(50)

        edges_plot = draw_networkx_edges(G=self.road_graph, ax=ax, pos=centers, edge_color=edge_colors)
        for edge in range(len(edges_plot)):
            edges_plot[edge].set_zorder(49)

        labels_plot = draw_networkx_labels(G=self.road_graph, ax=ax, pos=centers)
        for label in labels_plot.keys():
            labels_plot[label].set_zorder(51)

        return nodes_plot, edges_plot, labels_plot

    def get_possible_resources(self, start_node: int, end_node: int) -> Tuple[Set[int], Set[Tuple[int, int]]]:
        """
        Compute all nodes where an agent could be when transiting from start_node to end_node
        Return both occupied nodes and occupied edges

        :param start_node: departure of agent
        :param end_node: destination of agent
        """
        paths = all_simple_paths(self.road_graph, source=start_node, target=end_node)

        occupancy_nodes = set()
        occupancy_edges = set()
        for path in paths:
            # add new nodes only if they are not yet in occupancy_nodes
            occupancy_nodes |= set(path)
            # add new edges only if they are not yet in occupancy_edges
            path_edges = nx.utils.pairwise(path)  # transforms [a,b,c,...] in [(a,b),(b,c),...]
            occupancy_edges |= set(path_edges)
        return occupancy_nodes, occupancy_edges

    def get_occupancy_children(self, occupied_resources: Set[int]) -> List[int]:
        """
        Compute the children of the nodes in the "occupancy zone" as computed by "get_possible_resources"

        :param occupied_resources: part of digraph where the ego could be on his journey to the goal
        """
        children = []
        for node in occupied_resources:
            candidates = list(self.road_graph.successors(node))
            for cand in candidates:
                if cand in occupied_resources:
                    continue
                else:
                    children.append(cand)

        return children

    def is_upstream(self, node_id: int, nodes: Set[int]) -> bool:
        """
        Determine wether a given node is upstream a set of nodes.
        Returns True even if node_id is in nodes

        :param node_id: node id of node we want to know if is upstream
        :param nodes: set of nodes to check against
        """

        children = nx.traversal.bfs_tree(G=self.road_graph, source=node_id).nodes
        is_upstream = (set(children) & nodes) != set()

        return is_upstream

    # tbd: reward function
    def reward_1(self, weight: float):
        return -weight

    def reward_2(self):
        return

    def shortest_paths_rewards(self, start_node: int, end_node: int, reward: Callable) \
            -> Tuple[List[List[int]], List[float]]:  # tbd: what type returned
        """
        Compute all shortest simple paths between two nodes using dijkstra algorithm.
        Weight is considered. If several paths have same (shortest) length, return them all.
        Returns path and reward.

        :param start_node: starting node of path
        :param end_node: ending node of path
        :param reward: function to use to calculate reward of a specific path
        """

        paths = nx.all_shortest_paths(G=self.road_graph, source=start_node, target=end_node,
                                      weight='weight', method='dijkstra')  # question: check if weight is considered
        rewards = []
        paths_list = list(paths)
        for path in paths_list:
            rewards.append(self.get_cost_from_path(path=path, reward=reward))
        return paths_list, rewards

    def get_cost_from_path(self, path: List[int], reward: Callable):
        path_edges = nx.utils.pairwise(path)
        path_reward = 0
        for edge in path_edges:
            path_reward += reward(self.road_graph.get_edge_data(edge[0], edge[1])['weight'])
        return path_reward

    def get_lanelet_by_position(self, position: np.ndarray) -> int:
        """
        Compute lanelet that contains the queried position.
        :param position: query position
        """
        for lanelet_id, lanelet_polygon in list(self.road_graph.nodes(data='polygon')):
            if lanelet_polygon.contains_point(position):
                return lanelet_id

        print("Position: " + str(position) + ". No lanelet found that contains the position you asked for.")

    def set_node_attribute(self, attribute: str, value: Any, node: int) -> None:
        try:
            self.road_graph.nodes[node][attribute] = value
        except KeyError:
            print("Specified node does not exist.")

    def set_edge_attribute(self, attribute: str, value: Any, edge: Tuple[int, int]) -> None:
        try:
            self.road_graph.edges[(edge[0], edge[1])][attribute] = value
        except KeyError:
            print("Specified edge does not exist.")

    '''# for now keep for reference
    # naive implementation: depth_limit is just a user-defined integer. May not be the right choice.
    def breadth_first_search(self, source_node: int, depth_limit: int = None) -> DiGraph:
        edges = list(bfs_edges(G=self.road_graph, source=source_node, depth_limit=depth_limit))
        return DiGraph(edges)

    def generate_plausible_goals(self, lanelet_id: int, depth_limit: int = 1) -> Optional[List[int]]:
        """
        Search for possible goals for an agent currently in a specific lanelet.
        Returns all leaves at depth smaller or equal than depth_limit and
        all nodes at depth equal to depth_limit.

        :param lanelet_id: current lanelet where the observed agent is located
        :param depth_limit: maximum depth at which to search goals
        """
        # create graph from network
        self.create_topology_from_lanelet_network()

        # perform breadth first search
        search_bfs = self.breadth_first_search(source_node=lanelet_id,
                                               depth_limit=depth_limit)
        goals = []
        for depth in range(1, depth_limit + 1):
            descendants_at_depth = descendants_at_distance(G=search_bfs, source=lanelet_id, distance=depth)
            for desc in descendants_at_depth:
                temp = descendants(G=search_bfs, source=desc)
                if descendants(G=search_bfs, source=desc) == {}:
                    goals.append(desc)
                else:
                    continue
            if depth == depth_limit and descendants_at_depth != {}:
                goals.append(list(descendants_at_depth))
                goals = [item for sublist in goals for item in sublist]

        return goals

    # not used for now
    def return_circle(self, centers: List[np.ndarray], radius: float) -> Any: #todo: fix return type
        circles = []
        for pred in range(len(centers)):
            circles.append(Circle((centers[pred]), radius))
        colors = ["r"] * len(circles)

        return list(zip(circles, colors))

    def run_goal_generation(self, lanelet_id: int, depth_limit: int = 1):
        plausible_goals = self.generate_plausible_goals(lanelet_id=lanelet_id, depth_limit=depth_limit)
        goal_center_coordinate = self.get_lanelet_center_coordinates(plausible_goals)

        return self.return_circle(goal_center_coordinate,10)'''


class DynamicRoadGraph(RoadGraph):
    """
        Class to represent dynamic digraphs.
        Extend RoadGraph by adding positions of observed agents and start & end position of the ego
    """

    def __init__(self, lanelet_network: LaneletNetwork, excluded_lanelets: Optional[List[int]] = None):
        """
        """
        super().__init__(lanelet_network=lanelet_network, excluded_lanelets=excluded_lanelets)
        self.ego_start: Optional[int] = None
        self.ego_goal: Optional[int] = None
        self.ego_problem_updated: bool = False

        # initialize start and goal with initial planning problem

        self.locations: Dict[PlayerName, List[Tuple[SimTime, int]]] = {k: [] for k in
                                                                       ['P1', 'Ego']}  # fixme: make general
        # self.goal_dict = PredDict.from_dict()
        # fixme: player names should already be somewhere in the dynamic graph!
        #  tbd: better way than overwriting later when we have start/goal/other info?
        self.predictions: Optional[Prediction] = None
        self.graph_storage = []

    # get start and goal of ego
    # question: why would we need more than one goal for the ego? Here keep only the first.
    # tbd: may only use one problem instead of a set
    # tbd: currently called in __init__ of PredAgent.
    # Should be called somewhere else in case goals are updated?
    def start_and_goal_info(self, problem: PlanningProblem) -> None:
        """
        Get start an goal for ego vehicle from planning problem of scenario.
        Write start and goal in self.ego_start and self.ego_goal, respectively.
        Update of goal and start during simulation is supported

        :param problem: Planning Problem as defined by Commonroad
        """

        # fixme: at first iteration use get_lanelet_by_position.
        # get_lanelet_by_position does not work because it needs an initial id. Generalize this.

        start = self.get_lanelet_by_position(position=problem.initial_state.position)
        self.ego_start = start
        goal = problem.goal.lanelets_of_goal_position
        if goal is not None:
            self.ego_goal = goal[0][0]  # fixme: keep only first
        else:
            next_node = list(self.road_graph.successors(start))
            self.ego_goal = next_node[0]  # fixme: only works if there is a node. Nonsense, just to make it run.
            print("No goal in planning problem set. Goal is set to last lanelet "
                  "after merging successor of start lanelet.")  # fixme: comment is what should be done
        print("Start of planning problem: ")
        print(self.ego_start)
        print("Goal of planning problem: ")
        print(self.ego_goal)

        # there is a new plan
        self.ego_problem_updated = True
        return

    def get_lanelet_by_position_restricted(self, player: PlayerName, position: np.ndarray) -> int:
        """
        Version get_lanelet_by_position in road_graph with restricted search.
        Position is used to find lanelet id only by querying on current lanelet
        polygon and the subsequent lanelet polygons.
        :param position: query position
        """
        previous_id = self.locations[player][-1][1]
        # check if position is in previous polygon
        if self.road_graph.nodes[previous_id]['polygon'].contains_point(position):
            return previous_id

        # check if position is in subsequent polygons
        next_ids = self.road_graph.neighbors(previous_id)
        for next_id in list(next_ids):
            if self.road_graph.nodes[next_id]['polygon'].contains_point(position):
                return next_id

        print("Position: " + str(position) + ". No lanelet found that contains"
                                             " the position you asked for by using restricted algorithm.")

    # fixme: would be more efficient to only pass DiGraph and not entire graph
    def keep_track(self):
        self.graph_storage.append(deepcopy(self.road_graph))

    def instantiate_prediction_object(self):
        self.predictions = Prediction(self.get_goals_dict(players=['P1', 'Ego']))  # fixme: make general!!!!!

    def update_locations(self, sim_obs: SimObservations):
        t = sim_obs.time
        for player, player_obs in sim_obs.players.items():
            # leave out occupancy info for now. #todo: get position from X
            player_pos = np.array([player_obs.state.x, player_obs.state.y])

            # fixme: quickfix to handle initial state, when self.locations is empty. Needs general fix.
            if not self.locations[player]:
                lanelet_id = self.get_lanelet_by_position(position=player_pos)
            else:
                lanelet_id = self.get_lanelet_by_position_restricted(player=player, position=player_pos)
            self.locations[player].append((t, lanelet_id))
        return

    def get_past_path(self, player: PlayerName) -> List[int]:
        """
        Get past history. Skip repeating nodes.

        :param player: query player
        """
        past_path = []
        previous_id = None
        for t, current_id in self.locations[player]:
            if previous_id == current_id:
                continue
            past_path.append(current_id)
            previous_id = current_id
        return past_path

    # tbd: check this works properly
    def reachable_goals(self, goals: List[int], player: PlayerName) -> Dict[int, bool]:
        """
        From the goals of interest of the Ego, compute which ones are reachable by player.

        :param goals: goals of interest for the ego
        :param player: player for which to compute reachable goals
        """
        # latest position of player
        current_node = self.get_player_location(player)
        reachability_dict = {}

        for goal in goals:
            reachability_dict[goal] = self.is_upstream(node_id=current_node, nodes={goal})
        return reachability_dict

    # tbd: check this does what it should
    # fixme: make PredDict out of this
    def get_goals_dict(self, players: List[PlayerName]) -> Dict[PlayerName, Dict[int, bool]]:
        """
        Compute reachable goals for each player (apart from Ego) and store in a dictionary.
        :param players: list of players for which to compute goals
        """
        player_dict = {}
        goals_dict = {}

        # determine goals of interest just outside Ego occupancy
        # fixme: make function that returns goals directly
        goals = []
        for node_id, node_data in self.road_graph.nodes(data=True):
            if node_data['goal_of_interest']:
                goals.append(node_id)

        for player in players:
            if player.lower() == 'ego':
                continue
            reachability_dict = self.reachable_goals(goals=goals,
                                                     player=player)  # tbd: issue with immutability? Should not
            for goal, goal_reachable in reachability_dict.items():
                if goal_reachable:
                    player_dict[goal] = True  # for now, fill with True. Probabilities will be calculated

            goals_dict[player] = copy(player_dict)
            player_dict.clear()

        return goals_dict

    # fixme: workaround for reachability update
    def update_reachability(self, players: List[PlayerName]):
        for player in players:
            if player.lower() == 'ego':
                continue
            # fixme: make function getgoals()
            goals = []
            for node_id, node_data in self.road_graph.nodes(data=True):
                if node_data['goal_of_interest']:
                    goals.append(node_id)

            current_node = self.get_player_location(player)
            for goal in goals:
                is_reachable = self.is_upstream(node_id=current_node, nodes={goal})
                self.predictions.reachability_dict.add_datapoint_to_dict(player=player, goal=goal,
                                                                         data=is_reachable)

    # fixme: find better name
    def compute_rewards_and_paths(self) -> None:
        """
        Compute past rewards from t=0 to now, optimal path from now to goal and associated reward,
        optimal paths from initial position to goal and associated reward. Computation done for all players and all
        reachable goals.
        """

        for player, player_goals in self.predictions.reachability_dict.data.items():
            if player.lower() == 'ego':
                continue
            # loop over goals for each player
            for goal, goal_reachability in player_goals.items():
                total_reward = []
                # only consider reachable goals
                if goal_reachability:
                    loc = self.get_player_location(player)
                    # compute optimal path and associated reward from current position to final goal
                    # if self.is_upstream(node_id=loc, nodes={goal}): # check that shortest path can actually be computed
                    partial_shortest_paths, rewards = self.shortest_paths_rewards(start_node=loc,
                                                                                  end_node=goal, reward=self.reward_1)
                    # compute past path and associated reward (from t=0 to now)
                    past_path = self.get_past_path(player=player)
                    past_cost = self.get_cost_from_path(path=past_path, reward=self.reward_1)
                    # else:
                    #    print("From node "+ str(loc) +", goal node "+ str(goal) " can't be reached.")

                    for reward in rewards:
                        total_reward.append(reward + past_cost)

                    # fixme: do this only once at beginning
                    # compute optimal path and associated reward from initial position to goal
                    optimal_path, optimal_reward = self.shortest_paths_rewards(self.locations[player][0][1],
                                                                               end_node=goal, reward=self.reward_1)
                    # tbd: for now consider only one possible path (total_reward[0])
                    self.predictions.suboptimal_reward.add_datapoint_to_dict(player=player, goal=goal,
                                                                             data=total_reward[0])
                    # fixme: this does not need to be calculated every time
                    # fixme : [0] is a workaround
                    self.predictions.optimal_reward.add_datapoint_to_dict(player=player, goal=goal,
                                                                          data=optimal_reward[0])

                # goal is not reachable anymore. Set reward to approx. -Inf
                else:
                    self.predictions.suboptimal_reward.add_datapoint_to_dict(player=player, goal=goal,
                                                                             data=-99999999999.0)

    def compute_goal_probabilities(self) -> None:
        """
        Compute probability of each goal for each agent.
        """
        # fixme: could be made quicker by checking if node is the same as before and ending execution if this holds.
        # prob_dict is initially filled with 0.0
        # posterior update
        self.predictions.prob_dict.set_to_zero()
        self.predictions.prob_dict + self.predictions.suboptimal_reward
        self.predictions.prob_dict - self.predictions.optimal_reward

        self.predictions.prob_dict * self.predictions.params.beta
        self.predictions.prob_dict.valfun(func=np.exp)
        # multiply with prior
        self.predictions.prob_dict * self.predictions.params.priors
        self.predictions.prob_dict.normalize()

        # print("Probabilities predicted:")
        # print(self.predictions.prob_dict.pred_dict)

        return

    def get_player_location(self, player: PlayerName):
        return self.locations[player][-1][1]

    def update_dynamic_graph(self) -> None:
        """
        Update node attributes depending on ego_start, ego_goal and locations of all agents
        """

        if self.ego_problem_updated:
            # reset default values of node and edge attributes
            self.set_default_attributes()

            # compute new edge attributes
            self.set_node_attribute(attribute='start', value=True, node=self.ego_start)
            self.set_node_attribute(attribute='goal', value=True, node=self.ego_goal)

            ego_resources_nodes, ego_resources_edges = \
                self.get_possible_resources(start_node=self.ego_start, end_node=self.ego_goal)
            goals_of_interest = self.get_occupancy_children(ego_resources_nodes)

            for node in ego_resources_nodes:
                self.set_node_attribute(attribute='ego_occupied_resource', value=True, node=node)
            for edge in ego_resources_edges:
                self.set_edge_attribute(attribute='ego_occupied_edge', value=True, edge=edge)
            for node in goals_of_interest:
                self.set_node_attribute(attribute='goal_of_interest', value=True, node=node)

        for player, locations in self.locations.items():
            if player != 'Ego':
                # skip first loop iteration
                if len(locations) > 1:
                    # remove previous location
                    self.set_node_attribute(attribute='occupied_by_agent', value=False, node=locations[-2][1])
                    # set new location
                self.set_node_attribute(attribute='occupied_by_agent', value=player, node=locations[-1][1])

        # updated plan has been processed
        self.ego_problem_updated = False


# tbd: is it even needed?
class PredictionDictionaryIterator:

    def __iter__(self, pred_dict: PredDict):
        self._index = 0
        self._pred_dict = pred_dict

    def __next__(self):
        self._index += 1
        return


# fixme: issue when polygons are too small. These should be merged. How?
def split_lanelet_into_polygons(lanelet: Lanelet, max_length: float) -> List[Polygon]:
    """
    Split a lanelet in smaller polygons with length along centerline smaller or equal to max_length.
    Function respects lanelet boundaries exactly but does not guarantee polygons with uniform lenght.

    :param lanelet: lanelet to divide
    :param max_length: maximum length allowed for a polygon, along centerline.
    """
    assert max_length > 0.0, "You need a maximum cell length that is greater than 0.0."
    polygons = []
    centerline_distance = lanelet.distance
    left_vertices = lanelet.left_vertices
    right_vertices = lanelet.right_vertices
    counter = 0
    for idx, center_length in enumerate(centerline_distance):
        ddistance = centerline_distance[idx] - centerline_distance[idx - 1]
        current_polygon_id = lanelet.lanelet_id*resource_id_factor + counter
        if idx == 0:
            continue
        elif ddistance <= max_length:
            vertices = [left_vertices[idx - 1], right_vertices[idx - 1],
                        right_vertices[idx], left_vertices[idx]]
            current_polygon = Polygon(LinearRing(vertices))
            polygons.append((current_polygon, current_polygon_id))

        elif ddistance > max_length:
            ncells = math.ceil((ddistance / max_length))
            dcell = ddistance / ncells
            cell_frac = 1 / float(ncells)
            left_points = [left_vertices[idx - 1], left_vertices[idx]]
            right_points = [right_vertices[idx - 1], right_vertices[idx]]
            for cell in range(ncells):
                left_start = interpolate2d(fraction=cell * cell_frac, points=left_points)
                left_end = interpolate2d(fraction=(cell + 1) * cell_frac, points=left_points)
                right_start = interpolate2d(fraction=cell * cell_frac, points=right_points)
                right_end = interpolate2d(fraction=(cell + 1) * cell_frac, points=right_points)
                vertices = [left_start, right_start, right_end, left_end]
                current_polygon = Polygon(LinearRing(vertices))
                polygons.append((current_polygon, current_polygon_id))

        counter = counter+1

    return polygons

    """    def split_lanelet(self, lanelet: Lanelet, max_length: float) -> List[Polygon]:
    This function attempts to split polygons uniformly but does not respect lanelet boundaries exactly.
    Split a lanelet in smaller polygons such that max cell size is respected.
    Division happens independently for every lanelet.
    Length is measured along the central line, as euclidean distance.

    assert max_length > 0.0, "You need a maximum cell length that is greater than 0.0."
    length_centerline = lanelet.distance[-1]
    number_of_cells = math.ceil((length_centerline / max_length))
    cell_length_centerline = length_centerline / number_of_cells
    polygons = []
    previous_beta = 0.0
    ## works since between vertices the lines are straight

    for cell in range(0, number_of_cells + 1):
        cell_start_centerline = cell * cell_length_centerline
        polygon_vertices = []
        beta_start = get_beta_in_distance_vector(pos=cell_start_centerline, points=lanelet.distance)
        beta_end = get_beta_in_distance_vector(pos=cell_start_centerline+cell_length_centerline,
                                               points=lanelet.distance)
        cell_start_right = get_point_from_beta_2d(beta=beta_start, points=lanelet.right_vertices)
        cell_start_left = get_point_from_beta_2d(beta=beta_start, points=lanelet.left_vertices)
        cell_end_right = get_point_from_beta_2d(beta=beta_end, points=lanelet.right_vertices)
        cell_end_left = get_point_from_beta_2d(beta=beta_end, points=lanelet.left_vertices)
        intermediate_points = get_intermediate_points(previous_beta=previous_beta, beta=beta_start,
                                                      points_left=lanelet.left_vertices,
                                                      points_right=lanelet.right_vertices,
                                                      distance_vector=lanelet.distance)
        if len(intermediate_points) > 0:
            polygon_vertices.append(intermediate_points)
        polygon_vertices.append((cell_start_left, cell_start_right))
        polygon_vertices.append((cell_end_left, cell_end_right))
        a = polygon_vertices[0]
        c = a[0]
        b = polygon_vertices[1]
        points = [point for point in polygon_vertices]
        current_polygon = Polygon(points)  # check mutability
        polygons.append(current_polygon)
        previous_beta = beta_start

    # polygons = make_polygons_from_lanelet(point_list=splitted_points)
    return polygons"""


class ResourceNetwork:

    uncrossable_line_markings = ["solid", "broad_solid"]  # no vehicle can cross this

    def __init__(self, lanelet_network: LaneletNetwork, max_length: float):
        """

        :param lanelet_network:
        :param max_length: maximum length of a cell
        """
        # get lanelet network and create digraph
        # store in STRTree as well (or only in STRTree)
        self.excluded_lanelets = []
        self.resource_graph: DiGraph = DiGraph()
        self.tree: STRtree
        self._create_rtree(lanelet_network=lanelet_network, max_cell_length=max_length)
        self._init_resource_graph(lanelet_network=lanelet_network, max_cell_length=max_length)

    def _create_rtree(self, lanelet_network: LaneletNetwork, max_cell_length: float):
        resources = []

        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelets
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            new_resources = split_lanelet_into_polygons(
                lanelet=lanelet_network.find_lanelet_by_id(lanelet.lanelet_id), max_length=max_cell_length)
            resources.extend(new_resources)

        resource_polygons = [resource[0] for resource in resources]
        resource_ids = [resource[1] for resource in resources]
        self.tree = STRtree(geoms=resource_polygons, items=resource_ids)
        return

    def set_default_attributes(self) -> None:
        # set default attributes for nodes
        for node in list(self.resource_graph.nodes):
            self.set_node_attribute(attribute='start', value=False, node=node)
            self.set_node_attribute(attribute='goal', value=False, node=node)
            self.set_node_attribute(attribute='ego_occupied_resource', value=False, node=node)
            self.set_node_attribute(attribute='occupied_by_agent', value=False, node=node)
            self.set_node_attribute(attribute='goal_of_interest', value=False, node=node)

        # set default attributes for edges
        for edge in list(self.resource_graph.edges):
            self.set_edge_attribute(attribute='ego_occupied_edge', value=False, edge=edge)
        return

    def set_node_attribute(self, attribute: str, value: Any, node: int) -> None:
        try:
            self.resource_graph.nodes[node][attribute] = value
        except KeyError:
            print("Specified node does not exist.")

    def set_edge_attribute(self, attribute: str, value: Any, edge: Tuple[int, int]) -> None:
        try:
            self.resource_graph.edges[(edge[0], edge[1])][attribute] = value
        except KeyError:
            print("Specified edge does not exist.")

    def _init_resource_graph(self, lanelet_network: LaneletNetwork, max_cell_length: float):
        """
        Construct road graph from road network. All lanelets are added, including the lanelet type.
        Lanelets that are in "excluded_lanelets" will be omitted.
        Each lanelet is divided into cells smaller than max_cell_length.
        Edges are constructed between adjacent polygons.
        If a lane between adjacent lanelets is uncrossable, edges are omitted.
        Weight of a polygon is given by its size.
        """

        resources = {}

        # pre-compute and store all resources for all lanelets
        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelets
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            resources[lanelet.lanelet_id] = split_lanelet_into_polygons(
                lanelet=lanelet_network.find_lanelet_by_id(lanelet.lanelet_id), max_length=max_cell_length)

        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelet
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            # add resources of permissible lanelet to graph
            for current_polygon, polygon_id in resources[lanelet.lanelet_id]:
                self.resource_graph.add_node(polygon_id,
                                             lanelet_type=lanelet.lanelet_type,
                                             polygon=current_polygon)

            # add edges between subsequent resources in a lanelet
            for current_polygon, polygon_idx in resources[lanelet.lanelet_id]:
                # skip last resource of lanelet
                if polygon_idx == resources[lanelet.lanelet_id][-1][1]:
                    continue
                weight = 1
                self.resource_graph.add_edge(polygon_idx, polygon_idx+1, weight=weight)

            # add edge for all succeeding lanelets
            # specifically, connect last resource of a lanelet with first resource of succeeding lanelet
            for id_successor in lanelet.successor:
                # skip excluded lanelet (may be a successor of an allowed lanelet)
                if id_successor in self.excluded_lanelets:
                    continue
                weight = 1
                last_resource_lanelet = resources[lanelet.lanelet_id][-1][1]
                first_resource_successor = resources[id_successor][0][1]
                self.resource_graph.add_edge(last_resource_lanelet, first_resource_successor, weight=weight)

            # add edge for adjacent right lanelet (if existing)
            if lanelet.adj_right_same_direction and lanelet.adj_right is not None:

                # skip excluded lanelet (may be adj right of an allowed lanelet)
                if lanelet.adj_right in self.excluded_lanelets \
                        or lanelet.line_marking_right_vertices.value \
                        in self.uncrossable_line_markings:
                    continue
                current_resources = resources[lanelet.lanelet_id]
                right_resources = resources[lanelet.adj_right]
                for resource, resource_id in current_resources:
                    for right_resource, right_resource_id in right_resources:
                        if resource.intersects(right_resource):
                            weight = 1
                            self.resource_graph.add_edge(resource_id, right_resource_id, weight=weight)

            # add edge for adjacent left lanelet (if existing)
            if lanelet.adj_left_same_direction and lanelet.adj_left is not None:

                # skip excluded lanelet (may be adj right of an allowed lanelet)
                if lanelet.adj_left in self.excluded_lanelets \
                        or lanelet.line_marking_left_vertices.value \
                        in self.uncrossable_line_markings:
                    continue
                current_resources = resources[lanelet.lanelet_id]
                left_resources = resources[lanelet.adj_left]
                for resource, resource_id in current_resources:
                    for left_resource, left_resource_id in left_resources:
                        if resource.intersects(left_resource):
                            weight = 1
                            self.resource_graph.add_edge(resource_id, left_resource_id, weight=weight)

        self.set_default_attributes()

        return



if __name__ == '__main__':
    # scenario_path1 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Zip-1_66_T-1.xml"  # remove e.g. 24
    # scenario_path2 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/NGSIM/Peachtree/USA_Peach-1_1_T-1.xml"  # remove e.g. 52826
    scenario_path3 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/DEU_Muc-1_2_T-1.xml"  # remove e.g. 24
    scenario_path4 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Intersection-1_1_T-1_nando.xml"

    scenario, planning_problem_set = CommonRoadFileReader(scenario_path4).open(lanelet_assignment=True)
    net = scenario.lanelet_network

    obj = ResourceNetwork(lanelet_network=net, max_length=10.0)

    #Plot graph and scenario

    fig, axs = plt.subplots(2, figsize=(50, 50))

    nodes = obj.resource_graph.nodes
    cents = []
    for node in nodes.data():
        cent_point = node[-1]['polygon'].centroid
        cents.append([cent_point.x, cent_point.y])

    cents = dict(zip(nodes.keys(), cents))

    #centers = dict(zip(nodes.keys(), cents))
    nodes_plot = draw_networkx_nodes(G=obj.resource_graph, ax=axs[0], pos=cents, node_size=50)
    edges_plot = draw_networkx_edges(G=obj.resource_graph, ax=axs[0], pos=cents)
    #plt.savefig("graph_debug.png")
    #plt.close()

    #plt.subplots()
    rnd = MPRenderer(ax=axs[1],)
    scenario.draw(rnd)
    #planning_problem_set.draw(rnd)
    rnd.render()
    plt.savefig("scenario.png")
    plt.close()

    #test_lanelet = 3318
    #test_lanelet_adj_right = 3316
    #test_lanelet_adj_left = 3320
    #new_polygons = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet), max_length=10.0)
    #new_polygons_right = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet_adj_right), max_length=10.0)
    #new_polygons_left = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet_adj_left), max_length=10.0)

    inters_c_r = []
    inters_c_l = []
    inters_r_l = []
    inters_r_c = []

    """for idx, polygon_center in enumerate(new_polygons):
        for idx_left, polygon_left in enumerate(new_polygons_left):
            # polygon_left_scaled = scale(deepcopy(polygon_left), xfact=1.2, yfact=1.2)
            # polygon_left_rot = rotate(deepcopy(polygon_left), 30)
            # polygon_left = polygon_left.buffer(0)
            # polygon_center = polygon_center.buffer(0)
            inters_c_l.append((idx, idx_left, polygon_left.intersects(polygon_center)))
            if polygon_left.intersects(polygon_center):
                print("start")
                print(idx)
                print(idx_left)
                print("end")
    print(inters_c_l)
    print("******")"""

    """plt.plot()
    for i, polygon in enumerate(new_polygons_left):
        if i == 0 or i == 1:
            x, y = polygon.exterior.xy
            # swapping values just for plotting
            plt.plot(x[:], y[:], '--')
    for i, polygon in enumerate(new_polygons):
        if i == 0 or i == 1:
            x, y = polygon.exterior.xy
            # swapping values just for plotting
            plt.plot(x[:], y[:], '--')

    plt.savefig("debugging.png")
    plt.close()

    # works
    for idx, polygon_center in enumerate(new_polygons):
        for idx_right, polygon_right in enumerate(new_polygons_right):
            inters_c_r.append((idx, idx_right, polygon_right.intersects(polygon_center)))
            inters_r_c.append((idx, idx_right, polygon_center.intersects(polygon_right)))

    print(inters_c_r)
    print("******")
    """

    """
    for idx, polygon_right in enumerate(new_polygons_right):
        for idx_right, polygon_left in enumerate(new_polygons_left):
            inters_r_l.append((idx, idx_right, polygon_left.intersects(polygon_right)))
            if polygon_left.intersects(polygon_right):
                print("Error")

    print(inters_r_l)
    print("******")

    old_polygon = net.find_lanelet_by_id(test_lanelet).polygon.shapely_object
    old_polygon_right = net.find_lanelet_by_id(test_lanelet_adj_right).polygon.shapely_object
    old_polygon_left = net.find_lanelet_by_id(test_lanelet_adj_left).polygon.shapely_object
    fig, ax = plt.subplots()
    x, y = old_polygon.exterior.xy
    plt.plot(x, y, )
    for i, polygon in enumerate(new_polygons):
        x, y = polygon.exterior.xy
        # swapping values just for plotting
        plt.plot(x[:], y[:], '--')

    x, y = old_polygon_right.exterior.xy
    plt.plot(x, y, )
    for i, polygon in enumerate(new_polygons_right):
        x, y = polygon.exterior.xy
        # swapping values just for plotting
        plt.plot(x[:], y[:], '--')

    x, y = old_polygon_left.exterior.xy
    plt.plot(x, y, )
    for i, polygon in enumerate(new_polygons_left):
        x, y = polygon.exterior.xy
        # swapping values just for plotting
        plt.plot(x[:], y[:], '--')

    plt.savefig("polygon_decomposed_left.png")
    plt.close()"""

    """lanelet = net.find_lanelet_by_id(3316)
    center_line = lanelet.center_vertices
    left_line = lanelet.left_vertices
    right_line = lanelet.right_verticesfig, ax = plt.subplots()
    plt.scatter(center_line[:, 0], center_line[:, 1])
    plt.scatter( right_line[:, 0], right_line[:, 1])
    plt.scatter(left_line[:, 0], left_line[:, 1] )
    plt.savefig("lines_test.png")
    plt.close()"""

    # obj = RoadGraph(lanelet_network=net, planning_problem_set=planning_problem_set)
    # occupancy_nodes, _ = obj.get_possible_resources(start_node=3512, end_node=3450)
    # obj.plot_graph(filename='graph_only', start_node=3512, end_node=3450)

    # test mutability
    """
    players = ["P1", "P2"]

    goals = [[2, 4, 5, 7], [2, 4, 5, 7]]

    skeleton = {}
    player_dict = {}
    for player_id, player in enumerate(players):
        for goal in goals[player_id]:
            if player_id==0:
                player_dict[goal] = True
            else:
                player_dict[goal] = False
            #player_dict.clear()
        skeleton[player] = copy(player_dict) #solved it
    pred_dict_1 = PredDict.from_dict(skeleton=skeleton)

    skeleton["P1"][10]=False
    pred_dict_2 = PredDict.from_dict(skeleton=skeleton)
    a=1"""

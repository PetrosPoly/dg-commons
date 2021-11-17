import time

from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes, draw_networkx_labels, all_simple_paths
import networkx as nx
from typing import List, Optional, Tuple, Mapping, Set, Dict, Any, Callable, Union
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.visualization.mp_renderer import MPRenderer
from commonroad.common.file_reader import CommonRoadFileReader
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, LineCollection
from dg_commons.sim import SimObservations, SimTime
from dg_commons import PlayerName
from matplotlib.animation import FuncAnimation
from copy import copy


# currently not used
def get_lanelet_center_coordinates(lanelet_network: LaneletNetwork, lanelet_id: int) -> np.ndarray:
    center = lanelet_network.find_lanelet_by_id(lanelet_id).polygon.center
    return center


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


class PredDict:
    """
    Class to work with dictionaries.
    """

    def __init__(self, players: List[PlayerName], goals: List[List[int]], entry: Union[bool, float] = True):
        self.data = {}
        self.give_structure_dict(players=players, goals=goals, entry=entry)  # gives structure to data dictionary

    # def __iter__(self):
    # return PredictionDictionaryIterator(self)

    @staticmethod
    def from_dict(skeleton: Dict[PlayerName, Dict[int, bool]], entry: Union[bool, float] = True):
        skeleton_copy = copy(skeleton)  # tbd: needed? sufficient?
        players = []
        goals = []
        for player, player_dict in skeleton_copy.items():
            if player == 'ego':
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

    # tbd: check what this does to dict.
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

    def apply_function_everywhere(self, func: Callable) -> None:
        for player, goals in self.data.items():
            for goal in goals:
                self.data[player][goal] = func(self.data[player][goal])
        return

    def initialize_prior(self, prior: str) -> None:
        for player, goals in self.data.items():
            if prior == "Uniform":
                goals_list = list(goals.keys())
                uniform = np.ones((1, len(goals_list))) * 1.0 / float(len(goals_list))
                self.add_player_to_dict(player=player, goals=goals_list, data=uniform[0].tolist())
            elif prior != "Uniform":
                raise NotImplementedError
        self.normalize()
        return

    # question: check this works
    # fixme: division by 0 ignored. Should handle here or somewhere else?
    def normalize(self) -> None:
        """
        normalize according to func
        """
        for player, player_dict in self.data.items():
            if sum(player_dict.values()) == 0.0:  # just for debugging
                print("Division by zero encountered. Fixme.")
            norm_factor = 1.0 / sum(player_dict.values())
            for goal in player_dict.items():
                a = goal
                b = self.data[player][goal[0]]
                self.data[player][goal[0]] = self.data[player][goal[0]] * norm_factor
        return

    def __add__(self, other) -> None:
        """
        Element-wise sum
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
        Element-wise multiplication. Can be either between two PredictionDictionary or between
        a PredictionDictionary and a scalar.
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
                    self.data[player][goal][0] = self.data[player][goal][0] * other
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
        self.beta: float = 1.0
        self.prior_distribution: str = "Uniform"

        # dictionary containing information about reachability of each goal by each agent
        self.reachability_dict: PredDict = PredDict.from_dict(skeleton=goals_dict)

        # probabilities
        # dictionary containing probability of each goal for each agent
        self.prob_dict: PredDict = PredDict.from_dict(skeleton=goals_dict, entry=0.0)
        # dictionary containing prior probability for each goal
        self.priors: PredDict = PredDict.from_dict(skeleton=goals_dict)
        self.priors.initialize_prior(self.prior_distribution)

        # rewards
        # dictionary containing optimal rewards from current position to goal
        self.suboptimal_reward: PredDict = PredDict.from_dict(skeleton=goals_dict)
        # dictionary containing optimal rewards from initial position to goal
        self.optimal_reward: PredDict = PredDict.from_dict(skeleton=goals_dict)


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
        fig, ax = plt.subplots(figsize=(30, 30))
        _, _, _ = self.get_collections_networkx(ax=ax)
        if file_path is not None:
            plt.savefig(file_path)
        plt.close()
        return

    def get_collections_networkx(self, ax: Axes = None) -> Tuple[
        PathCollection, LineCollection, Mapping[int, plt.Text]]:
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
            if node[1]['goal']:
                node_colors[node_idx] = 'limegreen'
            if node[1]['start']:
                node_colors[node_idx] = 'gold'
            if node[1]['goal_of_interest']:
                node_colors[node_idx] = 'magenta'
            if node[1]['ego_occupied_resource']:
                node_colors[node_idx] = 'r'
            if node[1]['occupied_by_agent']:
                node_colors[node_idx] = 'cyan'

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
        # tbd: architecture design. Pass tuples [(P1,G1), (P1,G2), (P2, G3),...] instead of goals_dict?
        # self.goal_dict = PredDict.from_dict()
        # fixme: players should already be somewhere in the dynamic graph!
        self.predictions: Optional[
            Prediction] = None  # tbd: better way than overwriting later when we have start/goal/other info?

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

        start = self.get_lanelet_by_position(problem.initial_state.position)
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

    def instantiate_prediction_object(self):
        self.predictions = Prediction(self.get_goals_dict(players=['P1', 'Ego']))  # fixme: make general!!!!!

    def update_locations(self, sim_obs: SimObservations):
        t = sim_obs.time
        for player, player_obs in sim_obs.players.items():
            # leave out occupancy info for now. #todo: get position from X
            player_pos = np.array([player_obs.state.x, player_obs.state.y])
            lanelet_id = self.get_lanelet_by_position(player_pos)
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
        From the goals of interest of the Ego, only keep those reachable by a player.

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
        # fixme: make function getgoals()
        for player in players:
            if player.lower() == 'ego':
                continue
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
        Compute rewards from t=0 to now, optimal path from now to goal and associated reward,
        optimal paths from initial position to goal and associated reward. Computation done for all players and all
        reachable goals.
        """
        # fixme: still need to calculate cost up to now.
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
                    #    path_cost =

                    for reward in rewards:
                        total_reward.append(reward + past_cost)

                    # compute optimal path and associated reward from initial position to goal
                    optimal_path, optimal_reward = self.shortest_paths_rewards(self.locations[player][0][1],
                                                                               end_node=goal, reward=self.reward_1)
                    # tbd: for now consider only one possible path (total_reward[0])
                    self.predictions.suboptimal_reward.add_datapoint_to_dict(player=player, goal=goal,
                                                                             data=total_reward[0])
                    # fixme: this does not need to be calculated every time
                    self.predictions.optimal_reward.add_datapoint_to_dict(player=player, goal=goal, data=optimal_reward)
                # goal is not reachable anymore. Set reward to approx. -Inf
                else:
                    self.predictions.suboptimal_reward.add_datapoint_to_dict(player=player, goal=goal,
                                                                             data=-99999999999.0)

    def compute_goal_probabilities(self) -> None:
        """
        Compute probability of each goal for each agent.
        :param suboptimal_rewards:
        :param optimal_rewards:
        :param beta:
        """
        # prob_dict is filled with 0.0
        self.predictions.prob_dict + self.predictions.suboptimal_reward
        self.predictions.prob_dict - self.predictions.optimal_reward
        self.predictions.prob_dict * self.predictions.beta
        self.predictions.prob_dict.apply_function_everywhere(func=np.exp)
        self.predictions.prob_dict * self.predictions.priors
        self.predictions.prob_dict.normalize()

        # print("Probabilities predicted:")
        # print(self.predictions.prob_dict.pred_dict)

        """# ASSUME STUFF ABOVE WILL BE MOVED ELSEWHERE
        player_prob_dict = {}
        for player, player_data in self.predictions.prob_dict.items():
            if player.lower() == 'ego':
                continue
            for goal, reward in player_data.items():
                optimal_reward = self.predictions.optimal_reward[player][goal]
                optimal_reward = optimal_reward[
                    0]  # fixme: keeping only first (we assume there is only one for now). Same cheat at next line
                prob = self.predictions.modified_exponential(optimal_reward=optimal_reward, suboptimal_reward=reward[0])
                player_prob_dict[goal] = prob
            self.predictions.probability_dict[player] = copy(player_prob_dict)
            player_prob_dict.clear()"""

        """# step 4: compute probability depending on 2/3 -> Use Prediction class
        self.predictions.probab_dict
        # step 5: multiply with prior -> Use Prediction class
        for player in goal_dict.keys():
            prior = self.predictions.compute_prior_probability(n_goals=len(goal_dict[player].keys))
            self.predictions.priors[player] = self.predictions.compute_prior_probability()
            prob_after_prior = self.predictions.priors * self.predictions.probab_dict
            # step 6: normalize -> Use Prediction class
            prob_after_prior.normalize()  # won't work like this"""

        # update location in graph for all players. Need function def find_node_by_position
        # fixme: could be made quicker by checking if node is the same as before and ending execution if this holds.
        '''def update_locations(self, sim_obs: SimObservations):
        for player, player_obs in sim_obs.players:
            attribute = player + "_here"  # fixme: not optimal way to store
            # reset previous positions
            for node_id, node in self.road_graph.road_graph.nodes:
                if node[attribute]:
                    node[attribute] = False
            player_pos = sim_obs.players[player].state.X  # leave out occupancy info for now. #todo: get position from X
            # question: can go through player_obs?
            lanelet_id = self.road_graph.get_lanelet_by_position(player_pos)

            self.road_graph.set_node_attribute(attribute=attribute, value=True, node=lanelet_id)
        return'''
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


if __name__ == '__main__':
    scenario_path1 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Zip-1_66_T-1.xml"  # remove e.g. 24
    scenario_path2 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/NGSIM/Peachtree/USA_Peach-1_1_T-1.xml"  # remove e.g. 52826
    scenario_path3 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/DEU_Muc-1_2_T-1.xml"  # remove e.g. 24

    scenario, planning_problem_sett = CommonRoadFileReader(scenario_path3).open(lanelet_assignment=True)
    net = scenario.lanelet_network

    # obj = RoadGraph(lanelet_network=net, planning_problem_set=planning_problem_set)
    # occupancy_nodes, _ = obj.get_possible_resources(start_node=3512, end_node=3450)
    # obj.plot_graph(filename='graph_only', start_node=3512, end_node=3450)

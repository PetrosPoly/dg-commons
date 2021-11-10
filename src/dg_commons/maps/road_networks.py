from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes, draw_networkx_labels, all_simple_paths
import networkx as nx
from typing import List, Optional, Tuple, Mapping, Set, Dict, Any
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.planning.planning_problem import PlanningProblemSet
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad.geometry.shape import Polygon  #question: or LaneletPolygon?
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, LineCollection
from dg_commons.sim import SimObservations, SimTime
from dg_commons import PlayerName


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
            #center = get_lanelet_center_coordinates(lanelet_network, lanelet.lanelet_id)

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
        return

    def plot_graph(self, start_node: int, end_node: int, filename: str = None, ) -> None:
        plt.figure(1)
        plt.grid(True)
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        visited_nodes, special_edges = self.get_possible_resources(start_node=start_node, end_node=end_node)
        goal_nodes = self.get_occupancy_children(occupied_resources=visited_nodes)
        _, _, _ = self.get_collections_networkx(special_edges=special_edges, visited_nodes=visited_nodes,
                                                goal_nodes=goal_nodes, start_node=start_node, end_node=end_node)
        if filename is not None:
            plt.savefig(filename)
        plt.close()

    def get_collections_networkx(self, special_edges: Set[Tuple[int, int]] = None,
                                 visited_nodes: Set[int] = None, goal_nodes: List[int] = None,
                                 start_node: int = None, end_node: int = None) \
            -> Tuple[PathCollection, LineCollection, Mapping[int, plt.Text]]:
        """
        Get collections for plotting a graph on top of a scenario

        :param special_edges: Edges to color differently.
        :param visited_nodes: Nodes to color differently.
        :param goal_nodes: possible other agents goals.
        :param start_node: Departure node of ego.
        :param end_node: Arrival node of ego.
        """
        nodes = self.road_graph.nodes
        cents = []
        for node in nodes.data():
            cents.append(node[-1]['polygon'].center) #tbd: works??

        centers = dict(zip(nodes.keys(), cents))

        # set default edge and node colors
        edge_colors = ['k'] * len(self.road_graph.edges)
        node_colors = ['#1f78b4'] * len(self.road_graph.nodes)

        # color possible goal nodes
        if goal_nodes is not None:
            special_indices = [i for i, item in enumerate(self.road_graph.nodes) if item in goal_nodes]
            for ind in special_indices:
                node_colors[ind] = 'magenta'

        # color visited nodes
        if visited_nodes is not None:
            special_indices = [i for i, item in enumerate(self.road_graph.nodes) if item in visited_nodes]
            for ind in special_indices:
                node_colors[ind] = 'r'

        # color edges on ego's path
        if special_edges is not None:
            special_indices = [i for i, item in enumerate(self.road_graph.edges) if item in special_edges]
            for ind in special_indices:
                edge_colors[ind] = 'r'

        # color ego's start and end node
        if start_node is not None:
            node_colors[list(self.road_graph.nodes).index(start_node)] = 'gold'

        if end_node is not None:
            node_colors[list(self.road_graph.nodes).index(end_node)] = 'limegreen'

        # return collections and set zorders
        # the functions draw_* already plot on axes
        nodes_plot = draw_networkx_nodes(G=self.road_graph, pos=centers, node_size=500, node_color=node_colors)
        nodes_plot.set_zorder(50)

        edges_plot = draw_networkx_edges(G=self.road_graph, pos=centers, edge_color=edge_colors)
        labels_plot = draw_networkx_labels(G=self.road_graph, pos=centers)
        for edge in range(len(edges_plot)):
            edges_plot[edge] = edges_plot[edge].set_zorder(49)
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

    def shortest_path(self, start_node: int, end_node: int): #tbd: what type returned
        paths = nx.all_shortest_paths(G=self.road_graph, source=start_node, target=end_node,
                                      weight=None, method='dijkstra')
        return paths

    def get_lanelet_by_position(self, position: np.ndarray) -> int:
        for lanelet_id, lanelet in self.road_graph.nodes.items(): #question: iterator working?
            if lanelet['polygon'].contains_point(position):
                return lanelet_id
            else:
                print("No lanelet found that contains the position you asked for.")

    def set_node_attribute(self, attribute: str, value: Any, node: int) -> None:
        try:
            self.road_graph[node][attribute] = value
        except KeyError:
            print("Specified node does not exist.")


    # Following functions were written but are not currently used.
    # For now, keep for reference or reusage.

    # naive implementation: depth_limit is just a user-defined integer. May not be the right choice.
    '''def breadth_first_search(self, source_node: int, depth_limit: int = None) -> DiGraph:
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
        # question: will this initialize it if I write "DynamicRoadGraph(lanelet_network)" somewhere else?
        # self.road_graph = road_graph
        # self.planning_problem_set: PlanningProblemSet = planning_problem_set
        self.ego_start: Dict[int, Any] = {}  # fixme: what type should Any be
        self.ego_goal: Dict[int, Any] = {}  # fixme: what type should Any be
        self.locations: Dict[PlayerName, List[Tuple[SimTime, int]]] = {k: [] for k in ['P1', 'Ego']} #fixme: make general

    # get start and goal of ego
    # question: why would we need more than one goal for the ego? Here keep only the first.
    def _start_and_goal_info(self, planning_problem_set: PlanningProblemSet) -> None:
        """
        Get planning problems for ego vehicle from planning problem set of scenario.
        Write planning problems as dictionary in self.ego_start and self.ego_goals.
        If there is more than one goal, only keep first.
        """
        for problem_id, problem in planning_problem_set.planning_problem_dict.items():
            self.ego_start[problem_id] = self.get_lanelet_by_position(problem.initial_state.position)
            self.ego_goal[problem_id] = problem.goal.lanelets_of_goal_position[0]  # fixme: keep only first
        return


    """def get_start_and_goal_ego(self) -> Dict[int, Tuple[int, int]]:

        planning_problems = {}
        # todo: this needs to be validated
        for problem_id, problem in self.planning_problem_set.planning_problem_dict.keys():
            start: int = self.scenario.lanelet_network.find_lanelet_by_position(
                problem.initial_state.position)[0][0]
            goal: int = self.scenario.lanelet_network.find_lanelet_by_position(
                problem.goal.state_list[0].position)[0][0]

            planning_problems[problem_id] = tuple(start, goal)

        return planning_problems"""

    def update_locations(self, sim_obs: SimObservations):
        time = sim_obs.time
        for player, player_obs in sim_obs.players.items():
            player_pos = np.array([player_obs.state.x, player_obs.state.y])  # leave out occupancy info for now. #todo: get position from X
            # question: can go through player_obs directly?
            lanelet_id = self.get_lanelet_by_position(player_pos)
            self.locations[player].append((time, lanelet_id))
        return

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


if __name__ == '__main__':
    scenario_path1 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Zip-1_66_T-1.xml"  # remove e.g. 24
    scenario_path2 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/NGSIM/Peachtree/USA_Peach-1_1_T-1.xml"  # remove e.g. 52826
    scenario_path3 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/DEU_Muc-1_2_T-1.xml"  # remove e.g. 24

    scenario, planning_problem_sett = CommonRoadFileReader(scenario_path3).open(lanelet_assignment=True)
    net = scenario.lanelet_network

    obj = RoadGraph(lanelet_network=net, planning_problem_set=planning_problem_set)
    occupancy_nodes, _ = obj.get_possible_resources(start_node=3512, end_node=3450)
    obj.plot_graph(filename='graph_only', start_node=3512, end_node=3450)

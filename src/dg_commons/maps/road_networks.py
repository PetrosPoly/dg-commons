from networkx import DiGraph, draw, bfs_edges, descendants, descendants_at_distance, draw_networkx_edges, \
    draw_networkx_nodes, draw_networkx_labels, all_simple_paths
import networkx as nx
from typing import List, Optional, Tuple, Mapping
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork
from commonroad.scenario.scenario import Scenario
from commonroad.common.file_reader import CommonRoadFileReader
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection, LineCollection
from commonroad.visualization.mp_renderer import MPRenderer


class RoadGraph:
    """
    Class to represent Commonroad lane networks as directed graphs.
    Forbidden lanelets (for cars) are not added to the network.
    Forbidden lanelets can also be manually.
    """
    #fixme: may become an issue for other scenarios, condition may need a ".value" to be added. Wait and see.
    forbidden_lanelet_types = ["busLane", "bicycleLane", "sidewalk"]
    uncrossable_line_markings = ["solid", "broad_solid"]

    def __init__(self, lanelet_network: LaneletNetwork, prohibited_lanelets: Optional[List[int]] = None):
        self.road_graph: DiGraph = DiGraph()
        self.lanelet_network = lanelet_network
        if prohibited_lanelets is None:
            prohibited_lanelets = list()
            for lanelet in self.lanelet_network.lanelets:
                if lanelet.lanelet_type in self.forbidden_lanelet_types:
                    prohibited_lanelets.append(lanelet.lanelet_id)
        self.prohibited_lanelets = prohibited_lanelets

    def create_topology_from_lanelet_network(self, weighted: bool = False) -> None:
        """
        Create a digraph from a Commonroad lanelet network.
        If lanelets are in the prohibited_lanelets network, they will not
        be added to the digraph.

        :param weighted: if False, all weights will be set to 1. If True,
        length of lanelets will be considered.
        """
        nodes = list()
        edges = list()

        for lanelet in self.lanelet_network.lanelets:
            # skip prohibited lanelet
            if lanelet.lanelet_id in self.prohibited_lanelets:
                continue

            # add permissible lanelet to graph
            nodes.append(lanelet.lanelet_id)

            # add edge for all succeeding lanelets
            for id_successor in lanelet.successor:
                # skip if prohibited
                if id_successor in self.prohibited_lanelets:
                    continue
                if weighted:
                    weight = self.get_weight_from_lanelets(id_lanelet_1=lanelet.lanelet_id,
                                                           id_lanelet_2=id_successor, successor=True)
                    edges.append((lanelet.lanelet_id, id_successor, {'weight': weight}))
                else:
                    edges.append((lanelet.lanelet_id, id_successor, {'weight': 1}))

            # add edge for adjacent right lanelet (if existing)
            if lanelet.adj_right_same_direction and lanelet.adj_right is not None:

                # skip if prohibited
                if lanelet.adj_right in self.prohibited_lanelets\
                        or lanelet.line_marking_right_vertices.value \
                        in self.uncrossable_line_markings:
                    continue

                if weighted:
                    weight = self.get_weight_from_lanelets(id_lanelet_1=lanelet.lanelet_id,
                                                           id_lanelet_2=lanelet.adj_right,
                                                           successor=False)
                    edges.append((lanelet.lanelet_id, lanelet.adj_right, {'weight': weight}))
                else:
                    edges.append((lanelet.lanelet_id, lanelet.adj_right, {'weight': 1}))

            # add edge for adjacent left lanelets (if existing)
            if lanelet.adj_left_same_direction and lanelet.adj_left is not None:

                # skip if prohibited
                if lanelet.adj_left in self.prohibited_lanelets \
                        or lanelet.line_marking_left_vertices.value \
                        in self.uncrossable_line_markings:
                    continue

                if weighted:
                    weight = self.get_weight_from_lanelets(id_lanelet_1=lanelet.lanelet_id,
                                                           id_lanelet_2=lanelet.adj_left,
                                                           successor=False)
                    edges.append((lanelet.lanelet_id, lanelet.adj_left, {'weight': weight}))
                else:
                    edges.append((lanelet.lanelet_id, lanelet.adj_left, {'weight': 1}))

        self.road_graph.add_nodes_from(nodes)
        self.road_graph.add_edges_from(edges)

    # naive implementation of weight generation. Make more accurate when needed
    def get_weight_from_lanelets(self, id_lanelet_1: int, id_lanelet_2: int, successor: bool) -> float:
        """
        Adapted from Commonroad Route Planner.
        Calculate weights for edges on graph.
        For successor: calculate average lenght of the involved lanes.
        For right/left adjacent lanes: calculate average road width.
        Highly heuristic and may need some refinement in the future
        """
        if successor:
            length_1 = self.lanelet_network.find_lanelet_by_id(id_lanelet_1).distance[-1]
            length_2 = self.lanelet_network.find_lanelet_by_id(id_lanelet_2).distance[-1]
            return (length_1 + length_2) / 2.0
        # rough approximation by only calculating width on first point of polyline
        else:
            width_1 = np.linalg.norm(self.lanelet_network.find_lanelet_by_id(id_lanelet_1).left_vertices[0]
                                     - self.lanelet_network.find_lanelet_by_id(id_lanelet_1).right_vertices[0])
            width_2 = np.linalg.norm(self.lanelet_network.find_lanelet_by_id(id_lanelet_2).left_vertices[0]
                                     - self.lanelet_network.find_lanelet_by_id(id_lanelet_2).right_vertices[0])
            return (width_1 + width_2) / 2.0

    def get_lanelet_center_coordinates(self, lanelet_ids: List[int]) -> List[np.ndarray]:
        centers = []
        for lanelet_id in lanelet_ids:
            center_line = self.lanelet_network.find_lanelet_by_id(lanelet_id).center_vertices

            # center line has an even amount of points
            if len(center_line) % 2 == 0:
                centers.append(
                    (center_line[int(-1+len(center_line) / 2)]
                     + center_line[int(len(center_line) / 2)]) / 2)
            # center line has an odd amount of points
            elif len(center_line) % 2 == 1:
                centers.append(
                    (center_line[int(len(center_line) / 2)]))

            if lanelet_id == 3416:
                print("ID 3416: " + str(centers[-1]))
            if lanelet_id == 3426:
                print("ID 3426: " + str(centers[-1]))
            if lanelet_id == 3428:
                print("ID 3428: " + str(centers[-1]))
        return centers

    def plot_graph_on_road_static(self, renderer: MPRenderer, filename: str, start_node: int, end_node: int) -> None:
        plt.figure(1)
        # scenario.draw(renderer) #also draws cars
        scenario.lanelet_network.draw(renderer)
        # planning_problem_set.draw(rnd)
        renderer.render()
        plt.grid(True)
        plt.xlabel('X Axis')
        plt.ylabel('Y Axis')
        visited_nodes, special_edges = self.get_possible_occupancy(start_node=start_node, end_node=end_node)
        goal_nodes = self.get_occupancy_children(occupancy_nodes=visited_nodes)
        _, _, _ = self.get_collections_networkx(special_edges=special_edges, visited_nodes=visited_nodes,
                                                goal_nodes=goal_nodes, start_node=start_node, end_node=end_node)
        plt.savefig(filename)
        plt.close()

    def get_collections_networkx(self, special_edges: Optional[List[Tuple]] = None,
                                 visited_nodes: List[int] = None, goal_nodes: List[int] = None,
                                 start_node: int = None, end_node: int = None)\
                                 -> Tuple[PathCollection, LineCollection, Mapping[int, plt.Text]]:
        """
        Get collections for plotting a graph on top of a scenario

        :param special_edges: Edges to color differently.
        :param visited_nodes: Nodes to color differently.
        :param goal_nodes: possible other agents goals.
        :param start_node: Departure node of ego.
        :param end_node: Arrival node of ego.
        """
        nodes = list(self.road_graph.nodes)
        cents = obj.get_lanelet_center_coordinates(nodes)
        centers = dict(zip(nodes, cents))

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
        nodes_plot = draw_networkx_nodes(G=self.road_graph, pos=centers, node_size=500, node_color=node_colors)
        nodes_plot.set_zorder(50)
        edges_plot = draw_networkx_edges(G=self.road_graph, pos=centers, edge_color=edge_colors)
        labels_plot = draw_networkx_labels(G=self.road_graph, pos=centers)
        for edge in range(len(edges_plot)):
            edges_plot[edge] = edges_plot[edge].set_zorder(49)
        for label in labels_plot.keys():
            labels_plot[label].set_zorder(51)
        return nodes_plot, edges_plot, labels_plot

    def get_possible_occupancy(self, start_node: int, end_node: int) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        Compute all nodes where an agent could be when transiting from start_node to end_node
        Return both occupied nodes and occupied edges

        :param start_node: departure of agent
        :param end_node: destination of agent
        """
        paths = all_simple_paths(self.road_graph, source=start_node, target=end_node)

        occupancy_nodes = []
        occupancy_edges = []
        for path in paths:
            # add new nodes only if they are not yet in occupancy_nodes
            helper_nodes = set(path)-set(occupancy_nodes)
            occupancy_nodes = occupancy_nodes + list(helper_nodes)

            # add new edges only if they are not yet in occupancy_edges
            path_edges = nx.utils.pairwise(path)  # transforms [a,b,c,...] in [(a,b),(b,c),...]
            helper_edges = set(path_edges) - set(occupancy_edges)
            occupancy_edges = occupancy_edges + list(helper_edges)

        return occupancy_nodes, occupancy_edges

    # question: may be useful to also return the edge directed to that node if there are multiple possible edges
    def get_occupancy_children(self, occupancy_nodes) -> List[int]:
        """
        Compute the children of the nodes in the "occupancy zone" as computed by "get_possible_occupancy"

        :param occupancy_nodes: part og digraph where the ego could be on his journey to the goal
        """
        children = []
        for node in occupancy_nodes:
            candidates = list(self.road_graph.successors(node))
            for cand in candidates:
                if cand in occupancy_nodes:
                    continue
                else:
                    children.append(cand)

        return children

    # Following functions were written but are not currently used.
    # For now, keep for reference or reusage.

    # naive implementation: depth_limit is just a user-defined integer. May not be the right choice.
    # todo: make it also return weights
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








scenario_path1 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Zip-1_66_T-1.xml"  # remove e.g. 24
scenario_path2 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/NGSIM/Peachtree/USA_Peach-1_1_T-1.xml"  # remove e.g. 52826
scenario_path3 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/DEU_Muc-1_2_T-1.xml"  # remove e.g. 24

scenario, planning_problem_set = CommonRoadFileReader(scenario_path3).open(lanelet_assignment=True)
net = scenario.lanelet_network


obj = RoadGraph(lanelet_network=net)
obj.create_topology_from_lanelet_network()
obj.plot_graph_on_road_static(renderer=MPRenderer(figsize=(100, 100)), filename='demo', start_node=3512, end_node=3450)





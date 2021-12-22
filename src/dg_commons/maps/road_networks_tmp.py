from __future__ import annotations

import time

from networkx import DiGraph, draw_networkx_edges, draw_networkx_nodes, draw_networkx_labels, all_simple_paths
import networkx as nx
from typing import List, Optional, Tuple, Mapping, Set, Dict, Any, Callable, Union
import numpy as np
from commonroad.scenario.lanelet import LaneletNetwork, Lanelet
from commonroad.planning.planning_problem import PlanningProblem
from commonroad.common.file_reader import CommonRoadFileReader
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection, LineCollection
from dg_commons.sim import SimObservations, SimTime
from dg_commons import PlayerName
from copy import copy, deepcopy
from shapely.geometry import Polygon, LinearRing, Point, LineString, MultiLineString
from shapely.strtree import STRtree
from shapely.ops import nearest_points, split
import math
from commonroad.visualization.mp_renderer import MPRenderer
from utils import *
from prediction_structures import Prediction
from lanes import DgLanelet
import geometry as g

# factor to make resource ids unique
resource_id_factor = 100


# fixme: temporary workaround to plot with digraph as input. This can be used to avoid deepcopying entire
# dynamic graph but only Digraph at each simulation timestep.
def get_collections_networkx_temp(resource_graph: DiGraph(), ax: Axes = None) -> Tuple[PathCollection, LineCollection,
                                                                                       Mapping[int, plt.Text]]:
    """
    Get collections for plotting a graph on top of a scenario

    :param ax: Axes on which to draw the Artists
    :param resource_graph: graph to draw
    """
    nodes = resource_graph.nodes
    edges = resource_graph.edges
    cents = []
    for node in nodes.data():
        cents.append(node[-1]['polygon'].center)

    centers = dict(zip(nodes.keys(), cents))

    # set default edge and node colors
    edge_colors = ['k'] * len(resource_graph.edges)
    node_colors = ['#1f78b4'] * len(resource_graph.nodes)

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
    nodes_plot = draw_networkx_nodes(G=resource_graph, ax=ax, pos=centers, node_size=200, node_color=node_colors)
    nodes_plot.set_zorder(50)

    edges_plot = draw_networkx_edges(G=resource_graph, ax=ax, pos=centers, edge_color=edge_colors)
    for edge in range(len(edges_plot)):
        edges_plot[edge].set_zorder(49)

    labels_plot = draw_networkx_labels(G=resource_graph, ax=ax, pos=centers)
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


def split_lanelet_into_polygons_new(lanelet: Lanelet, max_length: float) -> List[Polygon]:
    lanelet_length = lanelet.distance[-1]
    n_polygons = lanelet_length // max_length + 1
    left_vertices = lanelet.left_vertices
    right_vertices = lanelet.right_vertices
    polygon_length = lanelet_length / n_polygons  # check this is of type float

    counter = 0
    current_polygon_base_id = lanelet.lanelet_id * resource_id_factor

    polygons = []

    current_polygon_vertices_l = []
    current_polygon_vertices_r = []
    previous_index_before = 0

    #
    previous_left_interp = Point(left_vertices[0])
    previous_right_interp = Point(right_vertices[0])

    for n in range(int(n_polygons)):

        # after
        current_length = (n + 1) * polygon_length

        index_after = np.where(lanelet.distance >= current_length)[0]

        if len(index_after) == 0:
            print("This is a strange case")
            print(current_length)
            print(lanelet.distance)

        index_after = index_after[0]

        index_before = index_after - 1

        if n != 0:
            current_polygon_vertices_l.append(previous_left_interp)
            current_polygon_vertices_r.append(previous_right_interp)

        # consider intermediate points
        if previous_index_before != index_before:
            for i in range(index_before - previous_index_before + 1):
                index_intermediate = previous_index_before + i
                current_polygon_vertices_l.append(left_vertices[index_intermediate])
                current_polygon_vertices_r.append(right_vertices[index_intermediate])

        # interpolate on left and right boundaries
        fraction = (current_length - lanelet.distance[index_before]) \
                   / (lanelet.distance[index_after] - lanelet.distance[index_before])
        left_points = [lanelet.left_vertices[index_before], lanelet.left_vertices[index_after]]
        right_points = [lanelet.right_vertices[index_before], lanelet.right_vertices[index_after]]
        left_interp_point = Point(interpolate2d(fraction=fraction, points=left_points))
        right_interp_point = Point(interpolate2d(fraction=fraction, points=right_points))

        previous_left_interp = left_interp_point
        previous_right_interp = right_interp_point

        current_polygon_vertices_l.append(left_interp_point)
        current_polygon_vertices_r.append(right_interp_point)

        #current_polygon_vertices_l.append(left_vertices[index_after])
        #current_polygon_vertices_r.append(right_vertices[index_after])

        # create desired structure for Shapely LinearRing
        current_polygon_vertices_l.insert(0, current_polygon_vertices_r[0])
        current_polygon_vertices_r.append(current_polygon_vertices_l[-1])
        del current_polygon_vertices_l[-1]
        del current_polygon_vertices_r[0]
        current_polygon_vertices_r.reverse()
        current_polygon = Polygon(LinearRing(current_polygon_vertices_l.extend(current_polygon_vertices_r)))

        # clear lists for next polygon
        current_polygon_vertices_l.clear()
        current_polygon_vertices_r.clear()

        # append polygon with id
        polygons.append((current_polygon, current_polygon_base_id + counter))

        previous_index_before = index_before
        counter = counter + 1

    return polygons


def split_lanelet_into_polygons(lanelet: Lanelet, max_length: float) -> List[Polygon]:
    dglanelet = DgLanelet.from_commonroad_lanelet(lanelet=lanelet)
    lanelet_shapely = lanelet.polygon.shapely_object
    max_distance = lanelet.distance[-1]
    n_polygons = np.ceil(max_distance / max_length)
    # make finer boundaries of lanelet
    lane_profile = dglanelet.lane_profile(points_per_segment=10)
    lane_profile = [Point(point) for point in lane_profile]
    print(len(lane_profile) / 2)
    right_boundary = lane_profile[:int(len(lane_profile) / 2)]
    right_boundary = LineString(right_boundary)
    left_boundary = lane_profile[int((len(lane_profile) / 2)):]
    # invert operation done in lane_profile()
    left_boundary.reverse()
    left_boundary = LineString(left_boundary)

    dbeta = 1.0 / n_polygons
    polygons = []
    division_lines = []
    for pol in range(int(n_polygons)):
        # skip last
        if pol == int(n_polygons) - 1:
            continue

        beta_end = dbeta * (pol + 1)
        division_point = dglanelet.center_point(beta=beta_end)
        division_point = g.translation_from_SE2(division_point)
        division_point = Point(division_point)
        # check type
        # find the nearest points in left and right boundary
        nearest_left = nearest_points(division_point, left_boundary)[1]
        nearest_right = nearest_points(division_point, right_boundary)[1]
        division_lines.append((np.asarray(nearest_left), np.asarray(nearest_right)))

    remaining_polygon = lanelet_shapely
    for division in division_lines:
        geoms = split(remaining_polygon, LineString(division))
        current_polygon = geoms[0]
        remaining_polygon = geoms[1]
        polygons.append(current_polygon)

    return polygons


# fixme: issue when polygons are too small. These should be merged. How?
def split_lanelet_into_polygons_old(lanelet: Lanelet, max_length: float) -> List[Polygon]:
    """
    Split a lanelet in smaller polygons with length along centerline smaller or equal to max_length.
    Function respects lanelet boundaries exactly but does not guarantee polygons with uniform length.

    :param lanelet: lanelet to divide
    :param max_length: maximum length allowed for a polygon, along centerline.
    """
    assert max_length > 0.0, "You need a maximum cell length that is greater than 0.0."
    polygons = []
    centerline_distance = lanelet.distance
    left_vertices = lanelet.left_vertices
    right_vertices = lanelet.right_vertices
    counter = 0

    cumulative_vertices = []
    cumulative_left_vertices = []
    cumulative_right_vertices = []

    cumulative_distance = 0

    for segment_id, _ in enumerate(centerline_distance):

        # distance between two successive centerline points
        ddistance = centerline_distance[segment_id] - centerline_distance[segment_id - 1]
        current_polygon_base_id = lanelet.lanelet_id * resource_id_factor

        if segment_id == 0:
            continue

        # new: tries making polygons in order of magnitude of max_length
        elif ddistance <= max_length:
            if cumulative_distance <= max_length:
                cumulative_distance += ddistance
                cumulative_left_vertices.append(left_vertices[segment_id - 1])
                cumulative_right_vertices.append(right_vertices[segment_id - 1])

            if segment_id == segment_id == len(centerline_distance) - 1:
                pass

            if cumulative_distance > max_length \
                    or (segment_id == len(centerline_distance) - 1 and cumulative_distance <= max_length):
                if segment_id == len(centerline_distance) - 1:
                    cumulative_left_vertices.append(left_vertices[segment_id])
                    cumulative_right_vertices.append(right_vertices[segment_id])

                cumulative_vertices.extend(cumulative_left_vertices)
                cumulative_right_vertices.reverse()
                cumulative_vertices.extend(cumulative_right_vertices)
                current_polygon = Polygon(LinearRing(cumulative_vertices))
                polygons.append((current_polygon, current_polygon_base_id + counter))

                cumulative_left_vertices.clear()
                cumulative_right_vertices.clear()
                cumulative_vertices.clear()
                counter = counter + 1
                # reset cumulative distance
                cumulative_distance = 0.0

        # divide segment into smaller polygons
        elif ddistance > max_length:
            ncells = math.ceil((ddistance / max_length))
            # dcell = ddistance / ncells
            cell_frac = 1 / float(ncells)
            left_points = [left_vertices[segment_id - 1], left_vertices[segment_id]]
            right_points = [right_vertices[segment_id - 1], right_vertices[segment_id]]
            for cell in range(ncells):
                left_start = interpolate2d(fraction=cell * cell_frac, points=left_points)
                left_end = interpolate2d(fraction=(cell + 1) * cell_frac, points=left_points)
                right_start = interpolate2d(fraction=cell * cell_frac, points=right_points)
                right_end = interpolate2d(fraction=(cell + 1) * cell_frac, points=right_points)
                vertices = [left_start, right_start, right_end, left_end]
                current_polygon = Polygon(LinearRing(vertices))
                polygons.append((current_polygon, current_polygon_base_id + counter))
                counter = counter + 1

    return polygons


class ResourceNetwork:
    uncrossable_line_markings = ["solid", "broad_solid"]  # no vehicle can cross this

    def __init__(self, lanelet_network: LaneletNetwork, max_length: Optional[float] = None,
                 excluded_lanelets: Optional[List[int]] = None):
        """
        Create a digraph, here called Resource Network, from a Commonroad lanelet network.
        :param lanelet_network: Commonroad lenelet network
        :param max_length: maximum length of a cell. If None, each lanelet is a cell.
        :param excluded_lanelets: lanelets that should not be added to the graph
        """
        # get lanelet network and create digraph
        # store in STRTree as well (or only in STRTree)
        self.resource_graph: DiGraph = DiGraph()
        self.tree: STRtree
        if excluded_lanelets is None:
            excluded_lanelets = list()
        self.excluded_lanelets = excluded_lanelets
        self._create_rtree(lanelet_network=lanelet_network, max_length=max_length)
        # fixme: merge the graph creation functions
        self._init_graph(lanelet_network=lanelet_network, max_length=max_length)

    def set_default_attributes(self) -> None:
        # set default attributes for nodes
        for current_node in list(self.resource_graph.nodes):
            self.set_node_attribute(attribute='start', value=False, node=current_node)
            self.set_node_attribute(attribute='goal', value=False, node=current_node)
            self.set_node_attribute(attribute='ego_occupied_resource', value=False, node=current_node)
            self.set_node_attribute(attribute='occupied_by_agent', value=False, node=current_node)
            self.set_node_attribute(attribute='goal_of_interest', value=False, node=current_node)

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
        nodes = self.resource_graph.nodes
        edges = self.resource_graph.edges
        cents = []
        for node in nodes.data():
            cents.append(np.asarray(node[-1]['polygon'].centroid))

        centers = dict(zip(nodes.keys(), cents))

        # set default edge and node colors
        edge_colors = ['k'] * len(self.resource_graph.edges)
        node_colors = ['#1f78b4'] * len(self.resource_graph.nodes)

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
        nodes_plot = draw_networkx_nodes(G=self.resource_graph, ax=ax, pos=centers, node_size=200,
                                         node_color=node_colors)
        nodes_plot.set_zorder(50)

        edges_plot = draw_networkx_edges(G=self.resource_graph, ax=ax, pos=centers, edge_color=edge_colors)
        for edge in range(len(edges_plot)):
            edges_plot[edge].set_zorder(49)

        labels_plot = draw_networkx_labels(G=self.resource_graph, ax=ax, pos=centers)
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
        t1 = time.time()
        from sys import getsizeof
        print("calculating possible resources")
        paths = all_simple_paths(self.resource_graph, source=start_node, target=end_node)

        # works
        # paths_test = all_simple_paths(self.resource_graph, source=354013, target=354018)
        # print(list(paths_test))
        print("time needed for simple paths " + str(time.time() - t1))
        print(paths)
        print(id(paths))
        print(getsizeof(paths))
        print(list(paths))
        occupancy_nodes = set()
        occupancy_edges = set()
        for path in paths:
            # add new nodes only if they are not yet in occupancy_nodes
            occupancy_nodes |= set(path)
            # add new edges only if they are not yet in occupancy_edges
            path_edges = nx.utils.pairwise(path)  # transforms [a,b,c,...] in [(a,b),(b,c),...]
            occupancy_edges |= set(path_edges)
        print("done")
        print("time needed " + str(time.time() - t1))
        return occupancy_nodes, occupancy_edges

    def get_occupancy_children(self, occupied_resources: Set[int]) -> List[int]:
        """
        Compute the children of the nodes in the "occupancy zone" as computed by "get_possible_resources"

        :param occupied_resources: part of digraph where the ego could be on his journey to the goal
        """
        children = []
        for node in occupied_resources:
            candidates = list(self.resource_graph.successors(node))
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

        children = nx.traversal.bfs_tree(G=self.resource_graph, source=node_id).nodes
        is_upstream = (set(children) & nodes) != set()

        return is_upstream

    # tbd: reward function
    def reward_1(self, weight: float):
        return -weight

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

        paths = nx.all_shortest_paths(G=self.resource_graph, source=start_node, target=end_node,
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
            if self.resource_graph.get_edge_data(edge[0], edge[1]) is None:
                print("NONE FOUND")
            path_reward += reward(self.resource_graph.get_edge_data(edge[0], edge[1])['weight'])
        return path_reward

    # fixme: remove when StrTree method tested
    def get_lanelet_by_position_old(self, position: np.ndarray) -> int:
        """
        Compute lanelet that contains the queried position.
        :param position: query position
        """
        for lanelet_id, lanelet_polygon in list(self.resource_graph.nodes(data='polygon')):
            if lanelet_polygon.contains_point(position):
                return lanelet_id

        print("Position: " + str(position) + ". No lanelet found that contains the position you asked for.")

    # new function
    # fixme: current implementation can not handle 3D structures (e.g. underpass). STRTree has issues!
    def get_resource_by_position(self, position: Union[np.ndarray, Point]) -> int:
        """
        Compute resource (possibly multiple) that contains the queried position by exploiting StrTree.
        :param position: query position. CoM of agent.
        """
        if isinstance(position, np.ndarray):
            position = Point(position)
        resource = self.tree.nearest_item(position)
        # resources = self.tree.query_items(position)
        # resources_geom = self.tree.query(position)
        # test = self.tree.nearest_item(resources_geom[0])
        # candidate_resources = [o for o in self.tree.query(position) if o.intersects(position)]
        # candidate_items = []
        # for res in candidate_resources:
        #    candidate_items.append(self.tree.nearest_item(res))

        if not isinstance(resource, int):
            print("Position: " + str(position) + ". No resource found that contains the position you asked for.")

        return resource










    def _init_graph(self, lanelet_network: LaneletNetwork, max_length: Optional[float]):
        if max_length is None:
            self._init_road_graph(lanelet_network=lanelet_network)
        else:
            self._init_resource_graph(lanelet_network=lanelet_network, max_length=max_length)

    def _init_resource_graph(self, lanelet_network: LaneletNetwork, max_length: Optional[float]):
        """
        Construct road graph from road network. All lanelets are added, including the lanelet type.
        Lanelets that are in "excluded_lanelets" will be omitted.
        Each lanelet is divided into cells smaller than max_length.
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
                lanelet=lanelet_network.find_lanelet_by_id(lanelet.lanelet_id), max_length=max_length)

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
                weight = 1.0
                self.resource_graph.add_edge(polygon_idx, polygon_idx + 1, weight=weight)

            # add edge for all succeeding lanelets
            # specifically, connect last resource of a lanelet with first resource of succeeding lanelet
            for id_successor in lanelet.successor:
                # skip excluded lanelet (may be a successor of an allowed lanelet)
                if id_successor in self.excluded_lanelets:
                    continue
                weight = 1.0
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
                            weight = 1.0
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
                            weight = 1.0
                            self.resource_graph.add_edge(resource_id, left_resource_id, weight=weight)

        self.set_default_attributes()

        file_path = "max_size=1000.jpg"

        self.plot_graph(file_path=file_path)

        """plt.figure()
        nx.draw(self.resource_graph)
        plt.savefig(file_path)
        plt.close()"""

        return

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
            polygon = lanelet_network.find_lanelet_by_id(lanelet.lanelet_id).polygon.shapely_object
            # center = get_lanelet_center_coordinates(lanelet_network, lanelet.lanelet_id)

            self.resource_graph.add_node(lanelet.lanelet_id * resource_id_factor,
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
                self.resource_graph.add_edge(lanelet.lanelet_id * resource_id_factor,
                                             id_successor * resource_id_factor, weight=weight)

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
                self.resource_graph.add_edge(lanelet.lanelet_id * resource_id_factor,
                                             lanelet.adj_right * resource_id_factor, weight=weight)

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
                self.resource_graph.add_edge(lanelet.lanelet_id * resource_id_factor,
                                             lanelet.adj_left * resource_id_factor, weight=weight)

        self.set_default_attributes()

        return

    def _create_rtree(self, lanelet_network: LaneletNetwork, max_length: Optional[float]):
        resources = []

        for lanelet in lanelet_network.lanelets:
            # skip excluded lanelets
            if lanelet.lanelet_id in self.excluded_lanelets:
                continue
            if max_length is not None:
                new_resources = split_lanelet_into_polygons(
                    lanelet=lanelet_network.find_lanelet_by_id(lanelet.lanelet_id), max_length=max_length)
            else:
                new_resources = [(lanelet.polygon.shapely_object, lanelet.lanelet_id * resource_id_factor)]
            resources.extend(new_resources)

        resource_polygons = [resource[0] for resource in resources]
        resource_ids = [resource[1] for resource in resources]
        self.tree = STRtree(geoms=resource_polygons, items=resource_ids)
        return


class DynamicRoadGraph(ResourceNetwork):
    """
        Class to represent dynamic digraphs.
        Extend RoadGraph by adding positions of observed agents and start & end position of the ego
    """

    def __init__(self, lanelet_network: LaneletNetwork,
                 excluded_lanelets: Optional[List[int]] = None, max_length: Optional[float] = None):
        """
        """
        super().__init__(lanelet_network=lanelet_network, excluded_lanelets=excluded_lanelets, max_length=max_length)
        self.ego_start: Optional[int] = None
        self.ego_goal: Optional[int] = None
        self.ego_problem_updated: bool = False
        self.prediction_instantiated: bool = False
        self.locations_instantiated: bool = False

        # initialize start and goal with initial planning problem

        self.locations: Dict[PlayerName, List[Tuple[SimTime, int]]] = {k: [] for k in
                                                                       ['P1', 'Ego']}  # fixme: make general
        # self.goal_dict = PredDict.from_dict()
        # fixme: player names should already be somewhere in the dynamic graph!
        #  tbd: better way than overwriting later when we have start/goal/other info?
        self.predictions: Optional[Prediction] = None
        self.graph_storage = []

    def update_predictions(self, sim_obs: SimObservations):
        self.update_locations(sim_obs)
        self.update_dynamic_graph()
        # fixme: temporary
        if not self.prediction_instantiated:
            self.prediction_instantiated = True
        # fixme: ugly workaround to update reachability dict when goals are not reachable anymore

        self.update_reachability(players=list(sim_obs.players.keys()))
        self.compute_rewards_and_paths()
        self.compute_goal_probabilities()

        return

    """def initialize_locations(self, initial_obs: SimObservations):
        # initialize locations for all agents
        for player, player_obs in initial_obs.players.items():
            player_pos = np.array([player_obs.state.x, player_obs.state.y])
            #resource_id = self.get_lanelet_by_position_restricted(player=player, position=player_pos)
            resource_id = self.get_resource_by_position(position=player_pos)
            self.locations[player].append((initial_obs.time, resource_id))
        return"""

    def initialize_prediction(self, initial_obs: SimObservations):
        # instantiate prediction object with correct goal structure
        self.update_locations(sim_obs=initial_obs)
        self.update_dynamic_graph()
        self.instantiate_prediction_object(players=list(initial_obs.players.keys()))
        return

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

        # fixme: at first iteration use get_resource_by_position.
        # get_resource_by_position does not work because it needs an initial id. Generalize this.

        start = self.get_resource_by_position(position=problem.initial_state.position)
        self.ego_start = start
        goal = problem.goal.lanelets_of_goal_position
        if goal is not None:
            self.ego_goal = goal[0][0] * resource_id_factor  # fixme: keep only first
        else:
            next_node = list(self.resource_graph.successors(start))
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
        Version get_resource_by_position in resource_graph with restricted search.
        Position is used to find lanelet id only by querying on current lanelet
        polygon and the subsequent lanelet polygons.
        :param position: query position
        """
        previous_id = self.locations[player][-1][1]
        # check if position is in previous polygon
        if self.resource_graph.nodes[previous_id]['polygon'].contains_point(position):
            return previous_id

        # check if position is in subsequent polygons
        next_ids = self.resource_graph.neighbors(previous_id)
        for next_id in list(next_ids):
            if self.resource_graph.nodes[next_id]['polygon'].contains_point(position):
                return next_id

        print("Position: " + str(position) + ". No lanelet found that contains"
                                             " the position you asked for by using restricted algorithm.")

    # fixme: would be more efficient to only pass DiGraph and not entire graph
    def keep_track(self):
        self.graph_storage.append(deepcopy(self.resource_graph))

    def instantiate_prediction_object(self, players: List[PlayerName]):
        self.predictions = Prediction(self.get_goals_dict(players=players))
        return

    def update_locations(self, sim_obs: SimObservations):
        t = sim_obs.time
        for player, player_obs in sim_obs.players.items():
            player_pos = np.array([player_obs.state.x, player_obs.state.y])
            resource_id = self.get_resource_by_position(position=player_pos)
            # only consider locations reachable from previous location.
            # measure to handle 3D situations where streets are on several layers
            """for candidate_resource in resource_id:
                pass"""

            # check if which resource is reachable
            # if more than one, return warning but pick one ( we probably are on limit of both)

            self.locations[player].append((t, resource_id))
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
        for node_id, node_data in self.resource_graph.nodes(data=True):
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
            for node_id, node_data in self.resource_graph.nodes(data=True):
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


if __name__ == '__main__':
    # scenario_path1 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Zip-1_66_T-1.xml"  # remove e.g. 24
    # scenario_path2 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/NGSIM/Peachtree/USA_Peach-1_1_T-1.xml"  # remove e.g. 52826
    scenario_path3 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/DEU_Muc-1_2_T-1.xml"  # remove e.g. 24
    scenario_path4 = "/home/leon/Documents/repos/driving-games/scenarios/commonroad-scenarios/scenarios/hand-crafted/ZAM_Intersection-1_1_T-1_nando.xml"

    scenario, planning_problem_set = CommonRoadFileReader(scenario_path3).open(lanelet_assignment=True)
    net = scenario.lanelet_network

    obj = ResourceNetwork(lanelet_network=net, max_length=10.0)

    # Plot graph and scenario

    fig, axs = plt.subplots(2, figsize=(50, 50))

    nodes = obj.resource_graph.nodes
    cents = []
    for node in nodes.data():
        # cent_point = node[-1]['polygon']
        cent_point = node[-1]['polygon'].centroid
        cents.append([cent_point.x, cent_point.y])

    cents = dict(zip(nodes.keys(), cents))

    # centers = dict(zip(nodes.keys(), cents))
    nodes_plot = draw_networkx_nodes(G=obj.resource_graph, ax=axs[0], pos=cents, node_size=50)
    edges_plot = draw_networkx_edges(G=obj.resource_graph, ax=axs[0], pos=cents)
    # plt.savefig("graph_debug.png")
    # plt.close()

    # plt.subplots()
    rnd = MPRenderer(ax=axs[1], )
    scenario.draw(rnd)
    # planning_problem_set.draw(rnd)
    rnd.render()
    plt.savefig("scenario_testing_10122021.png")
    plt.close()

    test_lanelet = 3318
    test_lanelet_adj_right = 3316
    test_lanelet_adj_left = 3320
    new_polygons = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet), max_length=10.0)
    new_polygons_right = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet_adj_right), max_length=10.0)
    new_polygons_left = split_lanelet_into_polygons(net.find_lanelet_by_id(test_lanelet_adj_left), max_length=10.0)

    old_polygon = net.find_lanelet_by_id(test_lanelet).polygon.shapely_object
    old_polygon_right = net.find_lanelet_by_id(test_lanelet_adj_right).polygon.shapely_object
    old_polygon_left = net.find_lanelet_by_id(test_lanelet_adj_left).polygon.shapely_object

    plt.plot()
    x, y = old_polygon.exterior.xy
    plt.plot(x[:], y[:], 'k')
    x, y = old_polygon_right.exterior.xy
    plt.plot(x[:], y[:], 'k')
    x, y = old_polygon_left.exterior.xy
    plt.plot(x[:], y[:], 'k')
    for i, polygon in enumerate(new_polygons_left):
        x, y = polygon[0].exterior.xy
        plt.plot(x[:], y[:], '--')
    for i, polygon in enumerate(new_polygons):
        x, y = polygon[0].exterior.xy
        plt.plot(x[:], y[:], '--')
    for i, polygon in enumerate(new_polygons_right):
        x, y = polygon[0].exterior.xy
        plt.plot(x[:], y[:], '--')

    plt.savefig("debugging_10-12-2021.png")
    plt.close()

    """

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

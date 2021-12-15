import math
from typing import Callable, List
from dg_commons_dev.planning.rrt_utils.dubins_path_planning import dubins_path_planning, mod2pi
from dg_commons_dev.planning.rrt_utils.utils import Node


def dubin_distance_cost(node1: Node, node2: Node, *args) -> float:
    """
    Compute dubin distance between two nodes
    @param node1: starting node
    @param node2: node to be reached
    @return: Dubin distance
    """
    curvature = args[0]
    p_x, _, _, _, _, cost = dubins_path_planning(node2.x, node2.y, node2.yaw,
                                                 node1.x, node1.y, node1.yaw, curvature, 0.1)

    if len(p_x) <= 1:
        return float("inf")

    return cost


def distance_cost(node1: Node, node2: Node, *args) -> float:
    """
    Compute distance between two nodes
    @param node1: First node
    @param node2: Second node
    @return: Squared distance
    """

    dx = node2.x - node1.x
    dy = node2.y - node1.y
    dist = math.hypot(dx, dy)

    return dist


def distance_angle_cost(node1: Node, node2: Node, *args) -> float:
    """
    Compute sum of angle and absolute distance between two nodes
    @param node1: starting node
    @param node2: node to be reached
    @return: Squared distance
    """

    def ang_diff(ang1, ang2):
        ang1 = mod2pi(ang1)
        ang2 = mod2pi(ang2)

        dang = abs(ang1 - ang2)
        dang = dang if dang < math.pi else 2 * math.pi - dang
        return dang

    dx = node2.x - node1.x
    dy = node2.y - node1.y
    dist = math.hypot(dx, dy)
    ang = math.atan2(dy, dx)

    delta1 = ang_diff(ang, node1.yaw)
    delta2 = ang_diff(ang, node2.yaw)
    delta3 = ang_diff(node1.yaw, node2.yaw)
    delta_ang = delta1 + delta2
    factor = 1 / math.pi * 30

    return dist + factor * delta3


def naive(node: Node, node_list: List[Node], cost_fct: Callable[[Node, Node, ...], float], *args) -> int:
    """
    Naive nearest neighbor search with cost_fct as discriminant choice
    @param node: Node of interest
    @param node_list: List of existing nodes
    @param cost_fct: Cost function for discrimination
    @return: Index of node closest to node in node_list
    """

    cost_list = [cost_fct(node, other_node, *args) for other_node in node_list]
    min_ind = cost_list.index(min(cost_list))

    return min_ind

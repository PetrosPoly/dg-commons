from dg_commons_dev.planning.rrt_utils.utils import Node, calc_distance_and_angle
import math
from dg_commons_dev.planning.rrt_utils.dubins_path_planning import dubins_path_planning
import copy


def straight_to(from_node: Node, to_node: Node, extend_length: float = float("inf"), resolution: float = 0.1,
                curvature: float = 0) -> Node:
    """
    Cut distance to a maximum of extend_length
    @param from_node: start node
    @param to_node: target node
    @param extend_length: maximum distance
    @param resolution: resolution of path
    @param curvature: zero in this case
    @return: node along the way from from node to to_node but at most extend_length distant and with a path description
             with the passed resolution
    """

    new_node = Node(from_node.x, from_node.y)
    dist, theta = calc_distance_and_angle(new_node, to_node)

    new_node.path_x = [new_node.x]
    new_node.path_y = [new_node.y]

    if extend_length > dist:
        extend_length = dist

    n_expand = math.floor(extend_length / resolution)

    for _ in range(n_expand):
        new_node.x += resolution * math.cos(theta)
        new_node.y += resolution * math.sin(theta)
        new_node.path_x.append(new_node.x)
        new_node.path_y.append(new_node.y)

    d, _ = calc_distance_and_angle(new_node, to_node)
    if d <= resolution:
        new_node.path_x.append(to_node.x)
        new_node.path_y.append(to_node.y)
        new_node.x = to_node.x
        new_node.y = to_node.y

    new_node.parent = from_node

    return new_node


def dubin_curves(from_node: Node, to_node: Node, extend_length: float = float("inf"), resolution: float = 0.1,
                 curvature: float = 1):
    """
    Formulate path from from_node to to_node with dubin curves
    @param from_node: start node
    @param to_node: target node
    @param curvature: max curvature (1/r_min)
    @param extend_length: maximum distance
    @param resolution: resolution of the returned path
    @return: New node with path description sampled at step_size resolution
    """
    bound_length: bool = False
    if bound_length:
        d, theta = calc_distance_and_angle(from_node, to_node)

        if extend_length > d:
            extend_length = d

        to_node.x += extend_length * math.cos(theta)
        to_node.y += extend_length * math.sin(theta)

    p_x, p_y, p_yaw, mode, course_lengths, _ = dubins_path_planning(from_node.x, from_node.y, from_node.yaw, to_node.x,
                                                                    to_node.y, to_node.yaw, curvature, resolution)

    if len(p_x) <= 1:
        return None

    new_node = copy.deepcopy(from_node)
    new_node.x = p_x[-1]
    new_node.y = p_y[-1]
    new_node.yaw = p_yaw[-1]

    new_node.path_x = p_x
    new_node.path_y = p_y
    new_node.path_yaw = p_yaw
    new_node.parent = from_node

    return new_node

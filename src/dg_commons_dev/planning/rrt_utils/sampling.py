from dg_commons_dev.planning.rrt_utils.utils import Node
import math
from abc import ABC, abstractmethod
import random
from typing import Tuple


class BaseBoundaries(ABC):
    """
    Base class for 2D bounded sampling
    """

    @abstractmethod
    def random_sampling(self) -> Tuple[float, float]:
        """
        Sample uniformly at random inside the boundaries, returns (x, y)
        """
        pass


class RectangularBoundaries(BaseBoundaries):
    """
    Class for 2D bounded sampling inside a rectangle
    """

    def __init__(self, boundaries: Tuple[float, float, float, float]):
        self.x_min, self.x_max = boundaries[0], boundaries[1]
        self.y_min, self.y_max = boundaries[2], boundaries[3]

    def x_bounds(self) -> Tuple[float, float]:
        """
        @return: tuple with x boundaries: (min, max)
        """
        return self.x_min, self.x_max

    def y_bounds(self) -> Tuple[float, float]:
        """
        @return: tuple with y boundaries: (min, max)
        """
        return self.y_min, self.y_max

    def random_sampling(self) -> Tuple[float, float]:
        """
        Sample uniformly at random inside the rectangle
        @return: x and y position
        """
        x: float = random.uniform(self.x_min, self.x_max)
        y: float = random.uniform(self.y_min, self.y_max)
        return x, y


def uniform_sampling(boundaries: BaseBoundaries,
                     goal_node: Node, goal_sample_rate: float) -> Node:
    """
    Sample a random node uniformly from available space.
    @param boundaries: limits the area, from which the sample is taken
    @param goal_node: goal pose
    @param goal_sample_rate: how often, on average, should the goal pose be sampled in %
    @return: A node with the sampled position (and sampled orientation if requested).
    """

    if random.randint(0, 100) > goal_sample_rate:
        pose: Tuple[float, float] = boundaries.random_sampling()

        if goal_node.is_yaw_considered:
            angle = random.uniform(-math.pi/2, math.pi/2)
            pose += (angle, )
        rnd = Node(*pose)
    else:
        rnd = Node(*goal_node.pose())
    return rnd

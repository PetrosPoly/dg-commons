from geometry import T2value
import random
import numpy as np
import math


class PCurve:
    """
    Model FP/FN/Accuracy as

    P(Event at distance d) = 2 * arctan(distance * parameter) / pi
    """

    def __init__(self, param: float):
        self.param = param

    def evaluate_distribution(self, rel_pos: T2value) -> float:
        """
        Evaluate distribution
        @param rel_pos: Relative position
        @return: Probability of event
        """
        pseudo_distance = np.linalg.norm(rel_pos) * self.param
        return math.atan(pseudo_distance) / math.pi * 2

    def sample(self, rel_pos: T2value) -> bool:
        """
        Sample from distribution
        @param rel_pos: Relative position
        @return: Whether event occurred or not
        """
        return random.uniform(0, 1) < self.evaluate_distribution(rel_pos)

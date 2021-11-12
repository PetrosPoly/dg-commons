from geometry import T2value
import random
import numpy as np
import math


class PCurve:
    """
    Model FP/FN/Accuracy as

    P(Event at distance d) = 2 * arctan(distance * parameter) / pi * convergence_param
    """

    def __init__(self, conv_speed: float, conv_val: float = 0.5):
        self.conv_speed = conv_speed
        self.conv_val = conv_val

    def evaluate_distribution(self, rel_pos: T2value) -> float:
        """
        Evaluate distribution
        @param rel_pos: Relative position
        @return: Probability of event
        """
        pseudo_distance = np.linalg.norm(rel_pos) * self.conv_speed
        return math.atan(pseudo_distance) / math.pi * 2 * self.conv_val

    def sample(self, rel_pos: T2value) -> bool:
        """
        Sample from distribution
        @param rel_pos: Relative position
        @return: Whether event occurred or not
        """
        return random.uniform(0, 1) < self.evaluate_distribution(rel_pos)

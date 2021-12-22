from __future__ import annotations

from typing import List, Optional, Dict, Any, Callable, Union
import numpy as np
from cytoolz import update_in

from dg_commons import PlayerName, valmap
from copy import copy


class GoalLikelihood(Dict[int, float]):
    """A distribution over possible resources (indexed by integers)"""

    def normalize(self):
        update_in(self, self.keys(), lambda x: x / sum(self.values()))

    def __add__(self, other: GoalLikelihood):
        assert isinstance(other, GoalLikelihood)

    def safe_add(self, other: GoalLikelihood):
        assert other.keys() == self.keys()
        self.__add__(other)


class PlayerGoalLikelihood([Dict[PlayerName, GoalLikelihood]]):
    """A player keeps track of the others' possible goals/resources"""


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

    # fixme: check out utils_toolz valmap
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
                print("Division by zero encountered. Approximate division done.")
                norm_factor = 999999999.0
            else:
                norm_factor = 1.0 / sum(player_dict.values())
            for goal in player_dict.items():
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

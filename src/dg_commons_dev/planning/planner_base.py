from abc import ABC, abstractmethod
from typing import List, Optional
from shapely.geometry.base import BaseGeometry
from dataclasses import dataclass
from dg_commons_dev.planning.rrt_utils.sampling import BaseBoundaries
from dg_commons_dev.planning.rrt_utils.utils import Node


class Planner(ABC):
    """ Planner interface """
    REF_PARAMS: dataclass

    @abstractmethod
    def planning(self, start: Node, goal: Node, obstacle_list: List[BaseGeometry], sampling_bounds: BaseBoundaries,
                 search_until_max_iter: bool = False) -> Optional[List[Node]]:
        """ Find path and returns it as a sequence of nodes """
        pass

    @abstractmethod
    def plot_results(self) -> None:
        """ Generate and save plots and animations """
        pass

    @abstractmethod
    def get_width(self) -> None:
        """ Returns security width of car """
        pass

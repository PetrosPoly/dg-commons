from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from dg_commons.maps.lanes import DgLanelet


Obs = TypeVar('Obs')
Rel = TypeVar('Rel')
S = TypeVar("S")
SParams = TypeVar("SParams")


class Situation(ABC, Generic[Obs, SParams]):
    """ A situation is a set of circumstances in which one finds oneself """

    @abstractmethod
    def update_observations(self, new_obs: Obs):
        """
        Update the information about the circumstances, choose whether the considered situation is occurring and
        computes some key parameters
        """
        pass

    @abstractmethod
    def is_true(self) -> bool:
        """ Returns whether the considered situation is occurring """
        pass

    @abstractmethod
    def infos(self) -> SParams:
        """ Returns important parameters describing the considered situation """
        pass


class Behavior(ABC, Generic[Obs, S]):
    """ Behavior manages the process of deciding which situation is occurring """

    @abstractmethod
    def update_observations(self, new_obs: Obs, path: DgLanelet):
        """ New observations come and a decision on the current situation is made """
        pass

    @abstractmethod
    def get_situation(self, at: float) -> S:
        """ The current situation is returned """
        pass

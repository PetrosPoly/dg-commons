from abc import ABC, abstractmethod
from typing import TypeVar, Generic
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams


Obs = TypeVar('Obs')
Rel = TypeVar('Rel')
S = TypeVar("S")
SParams = TypeVar("SParams")


@dataclass
class SituationParams(BaseParams):
    pass


class Situation(ABC, Generic[Obs, SParams]):

    @abstractmethod
    def update_observations(self, new_obs: Obs):
        pass

    @abstractmethod
    def is_true(self) -> bool:
        pass

    @abstractmethod
    def infos(self) -> SParams:
        pass


@dataclass
class BehaviorParams(BaseParams):
    pass


class Behavior(ABC, Generic[Obs, S]):

    @abstractmethod
    def update_observations(self, new_obs: Obs):
        pass

    @abstractmethod
    def get_situation(self, at: float) -> S:
        pass

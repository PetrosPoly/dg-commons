from typing import Generic, TypeVar
from abc import ABC, abstractmethod

Ref = TypeVar('Ref')
Obs = TypeVar('Obs')
U = TypeVar('U')


class Controller(ABC, Generic[Ref, Obs, U]):

    @abstractmethod
    def update_ref(self, new_ref: Ref):
        pass

    @abstractmethod
    def control(self, new_obs: Obs, t: float) -> U:
        pass

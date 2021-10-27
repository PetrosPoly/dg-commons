from typing import Union, Optional, Tuple, Generic
from dg_commons.maps.lanes import DgLanelet
from dg_commons_dev.maps.lanes import DgLaneletControl
from abc import ABC, abstractmethod
from dg_commons import X
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.controllers.interface import Controller, Obs, U, Ref


@dataclass
class Reference:
    speed_ref: float
    path: DgLanelet


@dataclass
class LateralControllerParam(BaseParams):
    pass


class LateralController(Controller[Reference, X, float]):
    def __init__(self):
        self.path: Optional[DgLanelet] = None
        self.control_path: Optional[DgLaneletControl] = None
        self.current_beta: Optional[float] = None

    @abstractmethod
    def _get_steering(self, at: float) -> float:
        pass

    @abstractmethod
    def _update_obs(self, new_obs: Obs):
        pass

    def _update_path(self, path: DgLanelet):
        assert isinstance(path, DgLanelet)
        self.path = path
        self.control_path = DgLaneletControl(path)

    def update_ref(self, new_ref: Reference):
        self._update_path(new_ref.path)

    def control(self, new_obs: Obs, t: float) -> float:
        self._update_obs(new_obs)
        return self._get_steering(t)


@dataclass
class LongitudinalControllerParam(BaseParams):
    pass


class LongitudinalController(Controller[Reference, X, float]):
    def __init__(self):
        self.speed_ref: float = 0

    @abstractmethod
    def _get_acceleration(self, at: float) -> float:
        pass

    @abstractmethod
    def _update_obs(self, new_obs: Obs):
        pass

    def _update_reference_speed(self, speed_ref: float):
        self.speed_ref = speed_ref

    def update_ref(self, new_ref: Reference):
        self._update_reference_speed(new_ref.speed_ref)

    def control(self, new_obs: Obs, t: float) -> float:
        self._update_obs(new_obs)
        return self._get_acceleration(t)


@dataclass
class LatAndLonControllerParam(BaseParams):
    pass


class LatAndLonController(LateralController, LongitudinalController,
                          Controller[Reference, X, Tuple[float, float]], ABC):

    def __init__(self):
        LateralController.__init__(self)
        LongitudinalController.__init__(self)

    def update_ref(self, new_ref: Reference):
        self._update_reference_speed(new_ref.speed_ref)
        self._update_path(new_ref.path)

    def control(self, new_obs: Obs, t: float) -> Tuple[float, float]:
        self._update_obs(new_obs)
        steer: float = self._get_steering(t)
        acc: float = self._get_acceleration(t)
        return steer, acc


@dataclass
class SteeringControllerParam(BaseParams):
    pass


class SteeringController(Controller[float, float, float]):
    def __init__(self):
        self.delta_ref: Optional[float] = None

    def update_ref(self, new_ref: float):
        self.delta_ref = new_ref

    def control(self, new_obs: float, t: float) -> U:
        assert self.delta_ref is not None
        return self._get_steering_vel(new_obs)

    @abstractmethod
    def _get_steering_vel(self, current_steering: float) -> float:
        pass

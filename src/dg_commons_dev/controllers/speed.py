from dataclasses import dataclass
from typing import Tuple, Optional, List, Union
from dg_commons.controllers.pid import PIDParam, PID
from dg_commons.sim.models import kmh2ms
from dg_commons_dev.controllers.controller_types import LongitudinalController
from dg_commons import X
from dg_commons_dev.utils import BaseParams

__all__ = ["SpeedControllerParam", "SpeedController"]


@dataclass
class SpeedControllerParam(BaseParams, PIDParam):
    """Default values are tuned roughly for a default car model"""
    kP: Union[float, List[float]] = 4
    kI: Union[float, List[float]] = 0.01
    kD: Union[float, List[float]] = 0.1
    antiwindup: Union[Tuple[float, float], List[Tuple[float, float]]] = (-2, 2)
    setpoint_minmax: Union[Tuple[float, float], List[Tuple[float, float]]] = (-kmh2ms(10), kmh2ms(150))
    output_minmax: Union[Tuple[float, float], List[Tuple[float, float]]] = (-8, 5)  # acc minmax

    def __post_init__(self):
        if isinstance(self.antiwindup, list):
            assert all([i[0] < i[1] for i in self.antiwindup])
        else:
            assert self.antiwindup[0] < self.antiwindup[1]
        if isinstance(self.setpoint_minmax, list):
            assert all([i[0] < i[1] for i in self.setpoint_minmax])
        else:
            assert self.setpoint_minmax[0] < self.setpoint_minmax[1]
        if isinstance(self.output_minmax, list):
            assert all([i[0] < i[1] for i in self.output_minmax])
        else:
            assert self.output_minmax[0] < self.output_minmax[1]
        super().__post_init__()


class SpeedController(PID, LongitudinalController):
    """Low-level controller for reference tracking of speed"""

    def __init__(self, params: Optional[PIDParam] = None):
        params = SpeedControllerParam() if params is None else params
        super(SpeedController, self).__init__(params)

    def _update_obs(self, new_obs: X):
        self.update_measurement(new_obs.vx)

    def _get_acceleration(self, at: float) -> float:
        self.update_reference(self.speed_ref)
        return self.get_control(at)

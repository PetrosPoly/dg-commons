from dg_commons import X
from dg_commons_dev.controllers.controller_types import BaseParams, Reference
from dataclasses import dataclass
from dg_commons.sim.models.vehicle_utils import steering_constraint, VehicleParameters
from dg_commons.sim.models.model_utils import acceleration_constraint
from typing import List, Union, Tuple, Optional
from dg_commons_dev.controllers.interface import Controller, Obs, U, Ref
from dg_commons_dev.behavior.emergency import EmergencySituation
from dg_commons_dev.behavior.yield_to import YieldSituation


@dataclass
class EmergencyControllerParams(BaseParams):
    vehicle_params: Union[List[VehicleParameters], VehicleParameters] = VehicleParameters.default_car()


class EmergencyController(Controller[Reference, EmergencySituation, Tuple[Optional[float], Optional[float]]]):

    def __init__(self):
        self.ref: Optional[Reference] = None

    def update_ref(self, new_ref: Reference):
        self.ref = new_ref

    def control(self, new_obs: EmergencySituation, t: float) -> Tuple[Optional[float], Optional[float]]:
        pass


@dataclass
class YieldControllerParams(BaseParams):
    vehicle_params: Union[List[VehicleParameters], VehicleParameters] = VehicleParameters.default_car()


class YieldController(Controller[Reference, YieldSituation, Tuple[Optional[float], Optional[float]]]):
    def __init__(self):
        self.ref: Optional[Reference] = None

    def update_ref(self, new_ref: Reference):
        self.ref = new_ref

    def control(self, new_obs: EmergencySituation, t: float) -> Tuple[Optional[float], Optional[float]]:
        pass

from typing import Tuple, get_args
import numpy as np
from dg_commons_dev.controllers.controller_types import *
from typing import Optional
from dg_commons_dev.behavior.behavior import SpeedBehavior
from dg_commons_dev.controllers.speed import SpeedController
from dg_commons.maps.lanes import DgLanelet
from dg_commons import PlayerName, X
from dg_commons.sim import SimObservations
from dg_commons.sim.agents.agent import Agent
from dg_commons.sim.models.vehicle import VehicleCommands
import time
from dg_commons_dev.controllers.steering_controllers import *
from dg_commons_dev.behavior.behavior_types import Behavior, BehaviorParams
from dg_commons_dev.behavior.emergency import EmergencySituation
from dg_commons_dev.controllers.steering_controllers import SCP
from dg_commons_dev.controllers.pure_pursuit_z import PurePursuit
from toolz.sandbox import unzip
from matplotlib.patches import Polygon


class LFAgent(Agent):
    """ This agent is a simple lane follower tracking the centerline of the given lane
    via a lateral controller. The reference speed is determined by the speed behavior and it
    is tracked by a speed controller.
    """

    def __init__(self,
                 lane: DgLanelet,
                 controller: Union[LateralController, LatAndLonController],
                 speed_behavior: Optional[Behavior] = None,
                 speed_controller: Optional[LongitudinalController] = None,
                 steering_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):

        single: bool = issubclass(type(controller), LatAndLonController) and \
            speed_controller is None
        decoupled: bool = not single and \
            issubclass(type(controller), LateralController) and \
            speed_controller is not None and \
            issubclass(type(speed_controller), LongitudinalController)

        assert single or decoupled

        # self.ref_lane = lane
        self.current_ref: Reference = Reference(0, lane)
        self.my_name: Optional[PlayerName] = None
        self.decoupled = decoupled

        self.controller: Union[LateralController, LatAndLonController] = controller
        self.speed_controller: Optional[LongitudinalController] = speed_controller
        self.speed_behavior: Behavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        self.steering_controller: SteeringController = SCP() if steering_controller is None else steering_controller

        self.my_name: Optional[PlayerName] = None
        self.return_extra: bool = return_extra
        self._emergency: bool = False
        self._my_obs: Optional[X] = None

        self.state_estimator = None
        self.state = None
        self.commands = None
        self.extra_plotting = None

        self.betas = []
        self.dt_commands = []

    def set_state_estimator(self, state_estimator):
        self.state_estimator = state_estimator

    def measurement_update(self, measurement):
        if self.state_estimator is not None:
            self.state_estimator.update_prediction(self.commands)
            self.state_estimator.update_measurement(measurement)
            self.state = self.state_estimator.state
        else:
            self.state = measurement

    @staticmethod
    def get_default_la(lane: DgLanelet):
        return LFAgent(lane, PurePursuit(), SpeedBehavior(), SpeedController(), SCP())

    def on_episode_init(self, my_name: PlayerName):
        self.my_name = my_name
        self.speed_behavior.my_name = my_name
        self.controller.update_ref(self.current_ref)
        # self.controller._update_path(self.current_ref.path)

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        t1 = time.time()
        my_obs = self.state
        t = float(sim_obs.time)

        self.speed_behavior.update_observations(sim_obs.players)
        speed_ref, situation, self.extra_plotting = self.speed_behavior.get_situation(t)
        self.current_ref.speed_ref = speed_ref if not self._emergency else 0

        if situation.is_emergency():
            self.emergency_subroutine(my_obs, t, situation.is_emergency())
            self._emergency = True
        else:
            self.normal_subroutine(my_obs, t)

        t2 = time.time()
        self.dt_commands.append(t2-t1)
        return self.commands

    def normal_subroutine(self, my_obs: X, t: float):
        if self.decoupled:
            acc, ddelta = self._get_decoupled_commands(my_obs, t)
        else:
            acc, ddelta = self._get_coupled_commands(my_obs, t)

        self.betas.append(self.controller.current_beta)
        self.commands = VehicleCommands(acc=acc, ddelta=ddelta)

    def simulation_ended(self):
        self.speed_behavior.simulation_ended()

    def emergency_subroutine(self, my_obs: X, t: float,
                             emergency: EmergencySituation) -> VehicleCommands:
        self.normal_subroutine(my_obs, t)
        acc = - self.speed_behavior.emergency.acc_limits[1]
        self.commands.acc = acc

    def _get_decoupled_commands(self, my_obs: X, t: float) -> Tuple[float, float]:

        self.speed_controller.update_ref(self.current_ref)
        acc = self.speed_controller.control(my_obs, t)

        self.controller.update_ref(self.current_ref)
        delta = self.controller.control(my_obs, t)

        self.steering_controller.update_ref(delta)
        ddelta = self.steering_controller.control(my_obs.delta, t)
        return acc, ddelta

    def _get_coupled_commands(self, my_obs: X, t: float) -> Tuple[float, float]:
        # compute commands
        self.controller.update_ref(self.current_ref)
        delta, acc = self.controller.control(my_obs, t)

        self.steering_controller.update_ref(delta)
        ddelta = self.steering_controller.control(my_obs.delta, t)
        return acc, ddelta

    def on_get_extra(self):
        if not self.return_extra:
            return None

        polygons, polygon_classes = unzip(self.extra_plotting)
        polygon_coll = []
        for poly in list(polygons):
            polygon_coll.append(Polygon(np.array(poly.exterior.coords.xy).T))

        colors = [cat.get_color() for cat in list(polygon_classes)]
        return list(zip(polygon_coll, colors))

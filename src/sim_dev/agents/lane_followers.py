from typing import Optional, Mapping, Any
import numpy as np
from duckietown_world.utils import SE2_apply_R2
from geometry import translation_from_SE2
from dg_commons.maps.lanes import DgLanelet
from dg_commons.planning.trajectory import Trajectory
from sim_dev.agents.lane_follower_z import LFAgent
from dg_commons.sim.models.vehicle import VehicleState
from dg_commons.sim import SimObservations, DrawableTrajectoryType
from dg_commons_dev.controllers.controller_types import *
from dg_commons_dev.controllers.speed import SpeedController
from dg_commons_dev.controllers.pure_pursuit_z import *
from dg_commons_dev.controllers.mpc.nmpc_full_kin_dis import *
from dg_commons_dev.controllers.mpc.nmpc_lateral_kin_cont import *
from dg_commons_dev.controllers.mpc.nmpc_lateral_kin_dis import *
from dg_commons_dev.controllers.mpc.nmpc_full_kin_cont import *
from dg_commons_dev.controllers.lqr import *
from dg_commons_dev.controllers.stanley_controller import *
from dg_commons_dev.behavior.behavior_types import Behavior, BehaviorParams
from dg_commons_dev.behavior.behavior import SpeedBehavior
from toolz.sandbox import unzip


class LFAgentPP(LFAgent):
    """ This agent is a simple lane follower tracking the centerline of the given lane
    via a pure pursuit controller. The reference in speed is determined by the speed behavior.
    """

    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[Behavior] = None,
                 controller: Optional[PurePursuit] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: Behavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        pure_pursuit: PurePursuit = PurePursuit() if controller is None else controller
        super().__init__(lane, pure_pursuit, speed_behavior, speed_controller, steer_controller, return_extra)

    def on_get_extra(self, ) -> Optional[Any]:
        if not self.return_extra:
            return None
        _, gpoint = self.controller.find_goal_point()
        pgoal = translation_from_SE2(gpoint)
        l = self.controller.params.length
        rear_axle = SE2_apply_R2(self.controller.pose, np.array([-l / 2, 0]))
        traj = Trajectory(
            timestamps=[0, 1],
            values=[VehicleState(x=rear_axle[0], y=rear_axle[1], theta=0, vx=0, delta=0),
                    VehicleState(x=pgoal[0], y=pgoal[1], theta=0, vx=1, delta=0),
                    ])

        traj_s = [traj]
        colors = ["gold"]

        items, item_colors = unzip(super().on_get_extra())
        items_list, item_colors_list = list(items), list(item_colors)
        items_list += traj_s
        item_colors_list += colors

        return list(zip(items_list, item_colors_list))


class LFAgentFullMPC(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[LatAndLonController] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller = None
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        full_mpc: LatAndLonController = NMPCFullKinCont() if controller is None else controller
        super().__init__(lane, full_mpc, speed_behavior, speed_controller, steer_controller, return_extra)

    def on_get_extra(self, ) -> Optional[DrawableTrajectoryType]:
        if not self.return_extra:
            return None
        dt = self.controller.params.t_step
        n_horizon = self.controller.params.n_horizon
        N = 20
        timestamps1 = []
        timestamps2 = []
        values1 = []
        values2 = []

        delta = self.controller.target_position[0] - self.controller.current_position[0]
        x_samples = np.linspace(self.controller.current_position[0],
                                self.controller.target_position[0] + 0.5*delta, num=N)

        for i in range(n_horizon):
            timestamps1.append(i*dt)
            values1.append(VehicleState(x=self.controller.prediction_x[i][0],
                                        y=self.controller.prediction_y[i][0], theta=0, vx=0, delta=0))

        '''for i in range(N):
            values2.append(VehicleState(x=x_samples[i], y=self.controller.current_f(x_samples[i]), theta=0, vx=0, delta=0))
            timestamps2.append(i / N * n_horizon * dt)'''


        traj1 = Trajectory(timestamps=timestamps1, values=values1)
        traj2 = Trajectory(timestamps=timestamps2, values=values2)

        traj_s = [traj1]
        colors = ["gold"]

        items, item_colors = unzip(super().on_get_extra())
        items_list, item_colors_list = list(items), list(item_colors)
        items_list += traj_s
        item_colors_list += colors

        return list(zip(items_list, item_colors_list))


class LFAgentLatMPC(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[LateralController] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        lat_mpc: LateralController = NMPCLatKinCont() if controller is None else controller
        super().__init__(lane, lat_mpc, speed_behavior, speed_controller, steer_controller, return_extra)

    def on_get_extra(self, ) -> Optional[DrawableTrajectoryType]:
        if not self.return_extra:
            return None
        dt = self.controller.params.t_step
        n_horizon = self.controller.params.n_horizon
        N = 20
        timestamps1 = []
        timestamps2 = []
        values1 = []
        values2 = []

        delta = self.controller.target_position[0] - self.controller.current_position[0]
        x_samples = np.linspace(self.controller.current_position[0],
                                self.controller.target_position[0] + 0.5 * delta, num=N)

        for i in range(n_horizon):
            timestamps1.append(i * dt)
            values1.append(VehicleState(x=self.controller.prediction_x[i][0],
                                        y=self.controller.prediction_y[i][0], theta=0, vx=0, delta=0))

        '''for i in range(N):
            values2.append(
                VehicleState(x=x_samples[i], y=self.controller.current_f(x_samples[i]), theta=0, vx=0, delta=0))
            timestamps2.append(i / N * n_horizon * dt)'''

        traj1 = Trajectory(timestamps=timestamps1, values=values1)
        traj2 = Trajectory(timestamps=timestamps2, values=values2)

        traj_s = [traj1]
        colors = ["gold"]

        items, item_colors = unzip(super().on_get_extra())
        items_list, item_colors_list = list(items), list(item_colors)
        items_list += traj_s
        item_colors_list += colors

        return list(zip(items_list, item_colors_list))


class LFAgentStanley(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[PurePursuit] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        stanley: Stanley = Stanley() if controller is None else controller
        super().__init__(lane, stanley, speed_behavior, speed_controller, steer_controller, return_extra)


class LFAgentLQR(LFAgent):
    def __init__(self,
                 lane: Optional[DgLanelet] = None,
                 speed_controller: Optional[SpeedController] = None,
                 speed_behavior: Optional[SpeedBehavior] = None,
                 controller: Optional[LQR] = None,
                 steer_controller: Optional[SteeringController] = None,
                 return_extra: bool = False):
        speed_controller: SpeedController = SpeedController() if speed_controller is None else speed_controller
        speed_behavior: SpeedBehavior = SpeedBehavior() if speed_behavior is None else speed_behavior
        steer_controller: SteeringController = SteeringController() if steer_controller is None else steer_controller
        lqr: LQR = LQR() if controller is None else controller
        super().__init__(lane, lqr, speed_behavior, speed_controller, steer_controller, return_extra)


LaneFollowerAgent = Union[LFAgentLQR, LFAgentPP, LFAgentLatMPC, LFAgentFullMPC, LFAgentStanley]
MapsConLF: Mapping[type(Union[LateralController, LatAndLonController]), type(LaneFollowerAgent)] = {
    LQR: LFAgentLQR,
    PurePursuit: LFAgentPP,
    Stanley: LFAgentStanley,
    NMPCLatKinCont: LFAgentLatMPC, NMPCLatKinDis: LFAgentLatMPC,
    NMPCFullKinCont: LFAgentFullMPC, NMPCFullKinDis: LFAgentFullMPC
}

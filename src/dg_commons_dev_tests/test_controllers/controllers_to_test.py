from sim_dev.agents.lane_followers import *
from dg_commons_dev.controllers.speed import SpeedController, SpeedControllerParam
from dg_commons_dev.behavior.behavior import SpeedBehaviorParam
from dg_commons_dev.controllers.full_controller_base import VehicleController
from dg_commons_dev.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons_dev.controllers.path_approximation_techniques import LinearPath
import copy
from dg_commons_dev.state_estimators.extended_kalman_filter import *
from dg_commons.sim.simulator import SimTime
from dg_commons_dev.controllers.steering_controllers import *
from dg_commons_dev.behavior.cruise import CruiseParams


DT: SimTime = SimTime("0.05")
DT_COMMANDS: SimTime = SimTime("0.1")
assert DT_COMMANDS % DT == SimTime(0)


state_estimator: type(Estimator) = ExtendedKalman
state_estimator_params: EstimatorParams = ExtendedKalmanParam(
    actual_model_var=SemiDef([i*1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
    actual_meas_var=SemiDef([i*0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]]),

    belief_model_var=SemiDef([i * 1 for i in [0.0001, 0.0001, 0.0001, 0.0001, 0.0001]]),
    belief_meas_var=SemiDef([i * 0 for i in [0.001, 0.001, 0.001, 0.001, 0.001]]),

    initial_variance=SemiDef(matrix=np.zeros((5, 5))),

    dropping_technique=LGB,
    dropping_params=LGBParam(
        failure_p=0.0,
    ),
    t_step=DT
)


TestLQR = VehicleController(
            controller=LQR,
            controller_params=LQRParam(
                r=SemiDef([0.5]),
                q=SemiDef(matrix=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0.01]])),
                t_step=float(DT_COMMANDS),
            ),

            longitudinal_controller=SpeedController,
            longitudinal_controller_params=SpeedControllerParam(
                kP=1,
                kI=0.01,
                kD=0.1
            ),

            speed_behavior_param=SpeedBehaviorParam(
                cruise_params=CruiseParams(
                    nominal_speed=8
                ),
                dt_commands=DT_COMMANDS
            ),

            state_estimator=state_estimator,
            state_estimator_params=state_estimator_params
        )

TestNMPCFullKinContPV = VehicleController(
        controller=NMPCFullKinCont,

        controller_params=NMPCFullKinContParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            delta_input_weight=1e-2,
            cost=QuadraticCost,
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(3)),
                r=SemiDef(matrix=np.eye(2))
            ),
            path_approx_technique=LinearPath,
            rear_axle=False,
            analytical=False,
        ),

        speed_behavior_param=SpeedBehaviorParam(
            cruise_params=CruiseParams(
                nominal_speed=8,
            ),
            dt_commands=DT_COMMANDS
        ),

        state_estimator=state_estimator,
        state_estimator_params=state_estimator_params,
    )
TestNMPCFullKinContPV.extra_folder_name = "path_variable"

TestNMPCFullKinContAN = copy.deepcopy(TestNMPCFullKinContPV)
if isinstance(TestNMPCFullKinContAN.controller_params, list):
    for i in range(len(TestNMPCFullKinContAN.controller_params)):
        TestNMPCFullKinContAN.controller_params[i].analytical = True
else:
    TestNMPCFullKinContAN.controller_params.analytical = True
TestNMPCFullKinContAN.extra_folder_name = "analytical"


TestNMPCFullKinDisPV = VehicleController(

        controller=NMPCFullKinDis,
        controller_params=NMPCFullKinDisParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            cost=QuadraticCost,
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(3)),
                r=SemiDef(matrix=np.eye(2))
            ),
            delta_input_weight=1e-2,
            dis_technique='Kinematic RK4',  # 'Kinematic Euler' or 'Kinematic RK4' or 'Anstrom Euler'
            path_approx_technique=LinearPath,
            dis_t=0.05,
            rear_axle=False,
            analytical=False
        ),

        speed_behavior_param=SpeedBehaviorParam(
            cruise_params=CruiseParams(
                nominal_speed=8
            ),
            dt_commands=DT_COMMANDS
        ),

        state_estimator=state_estimator,
        state_estimator_params=state_estimator_params,
)
TestNMPCFullKinDisPV.extra_folder_name = "path_variable"

TestNMPCFullKinDisAN = copy.deepcopy(TestNMPCFullKinDisPV)
if isinstance(TestNMPCFullKinDisAN.controller_params, list):
    for i in range(len(TestNMPCFullKinDisAN.controller_params)):
        TestNMPCFullKinDisAN.controller_params[i].analytical = True
else:
    TestNMPCFullKinDisAN.controller_params.analytical = True
TestNMPCFullKinDisAN.extra_folder_name = "analytical"


TestNMPCLatKinContPV = VehicleController(

        controller=NMPCLatKinCont,
        controller_params=NMPCLatKinContParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            cost=QuadraticCost,
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(2)),
                r=SemiDef(matrix=np.eye(1))
            ),
            delta_input_weight=1e-2,
            path_approx_technique=LinearPath,
            rear_axle=False,
            analytical=False
        ),

        longitudinal_controller=SpeedController,
        longitudinal_controller_params=SpeedControllerParam(
            kP=1,
            kI=0.01,
            kD=0.1
        ),

        speed_behavior_param=SpeedBehaviorParam(
            cruise_params=CruiseParams(
                nominal_speed=8
            ),
            dt_commands=DT_COMMANDS
        ),

        state_estimator=state_estimator,
        state_estimator_params=state_estimator_params,

)

TestNMPCLatKinContPV.extra_folder_name = "path_variable"

TestNMPCLatKinContAN = copy.deepcopy(TestNMPCLatKinContPV)
if isinstance(TestNMPCLatKinContAN.controller_params, list):
    for i in range(len(TestNMPCLatKinContAN.controller_params)):
        TestNMPCLatKinContAN.controller_params[i].analytical = True
else:
    TestNMPCLatKinContAN.controller_params.analytical = True
TestNMPCLatKinContAN.extra_folder_name = "analytical"


TestNMPCLatKinDisPV = VehicleController(
        controller=NMPCLatKinDis,
        controller_params=NMPCLatKinDisParam(
            n_horizon=15,
            t_step=float(DT_COMMANDS),
            cost=QuadraticCost,
            cost_params=QuadraticParams(
                q=SemiDef(matrix=np.eye(2)),
                r=SemiDef(matrix=np.eye(1))
            ),
            delta_input_weight=1e-2,
            dis_technique='Kinematic RK4',  # 'Kinematic Euler' or 'Kinematic RK4' or 'Anstrom Euler'
            path_approx_technique=LinearPath,
            dis_t=0.05,
            rear_axle=False,
            analytical=False
        ),

        longitudinal_controller=SpeedController,
        longitudinal_controller_params=SpeedControllerParam(
            kP=1,
            kI=0.01,
            kD=0.1
        ),

        speed_behavior_param=SpeedBehaviorParam(
            cruise_params=CruiseParams(
                nominal_speed=8
            ),
            dt_commands=DT_COMMANDS
        ),

        state_estimator=state_estimator,
        state_estimator_params=state_estimator_params,

)
TestNMPCLatKinDisPV.extra_folder_name = "path_variable"

TestNMPCLatKinDisAN = copy.deepcopy(TestNMPCLatKinDisPV)
if isinstance(TestNMPCLatKinDisAN.controller_params, list):
    for i in range(len(TestNMPCLatKinDisAN.controller_params)):
        TestNMPCLatKinDisAN.controller_params[i].analytical = True
else:
    TestNMPCLatKinDisAN.controller_params.analytical = True
TestNMPCLatKinDisAN.extra_folder_name = "analytical"


TestPurePursuit = VehicleController(

        controller=PurePursuit,
        controller_params=PurePursuitParam(
            k_lookahead=1.8,
            t_step=float(DT_COMMANDS)
        ),

        longitudinal_controller=SpeedController,
        longitudinal_controller_params=SpeedControllerParam(
            kP=1,
            kI=0.01,
            kD=0.1
        ),

        speed_behavior_param=SpeedBehaviorParam(
            cruise_params=CruiseParams(
                nominal_speed=8
            ),
            dt_commands=DT_COMMANDS
        ),

        steering_controller=SCP,
        steering_controller_params=SCPParam(
            ddelta_kp=10,
        ),

        state_estimator=state_estimator,
        state_estimator_params=state_estimator_params
    )


TestStanley = VehicleController(

        controller=Stanley,
        controller_params=StanleyParam(
            stanley_gain=2,
            t_step=float(DT_COMMANDS)
        ),

        longitudinal_controller=SpeedController,
        longitudinal_controller_params=SpeedControllerParam(
            kP=1,
            kI=0.01,
            kD=0.1
        ),

        speed_behavior_param=SpeedBehaviorParam(
            cruise_params=CruiseParams(
                nominal_speed=8
            ),
            dt_commands=DT_COMMANDS
        ),

        steering_controller=SCP,
        steering_controller_params=SCPParam(
            ddelta_kp=10,
        ),

        state_estimator=state_estimator,
        state_estimator_params=state_estimator_params
    )

contr_to_test = [TestLQR, TestStanley, TestPurePursuit,
                 TestNMPCFullKinContPV, TestNMPCFullKinContAN,
                 TestNMPCFullKinDisPV, TestNMPCFullKinDisAN,
                 TestNMPCLatKinContPV, TestNMPCLatKinContAN,
                 TestNMPCLatKinDisPV, TestNMPCLatKinDisAN]

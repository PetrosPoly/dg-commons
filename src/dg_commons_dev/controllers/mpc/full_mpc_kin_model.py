import do_mpc
from dg_commons_dev.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from geometry import translation_angle_from_SE2
from dg_commons_dev.controllers.mpc.mpc_utils.cost_functions import CostFunctions, CostParameters, \
    QuadraticCost, QuadraticParams
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from dg_commons_dev.controllers.path_approximation_techniques import PathApproximationTechniques, LinearPath
from casadi import *
from typing import List
from typing import Union, Optional, Tuple
from dg_commons.maps.lanes import DgLanelet
from dg_commons_dev.maps.lanes import DgLaneletControl
from abc import ABC, abstractmethod
from dg_commons import X
from dataclasses import dataclass
from dg_commons_dev.controllers.interface import Controller

vehicle_params = VehicleParameters.default_car()


@dataclass
class FullMPCKinParams:
    n_horizon: Union[List[int], int] = 15
    """ Horizon Length """
    t_step: Union[List[float], float] = 0.1
    """ Sample Time """
    cost: Union[List[CostFunctions], CostFunctions] = QuadraticCost
    """ Cost function """
    cost_params: Union[List[CostParameters], CostParameters] = QuadraticParams()
    """ Cost function parameters """
    delta_input_weight: Union[List[float], float] = 1e-2
    """ Weighting factor in cost function for varying input """
    vehicle_geometry: Union[List[VehicleGeometry], VehicleGeometry] = VehicleGeometry.default_car()

    v_delta_bounds: Union[List[Tuple[float, float]], Tuple[float, float]] = (-vehicle_params.ddelta_max,
                                                                             vehicle_params.ddelta_max)
    """ Ddelta Bounds """
    delta_bounds: Union[List[Tuple[float, float]], Tuple[float, float]] = (-vehicle_params.default_car().delta_max,
                                                                           vehicle_params.default_car().delta_max)
    """ Steering Bounds """
    path_approx_technique: Union[List[PathApproximationTechniques], PathApproximationTechniques] = LinearPath
    """ Path approximation technique """
    acc_bounds: Tuple[float, float] = vehicle_params.acc_limits
    """ Accelertion bounds """
    v_bounds: Tuple[float, float] = vehicle_params.vx_limits

    def __post_init__(self):
        if isinstance(self.cost, list):
            assert len(self.cost) == len(self.cost_params)
            for i, technique in enumerate(self.cost):
                assert MapCostParam[technique] == type(self.cost_params[i])

        else:
            assert MapCostParam[self.cost] == type(self.cost_params)

        super().__post_init__()


class FullMPCKin(Controller, ABC):
    USE_STEERING_VELOCITY: bool = True

    @abstractmethod
    def __init__(self, params: FullMPCKinParams):
        # First of all import your parameters: you can parametrize your choices and so on...
        self.params = params

        """############################### CREATE A MODEL ###############################"""

        model_type = 'continuous'  # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)
        """ Instantiate a continous model """

        self.state_x = self.model.set_variable(var_type='_x', var_name='state_x', shape=(1, 1))
        self.state_y = self.model.set_variable(var_type='_x', var_name='state_y', shape=(1, 1))
        self.theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
        self.v = self.model.set_variable(var_type='_x', var_name='v', shape=(1, 1))
        self.delta = self.model.set_variable(var_type='_x', var_name='delta', shape=(1, 1))
        """
        Create state variables, denoted as _x:
        state_x: x position
        state_y: y position
        theta: angle between x-axis and car orientation
        v: rear axle velocity
        delta: steering angle
        """

        self.a = self.model.set_variable(var_type='_u', var_name='a')
        self.v_delta = self.model.set_variable(var_type='_u', var_name='v_delta')
        """ 
        Create input variables, denoted as _u:
        a: rear axle acceleration along current direction
        v_delta: steering velocity
        """

        self.target_speed = self.model.set_variable(var_type='_tvp', var_name='target_speed', shape=(1, 1))
        self.path_approx: PathApproximationTechniques = self.params.path_approx_technique()
        self.path_params = self.model.set_variable(var_type='_tvp', var_name='path_params',
                                                   shape=(self.path_approx.n_params, 1))
        self.path_parameters = self.path_approx.n_params * [0]
        """  
        Create time varying parameters, denoted as _tvp:
        Really powerful tool: can be parametrized with time varying deterministic functions or with non-deterministic
        functions (see below self.func() below), where self.speed_ref change at every iterations depending on the 
        traffic and on other things. 
        You might want to use these for time-dependent constraints. Just a feeling, never done before with do_mpc.
        
        
        target_speed: this can potentially change at every iteration
        path_params: my personal implementation approximates the path at every iteration (in this example as a linear
            path). The parameters describing this line change at every iterations. 
        """

        self.cost = self.params.cost(self.params.cost_params)
        """ Instantiate cost function """

        dtheta = self.v * tan(self.delta) / self.params.vehicle_geometry.length
        vy = dtheta * self.params.vehicle_geometry.lr
        self.model.set_rhs('state_x', self.v * cos(self.theta) - vy * sin(self.theta))
        self.model.set_rhs('state_y', self.v * sin(self.theta) + vy * cos(self.theta))
        self.model.set_rhs('theta', dtheta)
        self.model.set_rhs('v', self.a)
        self.model.set_rhs('delta', self.v_delta)
        """ Create the model: Kinematic Bicycle Model in this case for cog """

        self.model.setup()
        """ Call this method to set up the model """

        """############################### SET UP MPC ###############################"""

        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }

        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**self.setup_mpc)
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)
        """ 
        Instantiate MPC and set up some parameters 
        the last part, suppress_ipopt, makes the solver less verbose
        """

        target_x, target_y, target_angle = self.compute_targets()
        """ Compute targets: closest position (pose) on path to my current position """
        lterm = self.lterm(target_x, target_y, self.target_speed)
        mterm = self.mterm(target_x, target_y, self.target_speed)
        """ Compute costs """

        self.mpc.set_objective(mterm=mterm, lterm=lterm)
        self.mpc.set_rterm(
            v_delta=self.params.delta_input_weight
        )
        """ Set the costs """

        self.set_bounds()
        self.set_scaling()
        """ Set bounds and scaling """

        self.tvp_temp = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.func)
        self.mpc.setup()
        """ Set up mpc """

        self.speed_ref = 0
        self.path: Optional[DgLanelet] = None
        self.control_path: Optional[DgLaneletControl] = None
        self.current_beta: Optional[float] = None
        self.u = None
        self.current_position, self.current_speed, self.current_beta, self.current_f = None, None, None, None
        self.prediction_x, self.prediction_y, self.target_position = None, None, None
        """ Some parameters initialization, not important """

    def func(self, t_now):
        """
        Function describing the behavior of the time-varying variable.
        It has to take time as input argument and can be deterministic and non-deterministic (simple function of time
        or, as in this case, can change depending on incoming observations).
        """
        temp = [self.speed_ref] + self.path_parameters
        self.tvp_temp['_tvp', :] = np.array(temp)
        return self.tvp_temp

    def set_scaling(self):
        self.mpc.scaling['_x', 'state_x'] = 1
        self.mpc.scaling['_x', 'state_y'] = 1
        self.mpc.scaling['_x', 'theta'] = 1
        self.mpc.scaling['_x', 'v'] = 1
        self.mpc.scaling['_x', 'delta'] = 1
        self.mpc.scaling['_u', 'v_delta'] = 1
        self.mpc.scaling['_u', 'a'] = 1

    def compute_targets(self):
        self.path_approx.update_from_parameters(self.path_params)
        return *self.path_approx.closest_point_on_path([self.state_x, self.state_y]), None

    def _update_obs(self, new_obs: Optional[X] = None):
        self.current_position = np.array([new_obs.x, new_obs.y])
        self.current_speed = new_obs.vx
        control_sol_params = self.control_path.ControlSolParams(self.current_speed, self.params.t_step)
        self.current_beta, _ = self.control_path.find_along_lane_closest_point(self.current_position,
                                                                               control_sol=control_sol_params)
        """ Update current state of the vehicle """
        pos1, angle1, pos2, angle2, pos3, angle3 = self.next_pos(self.current_beta)
        self.path_approx.update_from_data(pos1, angle1, pos2, angle2, pos3, angle3)
        params = self.path_approx.parameters
        """ Generate current path approximation """
        self.path_parameters = params[:self.path_approx.n_params]

        x0_temp = [self.current_position[0], self.current_position[1], new_obs.theta, self.current_speed, new_obs.delta]
        x0_temp = x0_temp if self.params.analytical else x0_temp + [pos1[0]]
        x0 = np.array(x0_temp).reshape(-1, 1)
        """ Define initial condition """
        self.mpc.x0 = x0
        self.mpc.set_initial_guess()
        self.u = self.mpc.make_step(x0)
        """ Compute input """

    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        """ This is the stage cost """
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        lterm, _ = self.cost.cost_function(error, inp)
        return lterm

    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        """ This is the terminal cost """
        error = [target_x - self.state_x, target_y - self.state_y, self.v - speed_ref]
        inp = [self.v_delta, self.a]
        _, mterm = self.cost.cost_function(error, inp)
        return mterm

    def set_bounds(self):
        """ Here you might set bounds with time-varying parameters. """
        self.mpc.bounds['lower', '_u', 'v_delta'] = self.params.v_delta_bounds[0]
        self.mpc.bounds['upper', '_u', 'v_delta'] = self.params.v_delta_bounds[1]
        self.mpc.bounds['lower', '_x', 'delta'] = self.params.delta_bounds[0]
        self.mpc.bounds['upper', '_x', 'delta'] = self.params.delta_bounds[1]
        self.mpc.bounds['lower', '_x', 'v'] = self.params.v_bounds[0]
        self.mpc.bounds['upper', '_x', 'v'] = self.params.v_bounds[1]
        self.mpc.bounds['lower', '_u', 'a'] = self.params.acc_bounds[0]
        self.mpc.bounds['upper', '_u', 'a'] = self.params.acc_bounds[1]

    def _update_reference_speed(self, speed_ref: float):
        self.speed_ref = speed_ref

    def _get_acceleration(self, at: float) -> float:
        return self.u[1][0]

    def _get_steering(self, at: float):
        return self.u[0][0]

    def next_pos(self, current_beta):
        """ Just a util method for computing the approximation of the path """
        along_lane = self.path.along_lane_from_beta(current_beta)
        delta_step = self.delta_step()
        along_lane1 = along_lane + delta_step/2
        along_lane2 = along_lane1 + delta_step/2

        beta1, beta2, beta3 = current_beta, self.path.beta_from_along_lane(along_lane1), \
            self.path.beta_from_along_lane(along_lane2)

        q1 = self.path.center_point(beta1)
        q2 = self.path.center_point(beta2)
        q3 = self.path.center_point(beta3)

        pos1, angle1 = translation_angle_from_SE2(q1)
        pos2, angle2 = translation_angle_from_SE2(q2)
        pos3, angle3 = translation_angle_from_SE2(q3)

        self.target_position = pos3
        return pos1, angle1, pos2, angle2, pos3, angle3

    def delta_step(self):
        """ Just a util method for computing the approximation of the path """
        return self.current_speed*self.params.t_step*self.params.n_horizon

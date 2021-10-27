from typing import Optional, Union, List
from abc import ABC, abstractmethod
from dg_commons import X
import do_mpc
from dataclasses import dataclass
from dg_commons_dev.controllers.mpc.mpc_utils.cost_functions import *
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons_dev.utils import BaseParams
import numpy as np
from dg_commons_dev.controllers.interface import Controller


@dataclass
class MPCKinBAseParam(BaseParams):
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
    rear_axle: Union[List[bool], bool] = False
    """ Whether to control rear axle position instead of cog """
    vehicle_geometry: Union[List[VehicleGeometry], VehicleGeometry] = VehicleGeometry.default_car()

    def __post_init__(self):
        if isinstance(self.cost, list):
            assert len(self.cost) == len(self.cost_params)
            for i, technique in enumerate(self.cost):
                assert MapCostParam[technique] == type(self.cost_params[i])

        else:
            assert MapCostParam[self.cost] == type(self.cost_params)

        super().__post_init__()


class MPCKinBase(Controller, ABC):
    @abstractmethod
    def __init__(self, params, model_type: str):
        self.params = params
        self.mpc = None

        self.setup_mpc = {
            'n_horizon': self.params.n_horizon,
            't_step': self.params.t_step,
            'store_full_solution': True,
        }

        self.model = do_mpc.model.Model(model_type)
        self.state_x = self.model.set_variable(var_type='_x', var_name='state_x', shape=(1, 1))
        self.state_y = self.model.set_variable(var_type='_x', var_name='state_y', shape=(1, 1))
        self.theta = self.model.set_variable(var_type='_x', var_name='theta', shape=(1, 1))
        self.v = self.model.set_variable(var_type='_x', var_name='v', shape=(1, 1))
        self.delta = self.model.set_variable(var_type='_x', var_name='delta', shape=(1, 1))

        self.target_speed = self.model.set_variable(var_type='_tvp', var_name='target_speed', shape=(1, 1))
        self.speed_ref = 0

    def __post_init__(self):
        assert self.mpc is not None

    def set_up_mpc(self):
        self.mpc = do_mpc.controller.MPC(self.model)
        self.mpc.set_param(**self.setup_mpc)
        suppress_ipopt = {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        self.mpc.set_param(nlpsol_opts=suppress_ipopt)
        target_x, target_y, target_angle = self.compute_targets()

        lterm = self.lterm(target_x, target_y, self.target_speed)
        mterm = self.mterm(target_x, target_y, self.target_speed)

        self.mpc.set_objective(mterm=mterm, lterm=lterm)

        self.mpc.set_rterm(
            v_delta=self.params.delta_input_weight
        )

        self.set_bounds()
        self.set_scaling()

        self.tvp_temp = self.mpc.get_tvp_template()
        self.mpc.set_tvp_fun(self.func)

        self.mpc.setup()

    def func(self, t_now):
        self.tvp_temp['_tvp', :] = np.array([self.speed_ref])
        return self.tvp_temp

    @abstractmethod
    def lterm(self, target_x, target_y, speed_ref, target_angle=None):
        pass

    @abstractmethod
    def mterm(self, target_x, target_y, speed_ref, target_angle=None):
        pass

    @abstractmethod
    def set_bounds(self):
        pass

    @abstractmethod
    def set_scaling(self):
        pass

    @abstractmethod
    def compute_targets(self):
        pass

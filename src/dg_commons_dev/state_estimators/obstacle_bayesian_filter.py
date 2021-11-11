import numpy as np
from geometry import SE2_from_xytheta
from dg_commons.sim.models.vehicle import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from typing import Tuple
from dg_commons import U
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.state_estimators.estimator_types import Estimator
from dg_commons_dev.state_estimators.utils import PDistributionDisParams, PDistributionDis, \
    Poisson, PoissonParams, GridTwoD
from dg_commons_dev.state_estimators.temp_curve import PCurve
from dg_commons.sim.models.vehicle import VehicleModel, VehicleState
from decimal import Decimal
from dg_commons.geo import SE2_apply_T2

lx = 0.5
""" Grid box length in x-direction """
ly = 0.5
""" Grid box length in y-direction """
lx_total = 5
""" Grid length in x-direction """
ly_total = 5
""" Grid length in y-direction """
A_obstacle = 1
""" Area occupied by obstacles inside the grid """

area = lx_total * ly_total
lamb = lx * ly * A_obstacle / area


@dataclass
class ObstacleBayesianParam(BaseParams):

    prior_distribution: type(PDistributionDis) = Poisson
    """ Prior probability distribution """
    prior_distribution_params: PDistributionDisParams = PoissonParams(lamb=lamb)
    """ Prior probability distribution parameters """

    fp_distribution: PCurve = PCurve(0.05)
    """ FN distribution """
    fn_distribution: PCurve = PCurve(0.05)
    """ FP distribution """
    acc_distribution: PCurve = PCurve(0.05)
    """ Acc distribution """

    grid_size: Tuple[float, float] = (lx_total, ly_total)
    """ x- and y-size of the grid wrt vehicle """
    grid_shape: Tuple[int, int] = (int(lx_total/lx), int(ly_total/ly))
    """ x- and y-size of a grid box """

    geometry_params: VehicleGeometry = VehicleGeometry.default_car()
    """ Vehicle Geometry """
    vehicle_params: VehicleParameters = VehicleParameters.default_car()
    """ Vehicle Parameters """
    t_step: float = 0.1
    """ Time interval between two calls """

    def __post_init__(self):
        assert isinstance(self.prior_distribution_params, self.prior_distribution.REF_PARAMS)


class ObstacleBayesian(Estimator):
    """ Bayesian estimator for presence of obstacles """
    REF_PARAMS: dataclass = ObstacleBayesianParam

    def __init__(self, params: ObstacleBayesianParam):
        self.params: ObstacleBayesianParam = params

        self.prior_distribution = self.params.prior_distribution(self.params.prior_distribution_params)
        self.prior_p = self.prior_distribution.pmf(1)

        values = self._values = [[self.prior_p] * self.params.grid_shape[1]] * self.params.grid_shape[0]
        self.current_belief = GridTwoD(self.params.grid_size[0], self.params.grid_shape[0],
                                       self.params.grid_size[1], self.params.grid_shape[1], values)

    def update_prediction(self, u_k: U) -> None:
        """
        Internal current belief get projected ahead by params.dt using the kinematic bicycle model
        and the input to the system u_k.
        @param u_k: Vehicle input k
        @return: None
        """
        vehicle_model: VehicleModel = VehicleModel(VehicleState(0, 0, 0, 0, 0),
                                                   vg=self.params.geometry_params,
                                                   vp=self.params.vehicle_params)
        vehicle_model.update(u_k, dt=Decimal(self.params.t_step))
        delta_state = vehicle_model.get_state()
        delta_SE2 = SE2_from_xytheta(delta_state.x, delta_state.y, delta_state.theta)

        for x_val in self.current_belief.length_grid.nodes:
            for y_val in self.current_belief.width_grid.nodes:
                new_position = list(SE2_apply_T2(delta_SE2, np.array([x_val, y_val])))
                try:
                    val = self.current_belief.value_by_position(new_position)
                except AssertionError:
                    val = self.prior_p

                indices = self.current_belief.index_by_position([x_val, y_val])
                assert abs(int(indices[0]) - indices[0]) < 10e-6 and abs(int(indices[1]) - indices[1]) < 10e-6
                indices = (int(indices[0]), int(indices[1]))
                self.current_belief.set(indices, val)

    def update_measurement(self, measurement_k: GridTwoD) -> None:
        """
        Internal current belief gets updated based on measurement at time step k. The measurement consists of a grid the
        same size as current belief and storing 1 at each node, where the lidar detected an obstacle and 0 otherwise.
        @param measurement_k: kth system measurement
        @return: None
        """
        assert len(self.current_belief.width_grid.nodes) == len(measurement_k.width_grid.nodes)
        assert len(self.current_belief.length_grid.nodes) == len(measurement_k.length_grid.nodes)
        for x_val in self.current_belief.length_grid.nodes:
            for y_val in self.current_belief.width_grid.nodes:
                position = [x_val, y_val]
                indices = self.current_belief.index_by_position(position)
                assert abs(int(indices[0]) - indices[0]) < 10e-6 and abs(int(indices[1]) - indices[1]) < 10e-6
                indices = (int(indices[0]), int(indices[1]))

                fn_p: float = self.params.fn_distribution.evaluate_distribution(np.array(position))
                fp_p: float = self.params.fp_distribution.evaluate_distribution(np.array(position))
                acc_p: float = self.params.acc_distribution.evaluate_distribution(np.array(position))

                prior = self.current_belief.value_by_index(indices)
                if measurement_k.value_by_position(position) == 1:
                    res = fn_p * prior / (fn_p * prior + fp_p * (1 - prior))
                else:
                    res = (1 - fn_p) * prior / ((1 - fn_p) * prior + fp_p * (1 - prior))

                self.current_belief.set(indices, res)

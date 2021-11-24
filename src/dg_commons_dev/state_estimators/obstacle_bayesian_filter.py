import copy
import itertools
import math
import numpy as np
from geometry import SE2_from_translation_angle, SE2value
from dg_commons.sim.models.vehicle import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters
from typing import Tuple, List
from dg_commons import U
from dataclasses import dataclass
from dg_commons_dev.utils import BaseParams
from dg_commons_dev.state_estimators.estimator_types import Estimator
from dg_commons_dev.state_estimators.utils import PDistributionDisParams, PDistributionDis, \
    Poisson, PoissonParams, CircularGrid
from dg_commons_dev.state_estimators.temp_curve import PCurve
from dg_commons.sim.models.vehicle import VehicleModel, VehicleState, VehicleCommands
from decimal import Decimal
from dg_commons.geo import SE2_apply_T2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from itertools import product

radius: float = 10
""" Radius of the circular grid considered """
point_density: float = 10
""" Density of points on a line [1/m] """
n_lines: int = 10
""" Number of lines """
n_points: int = math.floor(point_density*radius)
""" Number of points on a line """

average_ratio: float = 1/100
""" Average ratio (area of obstacle) / (area total) """
lamb = average_ratio * radius
""" Lambda for Poisson distribution """
# TODO: should be distance-dependent and should be dependent of grid cell area


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

    grid_radius: float = radius
    """ Radius of circular grid considered """
    n_points: int = n_points
    """ Number of points on a line """
    n_lines: int = n_lines
    """ Number of lines """
    distance_to_first_ring: float = 2
    """ Distance from center of lidar to first detection ring """

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

        self._values = np.array([[self.prior_p] * self.params.n_points] * self.params.n_lines)
        self.current_belief = CircularGrid(self.params.grid_radius, self.params.n_points,
                                           self.params.n_lines, self.params.distance_to_first_ring,
                                           self._values)

        self.fn = copy.deepcopy(self.current_belief)
        self.fp = copy.deepcopy(self.current_belief)
        self.acc = copy.deepcopy(self.current_belief)
        self.pos_x = self.current_belief.pos_x
        self.pos_y = self.current_belief.pos_y
        for idx, position in self.current_belief.gen_structure():
            self.fn.set(idx, self.params.fn_distribution.evaluate_distribution(np.array(position)))
            self.fp.set(idx, self.params.fp_distribution.evaluate_distribution(np.array(position)))
            self.acc.set(idx, self.params.acc_distribution.evaluate_distribution(np.array(position)))

        acc_areas = 4 * np.power(self.acc.as_numpy(), 2)
        acc_areas = np.where(acc_areas > 0, acc_areas, 10e-8)
        self.p_detection_given_origin = np.where(acc_areas > 0, np.divide(1, acc_areas), 0)

        self.data = []
        self.poses = []

    def update_prediction(self, u_k: U) -> None:
        """
        Internal current belief get projected ahead by params.dt using the kinematic bicycle model
        and the input to the system u_k.
        @param u_k: Vehicle input k
        @return: None
        """
        if u_k is None:
            return

        vehicle_model: VehicleModel = VehicleModel(VehicleState(0, 0, 0, 0, 0),
                                                   vg=self.params.geometry_params,
                                                   vp=self.params.vehicle_params)
        vehicle_model.update(u_k, dt=Decimal(self.params.t_step))
        delta_state = vehicle_model.get_state()
        delta_se2 = SE2_from_translation_angle(np.array([-delta_state.x, -delta_state.y]), -delta_state.theta)

        current_belief = copy.deepcopy(self.current_belief)
        for idx, position in current_belief.gen_structure():
            old_position = list(SE2_apply_T2(delta_se2, position))

            try:
                val = current_belief.value_by_position(old_position)
            except AssertionError:
                val = self.prior_p

            self.current_belief.set(idx, val)

    def update_measurement(self, detections: List[Tuple[float, float]], my_current_pose: SE2value) -> None:
        self.poses.append(my_current_pose)
        # For plotting purposes

        current_belief: CircularGrid = copy.deepcopy(self.current_belief)

        fn = self.fn.as_numpy()
        fp = self.fp.as_numpy()
        current_b = current_belief.as_numpy()

        mat = self.p_did_not_cause_any_given_d_only(detections)
        num = np.multiply(1-fn, current_b)
        den = num + np.multiply(fp, 1-current_b)
        p_obs_given_causing = np.divide(num, den)

        num = np.multiply(fn, current_b)
        den = num + np.multiply(1-fp, 1 - current_b)
        p_obs_given_not_causing = np.divide(num, den)

        result = np.multiply(p_obs_given_causing, 1-mat) + np.multiply(p_obs_given_not_causing, mat)
        self.current_belief.values = result
        val_of_interest = result[0, :]

        self.data.append(val_of_interest.tolist())

    def p_did_not_cause_any_given_d_only(self, detections: List[Tuple[float, float]]):
        acc = self.acc.as_numpy()
        shape = acc.shape

        involvement = {}
        candidates = []

        for detection in detections:
            in_accuracy: np.ndarray = np.where((self.pos_x - acc <= detection[0]) & (detection[0] <= self.pos_x + acc) &
                                               (self.pos_y - acc <= detection[1]) & (detection[1] <= self.pos_y + acc),
                                               self.p_detection_given_origin, 0)
            candidates_i = np.nonzero(in_accuracy)
            candidates.append(list(zip(list(candidates_i[0]), list(candidates_i[1]))))

            sum_acc_prob: float = float(np.sum(in_accuracy))
            in_accuracy = in_accuracy / sum_acc_prob if sum_acc_prob != 0 else in_accuracy

            involvement[detection] = in_accuracy

        def helper_fct(nth: int, idx: Tuple[int, int]):
            my_detection = detections[nth]
            candidate_of_interest = []

            idx_to_consider = []
            threshold = 1.0
            for i in range(nth-1):
                other_detection = detections[i]
                dist = np.linalg.norm(np.array([my_detection[0] - other_detection[0],
                                                my_detection[1] - other_detection[1]]))
                if dist <= threshold:
                    candidate_of_interest.append(candidates[i])
                    idx_to_consider.append(i)

            pairs = itertools.product(*candidate_of_interest)

            value = 0
            for count1, pair in enumerate(pairs):
                val = 1
                for count2, element in enumerate(pair):
                    # helper = 1 - previous_results[:, :, count2]
                    total_p = 1
                    helper = list(involvement.values())[idx_to_consider[count2]]
                    total_p -= helper[idx]
                    helper[idx] = 0
                    for i in range(count2):
                        pair_of_interest = pair[i]
                        total_p -= helper[pair_of_interest]
                        helper[pair_of_interest] = 0

                    val *= helper[element] / total_p if total_p != 0 else 0

                total_p = 1
                helper = involvement[detection]
                for element in pair:
                    total_p -= helper[element]
                    helper[element] = 0

                val *= 1 - helper[idx] / total_p if total_p != 0 else 1
                value += val
            return value

        p_matrices = np.ones(shape + (len(detections),))
        for nth, detection in enumerate(detections):
            for idx in candidates[nth]:
                p_matrices[idx + (nth, )] = helper_fct(nth, idx)

        mat = np.prod(p_matrices, axis=2)
        return mat

    def simulation_ended(self):
        fig, ax = plt.subplots()
        ax.set_xlabel('Longitudinal position [m]')
        ax.set_ylabel('Probability of obstacle')
        ax.set(ylim=(0, 1), xlim=(30, 70))

        n_x = self.current_belief.n_points
        x_values = self.pos_x[0, :]
        n_t = len(self.data)

        line, = ax.plot(np.array(x_values), np.zeros(n_x))
        txt = ax.text(0.1, 0.1, 'matplotlib', horizontalalignment='center',
                      verticalalignment='center', transform=ax.transAxes)

        def animate(i):
            pose = self.poses[i]
            x_values_world = [SE2_apply_T2(pose, np.array([val, 0]))[0] for val in x_values]
            line.set_xdata(x_values_world)
            line.set_ydata(np.array(self.data[i]))  # update the data.
            txt.set_text(str(round(i*0.05, 1)) + "s")
            return [line, txt]

        ani = animation.FuncAnimation(
              fig, animate, interval=n_t, blit=True)

        writer = animation.PillowWriter(
                    fps=20, metadata=dict(artist='Me'), bitrate=1800)
        ani.save("test.gif", writer=writer)

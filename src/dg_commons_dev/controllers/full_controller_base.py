from dg_commons_dev.controllers.controller_types import *
from sim_dev.agents.lane_followers import MapsConLF
import os
from dg_commons_dev.state_estimators.estimator_types import *
from dg_commons_dev.state_estimators.dropping_trechniques import *
from sim_dev.agents.lane_followers import LaneFollowerAgent
from dg_commons_dev.controllers.steering_controllers import *
from dg_commons_dev.behavior.behavior_types import Behavior, BehaviorParams


def func(arguments):
    return True


@dataclass
class VehicleController:

    controller: Union[List[type(Union[LateralController, LatAndLonController])],
                      type(Union[LateralController, LatAndLonController])]

    controller_params: Union[List[Union[LateralControllerParam, LatAndLonControllerParam]],
                             Union[LateralControllerParam, LatAndLonControllerParam]]

    speed_behavior_param: BehaviorParams

    longitudinal_controller: Union[List[Optional[type(LongitudinalController)]],
                                   Optional[type(LongitudinalController)]] = None

    longitudinal_controller_params: Union[List[Optional[LongitudinalControllerParam]],
                                          Optional[LongitudinalControllerParam]] = None

    steering_controller: Union[List[Optional[type(SteeringController)]],
                               Optional[type(SteeringController)]] = None

    steering_controller_params: Union[List[Optional[SteeringControllerParam]],
                                      Optional[SteeringControllerParam]] = None

    state_estimator: Union[List[Optional[type(Estimator)]],
                           Optional[type(Estimator)]] = None

    state_estimator_params: Union[List[Optional[EstimatorParams]],
                                  Optional[EstimatorParams]] = None

    condition: Callable = func

    lf_agent: Optional[LaneFollowerAgent] = None

    _is_single: Optional[bool] = None

    _extra_folder_name: str = ""

    folder_name: str = ""

    def __post_init__(self):
        self.single_controller = self.is_single_item(self.controller)

    def on_init(self):
        cond = all([self.single_controller, self.is_single_item(self.longitudinal_controller),
                    self.is_single_item(self.steering_controller), self.is_single_item(self.state_estimator),
                    self.is_single_param(self.controller_params),
                    self.is_single_param(self.longitudinal_controller_params),
                    self.is_single_param(self.steering_controller_params),
                    self.is_single_param(self.state_estimator_params),
                    self.is_single_param(self.speed_behavior_param)])

        self._is_single = cond
        if cond:
            self.process_single()
        else:
            self.process_multiple()

    def process_multiple(self):
        self.controller = self.to_list(self.controller)
        self.controller_params = self.to_list(self.controller_params)
        self.longitudinal_controller = self.to_list(self.longitudinal_controller)
        self.longitudinal_controller_params = self.to_list(self.longitudinal_controller_params)
        self.steering_controller = self.to_list(self.steering_controller)
        self.steering_controller_params = self.to_list(self.steering_controller_params)
        self.state_estimator = self.to_list(self.state_estimator)
        self.state_estimator_params = self.to_list(self.state_estimator_params)

        assert len(self.controller) == len(self.controller_params)
        n_controller = len(self.controller)
        assert len(self.longitudinal_controller) == len(self.longitudinal_controller_params)
        n_longi = len(self.longitudinal_controller)
        assert len(self.steering_controller) == len(self.steering_controller_params)
        n_steer = len(self.steering_controller)

        assert n_controller != 0
        self.triplets = []
        self.n_total = 0
        for i, controller in enumerate(self.controller):
            n_controller_count = self.controller_params[i].get_count()
            if not issubclass(controller, LatAndLonController):
                assert n_longi != 0
                for j, longi_params in enumerate(self.longitudinal_controller_params):
                    n_longi_count = longi_params.get_count()
                    if controller.USE_STEERING_VELOCITY:
                        self.triplets.append((i, j, None))
                        self.n_total += n_controller_count*n_longi_count
                    else:
                        assert n_steer != 0
                        for k, steer_params in enumerate(self.steering_controller_params):
                            n_steer_count = steer_params.get_count()
                            self.triplets.append((i, j, k))
                            self.n_total += n_controller_count*n_longi_count*n_steer_count
            else:
                if controller.USE_STEERING_VELOCITY:
                    self.triplets.append((i, None, None))
                    self.n_total += n_controller_count
                else:
                    assert n_steer != 0
                    for k, steer_params in enumerate(self.steering_controller_params):
                        n_steer_count = steer_params.get_count()
                        self.triplets.append((i, None, k))
                        self.n_total += n_controller_count*n_steer_count

        n_state_est = 0
        for count, se in enumerate(self.state_estimator):
            if se is not None:
                n_state_est += self.state_estimator_params[count].get_count()
            else:
                n_state_est = 1
                break
        n_speed_behavior = self.speed_behavior_param.get_count()

        self.n_total = self.n_total*n_state_est*n_speed_behavior
        self.n_triplets = len(self.triplets)

    def process_single(self):
        self.controller = self.to_val(self.controller)
        self.controller_params = self.to_val(self.controller_params)
        self.longitudinal_controller = self.to_val(self.longitudinal_controller)
        self.longitudinal_controller_params = self.to_val(self.longitudinal_controller_params)
        self.steering_controller = self.to_val(self.steering_controller)
        self.steering_controller_params = self.to_val(self.steering_controller_params)
        self.state_estimator = self.to_val(self.state_estimator)
        self.state_estimator_params = self.to_val(self.state_estimator_params)

        self.folder_name = self.controller.__name__
        self._is_single = True

        single: bool = issubclass(self.controller, LatAndLonController) and \
            self.longitudinal_controller is None
        decoupled: bool = not single and \
            issubclass(self.controller, LateralController) and \
            self.longitudinal_controller is not None and \
            issubclass(self.longitudinal_controller, LongitudinalController)

        assert single or decoupled
        self.decoupled = decoupled

        if self.steering_controller is None:
            self.steering_controller = SCIdentity
            self.steering_controller_params = SCIdentityParam()

        if self.longitudinal_controller is not None:
            assert self.longitudinal_controller_params is not None
        if self.state_estimator is not None:
            assert self.state_estimator_params is not None

        self.lf_agent = MapsConLF[self.controller]
        self.n_total = 1

    @staticmethod
    def to_list(val):
        val = val if isinstance(val, list) else [val]
        return val

    @staticmethod
    def to_val(val):
        val = val if not isinstance(val, list) else val[0]
        return val

    def get_count(self):
        return self.n_total

    @staticmethod
    def is_single_item(val):
        is_list = isinstance(val, list)
        return not is_list or len(val) == 1

    @staticmethod
    def is_single_param(val):
        is_list = isinstance(val, list)
        if is_list:
            if len(val) == 1:
                return val[0] is None or val[0].get_count() == 1
        else:
            return val is None or val.get_count() == 1
        return False

    @property
    def extra_folder_name(self):
        return self._extra_folder_name

    @extra_folder_name.setter
    def extra_folder_name(self, name):
        assert self.single_controller
        controller = self.to_val(self.controller)
        self._extra_folder_name = name
        self.folder_name = os.path.join(controller.__name__, name)

    def add_sub_folder(self, name):
        assert self.single_controller
        controller = self.to_val(self.controller)
        self._extra_folder_name = os.path.join(self.extra_folder_name, name)
        self.folder_name = os.path.join(controller.__name__, self._extra_folder_name)

    def gen(self):
        if self._is_single:
            yield self
        else:
            counter = 0
            for i in range(self.n_triplets):
                triplet = self.triplets[i]
                c_idx, l_idx, s_idx = triplet[0], triplet[1], triplet[2]

                controller = self.controller[c_idx]
                controller_params = list(self.controller_params[c_idx].gen())

                longi = self.longitudinal_controller[l_idx] if l_idx is not None else [None]
                longi_params = list(self.longitudinal_controller_params[l_idx].gen()) if l_idx is not None else [None]

                steer = self.steering_controller[s_idx] if s_idx is not None else [SCIdentity]
                steer_params = list(self.steering_controller_params[s_idx].gen()) if s_idx is not None else [SCIdentityParam()]

                for controller_param in controller_params:
                    for steer_param in steer_params:
                        for longi_param in longi_params:
                            for speed_behavior_param in self.speed_behavior_param.gen():
                                for count, state_estimator in enumerate(self.state_estimator):
                                    for state_estimator_param in self.state_estimator_params[count].gen():
                                        counter += 1
                                        res = VehicleController(
                                            controller=controller,
                                            controller_params=controller_param,
                                            speed_behavior_param=speed_behavior_param,
                                            steering_controller=steer,
                                            steering_controller_params=steer_param,
                                            longitudinal_controller=longi,
                                            longitudinal_controller_params=longi_param,
                                            state_estimator=state_estimator,
                                            state_estimator_params=state_estimator_param,
                                            )
                                        res.on_init()
                                        if self.condition(res):
                                            yield res

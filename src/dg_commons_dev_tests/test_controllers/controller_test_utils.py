from sim_dev.simulator import SimContext, Simulator, SimParameters, SimLog
from dg_commons.sim.models.vehicle import VehicleModel
from dg_commons.sim.models.vehicle_dynamic import VehicleModelDyn, VehicleStateDyn
from commonroad.scenario.obstacle import DynamicObstacle, StaticObstacle
from dg_commons.sim.scenarios.agent_from_commonroad import infer_lane_from_dyn_obs
from dg_commons_dev_tests.test_controllers.controller_scenarios.scenario_to_test import ScenarioData
from dg_commons_dev_tests.test_controllers.controllers_to_test import *
from dg_commons_dev.analysis.metrics import *
from dg_commons_dev.analysis.metrics_def import *
import matplotlib.pyplot as plt
from typing import List
import os
import math
from dg_commons_dev.controllers.full_controller_base import VehicleController
from dg_commons_dev.behavior.behavior import SpeedBehavior
import numpy as np
import time
from dg_commons_dev_tests.test_controllers.controller_scenarios.scenario_to_test import scenarios
from dg_commons_dev_tests.test_controllers.controller_scenarios.utils import model_agent_from_static_obstacle
from dg_commons_dev_tests.test_controllers.controllers_to_test import contr_to_test, DT, DT_COMMANDS

# TODO: this is a workaround and requires driving-games in the same folder as dg_commons
# from crash.reports import generate_report #TODO: find solution
from dg_commons_dev_tests.test_controllers.controller_scenarios.utils import get_project_root_dir
import sys
import os
sys.path.insert(0, os.path.join(get_project_root_dir(), "src"))
from crash.reports import generate_report


class Verbosity:
    def __init__(self, val: int):
        assert val in [0, 1, 2]
        self._val = val

    @property
    def val(self):
        return self._val


@dataclass
class Select:
    item: Union[VehicleController, type(Metrics), ScenarioData]
    test: bool

    def __post_init__(self):
        if self.test and hasattr(self.item, 'on_init'):
            self.item.on_init()


class TestMultipleControllerInstances:

    def __init__(self, controllers_to_test, metrics_to_test, scenarios_to_test, verbosity: Verbosity = Verbosity(1)):
        assert all([isinstance(c.item, VehicleController) for c in controllers_to_test])
        # assert all([isinstance(c.item(), Metrics) for c in metrics])
        assert all([isinstance(c.item, ScenarioData) for c in scenarios_to_test])

        assert set(scenarios.keys()) == set([item.item.fig_name for item in scenarios_to_test])
        helper = [item.item for item in controllers_to_test]
        assert all([contr in helper for contr in contr_to_test])
        assert all([contr in contr_to_test for contr in helper])
        helper = [item.item for item in metrics_to_test]
        assert all([metr in helper for metr in metrics_list])
        assert all([metr in metrics_list for metr in helper])

        self.controllers: List[Select] = controllers_to_test
        self.metrics: List[Select] = metrics_to_test
        self.scenarios: List[Select] = scenarios_to_test

        self.data = None
        self.controllers_data = []
        self.scenarios_data = []
        self.timing = {}
        self.verbosity = verbosity

        n_scenarios = sum([1 for s in scenarios_to_test if s.test])
        n_controllers = sum([s.item.get_count() for s in controllers_to_test if s.test])
        self.steps: int = n_controllers * n_scenarios

    def run(self):
        metrics_to_test = [met.item for met in self.metrics if met.test]
        self.data = {key.item.__name__: [] for key in self.metrics}
        counter = 0

        if self.verbosity.val > 0:
            print("[Testing Cases]... There are {} testing cases".format(self.steps))

        t1 = time.time()
        for controllers_to_test in self.controllers:
            if controllers_to_test.test:
                it = 0
                for controller_to_test in controllers_to_test.item.gen():
                    controller_to_test.add_sub_folder("Test{}".format(it))
                    root_name = controller_to_test.folder_name + "Test{}".format(it)
                    it += 1
                    for scenario_to_test in self.scenarios:
                        if scenario_to_test.test:
                            dict_name = root_name + scenario_to_test.item.fig_name
                            counter += 1
                            scenario = scenario_to_test.item
                            result = self.run_single(controller_to_test, metrics_to_test, scenario_to_test.item)
                            self.collect_data(dict_name, result, scenario.fig_name,
                                              round(counter / self.steps * 100, 3),
                                              controllers_to_test.item.folder_name)

        t2 = time.time()
        if self.verbosity.val > 0:
            print("The whole process took {} seconds".format(round(t2 - t1, 3)))

        self.show_data()

    def collect_data(self, dict_name, result, scenario_fig_name, percentage, controller_folder_name):
        self.timing[dict_name] = {}
        helper = {key: [] for key in self.data.keys()}
        if self.verbosity.val > 0:
            print("[Testing]...")
            print("[Controller]...", controller_folder_name)
            print("[Scenario]...", scenario_fig_name)
            print("[Percentage]...", percentage, "%")
            for key in result[0].keys():
                print()
                print("[Results for {}]...".format(key))
                for item in result:
                    if item[key].title == DTForCommand.__name__:
                        delta = round(float(np.average(np.array(item[key].incremental.values))), 4)
                        print(DTForCommand.__name__, " = {}".format(delta))
                        helper[DTForCommand.__name__].append(delta)
                    else:
                        helper[item[key].title].append(item[key].total)
                        print(item[key])

        for key in self.data.keys():
            self.data[key].append(np.average(np.array(helper[key])))
        self.controllers_data.append(controller_folder_name)
        self.scenarios_data.append(scenario_fig_name)
        self.timing[dict_name][scenario_fig_name] = np.average(np.array(helper[DTForCommand.__name__]))

    def show_data(self):
        name = "OVERALL STATISTICS"
        output_dir = os.path.join("out", "simulation_timing_statistics")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        def show_results(ch_name, data, ch_controllers, plots=False):
            if self.verbosity.val > 1:
                print(ch_name)

            for key in data.keys():
                if self.verbosity.val > 1:
                    minimum = min(data[key])
                    idx_min = data[key].index(minimum)
                    maximum = max(data[key])
                    idx_max = data[key].index(maximum)
                    print("Total/Average {}: {}".format(key, round(sum(data[key]) / len(data[key]), 3)))
                    max_scene = ", Scenario: " + self.scenarios_data[idx_max] if ch_name == "OVERALL STATISTICS" else ""
                    print("Max {} is {} for: ".format(key, round(maximum, 3)),
                          " Controller: {}{}".format(ch_controllers[idx_max], max_scene))
                    min_scene = ", Scenario: " + self.scenarios_data[idx_min] if name == "OVERALL STATISTICS" else ""
                    print("Min {} is {} for: ".format(key, round(minimum, 3)),
                          " Controller: {}{}".format(ch_controllers[idx_min], min_scene))
                    print()
                if plots:
                    plt.hist(data[key])
                    plt.title(ch_name + " " + key)
                    plt.savefig(os.path.join(output_dir, ch_name + key))
                    plt.clf()

        scenario_list = list(dict.fromkeys(self.scenarios_data))
        show_results(name, self.data, self.controllers_data)
        for scene in scenario_list:
            helper = [i for i, t in enumerate(self.scenarios_data) if t == scene]
            temp_data = {key: [self.data[key][i] for i in helper] for key in self.data.keys()}
            temp_controller = [self.controllers_data[i] for i in helper]
            name = "STATISTICS ABOUT " + scene.upper()
            show_results(name, temp_data, temp_controller)

    def run_single(self, controller, metric, scenario):
        test = TestSingleControllerInstance(scenario=scenario, metrics=metric, controller=controller)
        test.run()
        test.evaluate_metrics()
        return test.result


class TestSingleControllerInstance:

    def __init__(self, scenario: ScenarioData, metrics, controller: VehicleController,
                 vehicle_model: Optional[str] = None):

        self.scenario: ScenarioData = scenario
        self.lanelet_net = scenario.scenario.lanelet_network
        self.agent = controller.lf_agent
        self.controller = controller
        self.vehicle_model = vehicle_model

        players, models = {}, {}
        dyn_obstacles = scenario.scenario.dynamic_obstacles
        for i, dyn_obs in enumerate(dyn_obstacles):
            if i in scenario.cars_idx:
                if scenario.nominal_speed is not None:
                    idx = scenario.cars_idx.index(i)
                    self.controller.speed_behavior_param.cruise_params.nominal_speed = scenario.nominal_speed[idx]
                agent = self._agent_from_dynamic_obstacle(dyn_obs)
                model, estimator = self._model_se_from_dynamic_obstacle(dyn_obs, False)
                agent.set_state_estimator(estimator)
                player_name = PlayerName(f"P{i}")
                players.update({player_name: agent})
                models.update({player_name: model})

        if self.scenario.static_obs is not None:
            for j, static_obs in enumerate(self.scenario.scenario.static_obstacles):
                if j in self.scenario.static_obs:
                    model, agent = model_agent_from_static_obstacle(static_obs)
                    player_name = PlayerName(f"PStatic{j}")
                    players.update({player_name: agent})
                    models.update({player_name: model})

        sim_parameters: SimParameters = SimParameters(dt=DT, dt_commands=DT_COMMANDS,
                                                      max_sim_time=SimTime(scenario.simulation_time))
        self.sim_context: SimContext = SimContext(scenario=scenario.scenario, models=models, players=players,
                                                  param=sim_parameters, log=SimLog())
        self.metrics = metrics
        self.metrics_context: Optional[MetricEvaluationContext] = None
        self.result = []
        self.simulator: Simulator = Simulator()

        self.output_dir = os.path.join("out", self.controller.folder_name, self.scenario.fig_name)

    def _agent_from_dynamic_obstacle(self, dyn_obs: DynamicObstacle):
        controller = self.controller.controller(self.controller.controller_params)
        speed_behavior = SpeedBehavior(self.controller.speed_behavior_param)
        steering_controller = self.controller.steering_controller()
        steering_controller.params = self.controller.steering_controller_params

        dg_lane = infer_lane_from_dyn_obs(dyn_obs, self.lanelet_net)
        longitudinal_controller = None

        if self.controller.longitudinal_controller:
            longitudinal_controller = self.controller.longitudinal_controller()
            longitudinal_controller.params = self.controller.longitudinal_controller_params

        agent = self.agent(dg_lane, controller=controller, speed_behavior=speed_behavior,
                           speed_controller=longitudinal_controller, steer_controller=steering_controller,
                           return_extra=True)

        return agent

    def _model_se_from_dynamic_obstacle(self, dyn_obs: DynamicObstacle, is_dynamic: bool):
        orient_0, orient_1 = dyn_obs.prediction.trajectory.state_list[0].orientation, \
                             dyn_obs.prediction.trajectory.state_list[1].orientation
        vel_0 = dyn_obs.prediction.trajectory.state_list[0].velocity
        dtheta = orient_1 - orient_0
        l = dyn_obs.obstacle_shape.length
        delta_0 = math.atan(l * dtheta / vel_0) if vel_0 > 10e-6 else 0

        state_estimator = None
        if is_dynamic:
            x0 = VehicleStateDyn(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                                 theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                                 delta=delta_0, vy=0, dtheta=dtheta)
            model = VehicleModelDyn.default_car(x0=x0)
        else:
            x0 = VehicleState(x=dyn_obs.initial_state.position[0], y=dyn_obs.initial_state.position[1],
                              theta=dyn_obs.initial_state.orientation, vx=dyn_obs.initial_state.velocity,
                              delta=delta_0)
            model = VehicleModel.default_car(x0=x0)
            if self.controller.state_estimator:
                state_estimator = self.controller.state_estimator(params=self.controller.state_estimator_params)

        return model, state_estimator

    def run(self):
        self.simulator.run(self.sim_context)
        name = "simulation"

        nominal_velocity = self.controller.speed_behavior_param.cruise_params.nominal_speed
        dg_lanelets = {}
        states = {}
        commands = {}
        velocities = {}
        betas = {}
        dt_commands = {}
        for key in self.sim_context.log.keys():
            if not issubclass(type(self.sim_context.players[key]), LFAgent):
                continue

            dg_lanelets[key] = self.sim_context.players[key].current_ref.path
            states[key] = self.sim_context.log[key].states
            commands[key] = self.sim_context.log[key].actions
            dt_timestamps = states[key].timestamps
            dt_commands_timestamps = self.sim_context.log[key].actions.timestamps
            velocities[key] = DgSampledSequence(dt_timestamps, len(dt_timestamps)*[nominal_velocity])
            betas[key] = DgSampledSequence(dt_commands_timestamps, self.sim_context.players[key].betas)
            dt_commands[key] = DgSampledSequence(dt_commands_timestamps, self.sim_context.players[key].dt_commands)

        self.metrics_context = MetricEvaluationContext(dg_lanelets, states, commands,
                                                       velocities, dt_commands, betas)

        report = generate_report(self.sim_context)
        # save report
        report_file = os.path.join(self.output_dir, f"{name}.html")
        report.to_html(report_file)

    def evaluate_metrics(self):

        if self.metrics is not None:
            self.result = []
            for metric in self.metrics:
                met = metric()
                self.result.append(met.evaluate(self.metrics_context, plot=True, output_dir=self.output_dir))
        else:
            print("No Metric to Evaluate")

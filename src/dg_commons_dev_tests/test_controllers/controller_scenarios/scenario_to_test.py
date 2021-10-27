from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Any, Union
from dg_commons.sim.scenarios.utils import load_commonroad_scenario
from commonroad.scenario.scenario import Scenario
from dg_commons_dev_tests.test_controllers.controller_scenarios.utils import race_track_generate_dyn_obs, \
    SCENARIOS_DIR, collision_generate_dyn_obs


@dataclass
class ScenarioData:
    scenario_name: str

    fig_name: str

    cars_idx: Optional[List[int]] = None

    simulation_time: float = 6

    params: Optional[List[float]] = None

    static_obs: Optional[List[int]] = None

    nominal_speed: Optional[List[float]] = None

    def on_init(self):
        self.scenario: Scenario
        self.scenario, _ = load_commonroad_scenario(self.scenario_name, SCENARIOS_DIR)

        if self.scenario_name == "DEU_Hhr-1_1":
            dyn_obs = race_track_generate_dyn_obs(self.scenario,
                                                  starting_position=self.params[0],
                                                  length_perc=self.params[1]) if self.params else \
                                                  race_track_generate_dyn_obs(self.scenario)
            self.scenario.add_objects(dyn_obs[0])

        if self.scenario_name == "ZAM_Urban-2_1":
            dyn_obs = collision_generate_dyn_obs(self.scenario,
                                                 starting_distance=self.params[0],
                                                 starting_vel=self.params[1])
            self.scenario.add_objects(dyn_obs[0])

        n_cars = len(self.scenario.dynamic_obstacles)
        assert all([idx <= n_cars for idx in self.cars_idx])
        if self.nominal_speed is not None:
            assert len(self.nominal_speed) == len(self.cars_idx)


scenarios: Dict[str, ScenarioData] = {
    "lane_change_left": ScenarioData("ZAM_Zip-1_60_T-1", "lane_change_left", [1], simulation_time=7),
    "turn_90_right": ScenarioData("ARG_Carcarana-1_1_T-1", "turn_90_right", [1], simulation_time=7),
    "turn_90_left": ScenarioData("DEU_Muehlhausen-2_2_T-1", "turn_90_left", [3], simulation_time=5),
    "small_snake": ScenarioData("ZAM_Tjunction-1_320_T-1", "small_snake", [3], simulation_time=6),
    "u-turn": ScenarioData("DEU_Hhr-1_1", "u-turn", [0],  simulation_time=30, params=[43, 10]),
    "left_cont_curve": ScenarioData("DEU_Hhr-1_1", "left_cont_curve", [0], simulation_time=15, params=[60, 10]),
    "vertical": ScenarioData("USA_Peach-3_2_T-1", "vertical", [4, 7], simulation_time=7),
    "cruise_control": ScenarioData("USA_US101-14_1_T-1", "cruise_control", [30, 22], simulation_time=6,
                                   nominal_speed=[10, 3]),
    "emergency_brake": ScenarioData("ZAM_Urban-2_1", "emergency_brake", [0], simulation_time=5,
                                    static_obs=[1], params=[20, 11]),
    "race": ScenarioData("DEU_Hhr-1_1", "race", [0],  simulation_time=180, params=[0, 95])
}

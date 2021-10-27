from dg_commons.sim.scenarios.utils import load_commonroad_scenario, Scenario
from dg_commons.sim.scenarios.agent_from_commonroad import *
from dg_commons import PlayerName
from dg_commons.sim.simulator import SimContext, Simulator, SimParameters, SimLog
import os
#from crash.reports import generate_report
from dg_commons_dev_tests.test_controllers.controller_scenarios.utils import race_track_generate_dyn_obs, \
    SCENARIOS_DIR, collision_generate_dyn_obs, model_agent_from_static_obstacle, \
    model_agent_from_dynamic_obstacle_mine
from dg_commons.sim import SimTime


def collision(scenario: Scenario):
    players, models = {}, {}
    lanelet_net = scenario.lanelet_network

    dyn_obs = collision_generate_dyn_obs(scenario)

    model1, agent1 = model_agent_from_dynamic_obstacle_mine(dyn_obs[0], lanelet_net)
    player_name = PlayerName(f"Player0")
    players.update({player_name: agent1})
    models.update({player_name: model1})
    return players, models


def race_track(scenario: Scenario):
    players, models = {}, {}
    lanelet_net = scenario.lanelet_network

    dyn_obs = race_track_generate_dyn_obs(scenario, 60, 10)

    model1, agent1 = model_agent_from_dynamic_obstacle(dyn_obs[0], lanelet_net)
    player_name = PlayerName(f"Player0")
    players.update({player_name: agent1})
    models.update({player_name: model1})
    return players, models


def visualize_scenario(scenario_name: str):
    scenario, _ = load_commonroad_scenario(scenario_name, SCENARIOS_DIR)
    lanelet_net = scenario.lanelet_network
    players, models = {}, {}

    if scenario_name == "DEU_Hhr-1_1":
        players, models = race_track(scenario)

    if scenario_name == "ZAM_Urban-2_1":
        players, models = collision(scenario)

    for i, dyn_obs in enumerate(scenario.dynamic_obstacles):
        if dyn_obs.obstacle_type is not ObstacleType.CAR:
            continue

        model, agent = model_agent_from_dynamic_obstacle_mine(dyn_obs, lanelet_net)
        player_name = PlayerName(f"P{i}")
        players.update({player_name: agent})
        models.update({player_name: model})

    for i, static_obs in enumerate(scenario.static_obstacles):
        if scenario_name == "ZAM_Urban-2_1" and i == 0:
            continue
        model, agent = model_agent_from_static_obstacle(static_obs)
        player_name = PlayerName(f"PStatic{i}")
        players.update({player_name: agent})
        models.update({player_name: model})

    sim_parameters: SimParameters = SimParameters(max_sim_time=SimTime(5))
    sim_context: SimContext = SimContext(scenario=scenario, models=models, players=players,
                                         param=sim_parameters, log=SimLog())

    simulator: Simulator = Simulator()

    simulator.run(sim_context)
    name = "test_" + scenario_name

    #report = generate_report(sim_context)
    # save report
    # output_dir = "out"
    #report_file = os.path.join(output_dir, f"{name}.html")
    #report.to_html(report_file)


scenario_name = "USA_US101-14_1_T-1"
visualize_scenario(scenario_name)

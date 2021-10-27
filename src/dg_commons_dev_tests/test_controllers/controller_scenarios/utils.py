import math
import numpy as np
from dg_commons.sim.scenarios.utils import Scenario
from commonroad.visualization.util import Rectangle
from commonroad.prediction.prediction import TrajectoryPrediction, Trajectory
from dg_commons.sim.scenarios.agent_from_commonroad import *
import matplotlib.pyplot as plt
import os
import re
from dg_commons.sim.models.vehicle_dynamic import VehicleModel, VehicleState, VehicleCommands
from sim_dev.agents.lane_follower_z import LFAgent
from dg_commons.sim.agents.agent import PolicyAgent
from dg_commons import X, U


# TODO: this is a workaround and requires driving-games in the same folder as dg_commons
def get_project_root_dir() -> str:
    project_root_dir = __file__
    src_folder = "src"
    assert src_folder in project_root_dir, project_root_dir
    project_root_dir = re.split(src_folder, project_root_dir)[0]
    assert os.path.isdir(project_root_dir)
    driving_games_dir = os.path.join(project_root_dir, "..", "driving-games")
    assert os.path.isdir(driving_games_dir)
    return driving_games_dir


SCENARIOS_DIR = os.path.join(get_project_root_dir(), "scenarios")


def race_track_generate_dyn_obs(scenario: Scenario, starting_position: float = 0, length_perc: float = 100):
    lanelets = scenario.lanelet_network.lanelets
    states = []
    n_states = []

    for lanelet in lanelets:
        center_vertices = lanelet.center_vertices.tolist()
        n_states.append(len(center_vertices))
        for i, center_vertice in enumerate(center_vertices):
            n = len(center_vertices)
            if i == n-1:
                helper = center_vertices[i-1]
                orientation = math.atan2(center_vertice[1] - helper[1], center_vertice[0] - helper[0])
            else:
                helper = center_vertices[i + 1]
                orientation = math.atan2(helper[1] - center_vertice[1], helper[0] - center_vertice[0])
            state = State(position=np.array(center_vertice), orientation=orientation, time_step=0, velocity=0.0)
            states.append(state)

    further_separation = 2
    n_lanelet = int(len(lanelets)*starting_position/100)
    n_center = sum(n_states[:n_lanelet])-1+further_separation

    dec_states = states[n_center:]
    length = int(len(states) * length_perc / 100)
    n_add_states = length - len(dec_states)
    if n_add_states <= 0:
        dec_states = states[n_center:(n_center+length)]
    else:
        dec_states = dec_states + states[:n_add_states]

    '''x, y = [q.position[0] for q in dec_states], [q.position[1] for q in dec_states]
    plt.scatter([x[0], x[-1]], [y[0], y[-1]])
    plt.plot(x, y)
    plt.savefig("Test")'''

    initial_state = dec_states[0]
    trajectory = Trajectory(0, dec_states)

    dyn_obs = DynamicObstacle(obstacle_id=0, obstacle_type=ObstacleType.CAR,
                              obstacle_shape=Rectangle(1, 1), initial_state=initial_state,
                              prediction=TrajectoryPrediction(trajectory, shape=Rectangle(1, 1)))

    return [dyn_obs]


def collision_generate_dyn_obs(scenario, starting_distance=15, starting_vel=5):
    lanelets = scenario.lanelet_network.lanelets
    lanelet = lanelets[0]
    center_vertices = lanelet.center_vertices.tolist()
    n = len(center_vertices)
    states = []

    for i, center_vertice in enumerate(center_vertices):
        if i == n - 1:
            helper = center_vertices[i - 1]
            orientation = math.atan2(center_vertice[1] - helper[1], center_vertice[0] - helper[0])
        else:
            helper = center_vertices[i + 1]
            orientation = math.atan2(helper[1] - center_vertice[1], helper[0] - center_vertice[0])
        state = State(position=np.array(center_vertice), orientation=orientation, time_step=0, velocity=0.0)
        states.append(state)

    start = int(70 - starting_distance)
    states = states[start:]
    initial_state = states[0]
    trajectory = Trajectory(0, states)
    dyn_obs = DynamicObstacle(obstacle_id=0, obstacle_type=ObstacleType.CAR,
                              obstacle_shape=Rectangle(1, 1), initial_state=initial_state,
                              prediction=TrajectoryPrediction(trajectory, shape=Rectangle(1, 1)))
    dyn_obs.initial_state.velocity = starting_vel
    return [dyn_obs]


def model_agent_from_dynamic_obstacle_mine(
    dyn_obs: DynamicObstacle, lanelet_network: LaneletNetwork, color: Color = "royalblue"
) -> (VehicleModel, Agent):
    if not dyn_obs.obstacle_type == ObstacleType.CAR:
        raise NotSupportedConversion(commonroad=dyn_obs.obstacle_type)

    axle_length_ratio = 0.8  # the distance between wheels is less than the car body
    l = dyn_obs.obstacle_shape.length * axle_length_ratio
    dtheta = dyn_obs.prediction.trajectory.state_list[0].orientation - dyn_obs.initial_state.orientation
    delta = dtheta / l
    x0 = VehicleState(
        x=dyn_obs.initial_state.position[0],
        y=dyn_obs.initial_state.position[1],
        theta=dyn_obs.initial_state.orientation,
        vx=dyn_obs.initial_state.velocity,
        delta=delta,
    )
    model = VehicleModel.default_car(x0)

    # Agent
    dglane = infer_lane_from_dyn_obs(dyn_obs=dyn_obs, network=lanelet_network)
    agent = LFAgent.get_default_la(dglane)
    agent.state = x0
    return model, agent


def static_policy(state: X) -> U:
    return VehicleCommands(acc=0, ddelta=0)


def model_agent_from_static_obstacle(static_obs,  color: Color = "royalblue") -> (VehicleModel, Agent):

    x0 = VehicleState(
        x=static_obs.initial_state.position[0],
        y=static_obs.initial_state.position[1],
        theta=static_obs.initial_state.orientation,
        vx=0,
        delta=0,
    )
    model = VehicleModel.default_car(x0)

    # Agent
    agent = PolicyAgent(static_policy)
    return model, agent

import numpy as np

from dg_commons_dev.behavior.behavior_types import Situation
from dataclasses import dataclass, field
from typing import Union, List, Tuple, MutableMapping, Optional
from dg_commons_dev.behavior.utils import SituationObservations, occupancy_prediction, entry_exit_t, SituationPolygons, \
    Polygon, PlayerObservations, l_w_from_rectangle, intentions_prediction
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from dg_commons.sim.models.vehicle import VehicleParameters
from dg_commons import PlayerName, X
from dg_commons_dev.utils import BaseParams
from shapely.geometry.base import BaseGeometry
from dg_commons.sim.scenarios.structures import StaticObstacle
from shapely.geometry import LineString
from dg_commons.maps.lanes import DgLanelet
from geometry import translation_angle_from_SE2


@dataclass
class ReplanDescription:
    """ Parameters describing an emergency """

    is_replan: bool = False

    obstacles: List[StaticObstacle] = field(default_factory=list)
    """ Obstacles in the scene """

    obs_id: List[StaticObstacle] = field(default_factory=list)
    """ Obstacles intersected bstacle that caused the replan """

    entry_along_lane: List[float] = field(default_factory=list)
    """ Entry places """

    exit_along_lane: List[float] = field(default_factory=list)
    """ Exit places """

    my_player: PlayerName = None
    """ My PlayerName """
    other_player: PlayerName = None
    """ Other Playername """

    def __post_init__(self):
        if self.is_replan:
            assert self.obstacles
            assert self.obs_id


@dataclass
class ReplanParams(BaseParams):
    min_replanning_distance: float = 5.0
    """ Min distance for replanning to happen """


class Replan(Situation[SituationObservations, ReplanDescription]):
    """
    Replan situation, provides tools for:
     1) establishing whether an emergency is occurring
     2) computing important parameters describing the emergency situation
    """
    REF_PARAMS: dataclass = ReplanParams

    def __init__(self, params: ReplanParams, safety_time_braking: float,
                 vehicle_params: VehicleParameters = VehicleParameters.default_car(),
                 plot: bool = False):
        self.params = params
        self.safety_time_braking = safety_time_braking
        self.acc_limits: Tuple[float, float] = vehicle_params.acc_limits

        self.obs: Optional[SituationObservations] = None
        self.replan_situation: ReplanDescription = ReplanDescription()
        self.polygon_plotter = SituationPolygons(plot=plot)
        self.counter = 0

    def update_observations(self, new_obs: SituationObservations) \
            -> Tuple[List[Polygon], List[SituationPolygons.PolygonClass]]:
        """
        Use new SituationObservations to update the situation:
        1) Establish whether a replan is required
        2) Compute its parameters
        @param new_obs: Current SituationObervations
        @return: Polygons and polygon classes for plotting purposes
        """
        self.obs = new_obs
        my_name: PlayerName = new_obs.my_name
        agents: MutableMapping[PlayerName, PlayerObservations] = new_obs.agents
        path: DgLanelet = self.obs.planned_path
        beta, q = path.find_along_lane_closest_point(np.array([agents[my_name].state.x,
                                                               agents[my_name].state.y]), 10e-3)
        along_lane: float = path.along_lane_from_beta(beta)
        polygon, polygons = intentions_prediction(self._get_replanning_dist(agents[my_name].state.vx),
                                                  path, along_lane)
        other_name: PlayerName = PlayerName("StaticObs")

        obstacles = []
        for obs in new_obs.static_obstacles:
            if (not isinstance(obs.shape, LineString)) and polygon.intersects(obs.shape):
                obstacles.append(obs)

        entry_points = []
        exit_points = []
        entered: bool = False

        if obstacles:
            for (along_lane, poly) in polygons.items():

                intersects: bool = False
                for j, obs in enumerate(obstacles):
                    intersects_obs = poly.intersects(obs.shape)
                    intersects = intersects_obs or intersects

                if intersects and not entered:
                    entered = True
                    entry_points.append(along_lane)
                elif not intersects and entered:
                    exit_points.append(along_lane)
                    entered = False

        if len(entry_points) != len(exit_points):
            exit_points.append(along_lane)
        assert len(entry_points) == len(exit_points)

        if entry_points:
            self.replan_situation = ReplanDescription(True, other_player=other_name, obstacles=new_obs.static_obstacles,
                                                      obs_id=obstacles, entry_along_lane=entry_points,
                                                      exit_along_lane=exit_points)
        else:
            self.replan_situation = ReplanDescription(False)

        return self.polygon_plotter.next_frame()

    def _get_replanning_dist(self, vel: float) -> float:
        """
        The minimal distance to keep between two vehicles.
        @param vel: Current velocity
        @return: Distance
        """
        return vel * self.safety_time_braking + self.params.min_replanning_distance

    def is_true(self) -> bool:
        """
        Whether an emergency situation is occurring
        @return: True if it is occurring, False otherwise
        """
        assert self.obs is not None
        return self.replan_situation.is_replan

    def infos(self) -> ReplanDescription:
        """
        @return: Emergency Description
        """
        assert self.obs is not None
        return self.replan_situation

    def simulation_ended(self) -> None:
        """ Called when the simulation ends """
        pass

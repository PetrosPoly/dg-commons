from dg_commons_dev.behavior.behavior_types import Situation
from dataclasses import dataclass
from typing import Union, List, Tuple, MutableMapping, Optional
from dg_commons_dev.behavior.utils import SituationObservations, \
    occupancy_prediction, entry_exit_t, SituationPolygons, Polygon, PlayerObservations
from dg_commons.sim.models import kmh2ms, extract_vel_from_state
from dg_commons.sim.models.vehicle import VehicleParameters
from dg_commons import PlayerName, X
from dg_commons_dev.utils import BaseParams
from shapely.geometry import LineString


@dataclass
class EmergencyDescription:
    """ Parameters describing an emergency """

    is_emergency: bool = False

    drac: Tuple[float, float] = None
    """ deceleration to avoid a crash """
    ttc: float = None
    """ time-to-collision """
    pet: float = None
    """ post encroachment time """
    """ Reference: https://sumo.dlr.de/docs/Simulation/Output/SSM_Device.html """

    my_player: PlayerName = None
    """ My PlayerName """
    other_player: PlayerName = None
    """ Other Playername """

    def __post_init__(self):
        if self.is_emergency:
            assert self.ttc is not None
            assert self.drac is not None
            assert self.pet is not None


@dataclass
class EmergencyParams(BaseParams):
    min_dist: float = 7
    """Evaluate emergency only for vehicles within x [m]"""
    min_vel: float = kmh2ms(5)
    """emergency only to vehicles that are at least moving at.."""
    def __post_init__(self):
        assert 0 <= self.min_vel <= 350
        assert 0 <= self.min_dist <= 100


class Emergency(Situation[SituationObservations, EmergencyDescription]):
    """
    Emergency situation, provides tools for:
     1) establishing whether an emergency is occurring
     2) computing important parameters describing the emergency situation
    """
    REF_PARAMS: dataclass = EmergencyParams

    def __init__(self, params: EmergencyParams, safety_time_braking: float,
                 vehicle_params: VehicleParameters = VehicleParameters.default_car(),
                 plot: bool = False):
        self.params = params
        self.safety_time_braking = safety_time_braking
        self.acc_limits: Tuple[float, float] = vehicle_params.acc_limits

        self.obs: Optional[SituationObservations] = None
        self.emergency_situation: EmergencyDescription = EmergencyDescription()
        self.polygon_plotter = SituationPolygons(plot=plot)
        self.counter = 0

    def update_observations(self, new_obs: SituationObservations) \
            -> Tuple[List[Polygon], List[SituationPolygons.PolygonClass]]:
        """
        Use new SituationObservations to update the situation:
        1) Establish whether an emergency is occurring
        2) Compute its parameters
        @param new_obs: Current SituationObervations
        @return: Polygons and polygon classes for plotting purposes
        """
        self.obs = new_obs
        my_name: PlayerName = new_obs.my_name
        agents: MutableMapping[PlayerName, PlayerObservations] = new_obs.agents

        my_state: X = agents[my_name].state
        my_vel: float = my_state.vx
        my_occupancy: Polygon = agents[my_name].occupancy
        my_polygon, _ = occupancy_prediction(agents[my_name].state, self.safety_time_braking)

        # TODO: fix duplicated code
        for other_name, _ in agents.items():
            if other_name == my_name:
                continue
            other_state: X = agents[other_name].state
            other_vel: float = extract_vel_from_state(other_state)
            other_occupancy: Polygon = agents[other_name].occupancy
            other_polygon, _ = occupancy_prediction(agents[other_name].state, self.safety_time_braking)

            intersection: Polygon = my_polygon.intersection(other_polygon)
            if intersection.is_empty:
                self.emergency_situation = EmergencyDescription(False)
            else:
                my_entry_time, my_exit_time = entry_exit_t(intersection, my_state, my_occupancy,
                                                           self.safety_time_braking, my_vel, tol=0.01)
                other_entry_time, other_exit_time = entry_exit_t(intersection, other_state, other_occupancy,
                                                                 self.safety_time_braking, other_vel, tol=0.01)

                collision_score: float = 0
                collision_max: float = 1  # if collision_score > collision_max then there is an emergency

                def pet_score(pet: float):
                    pet_min = self.safety_time_braking*2
                    if pet < pet_min:
                        return 1.0
                    else:
                        return 0.0

                def ttc_score(ttc: float):
                    ttc_min = self.safety_time_braking*2
                    if ttc < ttc_min:
                        return 1.0
                    else:
                        return 0.0

                def drac_score(drac: float):
                    drac_max = self.acc_limits[1]
                    if drac_max < drac:
                        return 1.0
                    else:
                        return 0.0

                pet: float = other_entry_time - my_exit_time if my_exit_time < other_exit_time else \
                    my_entry_time - other_exit_time
                self.emergency_situation.my_player = my_name
                self.emergency_situation.pet = pet
                collision_score += pet_score(pet)

                pot1, pot2 = my_exit_time - other_entry_time > 0, other_exit_time - my_entry_time > 0
                if pot1 or pot2:
                    ttc = other_entry_time if my_entry_time < other_entry_time else my_entry_time
                    self.emergency_situation.ttc = ttc
                    collision_score += ttc_score(ttc)
                    drac1: float = 2 * (other_vel - other_vel * other_entry_time / my_exit_time) / my_exit_time \
                        if pot1 else 0.0
                    drac2: float = 2 * (my_vel - my_vel * my_entry_time / other_exit_time) / other_exit_time \
                        if pot2 else 0.0
                    drac: float = max(drac1, drac2)
                    collision_score += drac_score(drac)
                    self.emergency_situation.drac = [drac1, drac2]
                    if collision_max < collision_score:
                        self.emergency_situation.is_emergency = True
                        self.emergency_situation.other_player = other_name

                        other_occupancy, _ = occupancy_prediction(agents[other_name].state, 0.1)
                        my_occupancy, _ = occupancy_prediction(agents[my_name].state, 0.1)
                        self.polygon_plotter.plot_polygon(my_occupancy,
                                                          SituationPolygons.PolygonClass(collision=True))
                        self.polygon_plotter.plot_polygon(other_occupancy,
                                                          SituationPolygons.PolygonClass(collision=True))

        for obs in new_obs.static_obstacles:
            if isinstance(obs.shape, LineString):
                continue

            other_vel: float = 0
            other_polygon: Polygon = obs.shape

            intersection: Polygon = my_polygon.intersection(other_polygon)
            if intersection.is_empty:
                self.emergency_situation = EmergencyDescription(False)
            else:
                my_entry_time, my_exit_time = entry_exit_t(intersection, my_state, my_occupancy,
                                                           self.safety_time_braking, my_vel, tol=0.01)
                other_entry_time, other_exit_time = 0, 10e6

                collision_score: float = 0
                collision_max: float = 1  # if collision_score > collision_max then there is an emergency

                def pet_score(pet: float):
                    pet_min = self.safety_time_braking
                    if pet < pet_min:
                        return 1.0
                    else:
                        return 0.0

                def ttc_score(ttc: float):
                    ttc_min = self.safety_time_braking
                    if ttc < ttc_min:
                        return 1.0
                    else:
                        return 0.0

                def drac_score(drac: float):
                    drac_max = self.acc_limits[1]
                    if drac_max < drac:
                        return 1.0
                    else:
                        return 0.0

                pet: float = other_entry_time - my_exit_time if my_exit_time < other_exit_time else \
                    my_entry_time - other_exit_time
                self.emergency_situation.my_player = my_name
                self.emergency_situation.pet = pet
                collision_score += pet_score(pet)

                pot1, pot2 = my_exit_time - other_entry_time > 0, other_exit_time - my_entry_time > 0
                if pot1 or pot2:
                    ttc = other_entry_time if my_entry_time < other_entry_time else my_entry_time
                    self.emergency_situation.ttc = ttc
                    collision_score += ttc_score(ttc)
                    drac1: float = 2 * (other_vel - other_vel * other_entry_time / my_exit_time) / my_exit_time \
                        if pot1 else 0.0
                    drac2: float = 2 * (my_vel - my_vel * my_entry_time / other_exit_time) / other_exit_time \
                        if pot2 else 0.0
                    drac: float = max(drac1, drac2)
                    collision_score += drac_score(drac)
                    self.emergency_situation.drac = [drac1, drac2]
                    if collision_max < collision_score:
                        self.emergency_situation.is_emergency = True
                        self.emergency_situation.other_player = other_name

                        other_occupancy, _ = occupancy_prediction(agents[other_name].state, 0.1)
                        my_occupancy, _ = occupancy_prediction(agents[my_name].state, 0.1)
                        self.polygon_plotter.plot_polygon(my_occupancy,
                                                          SituationPolygons.PolygonClass(collision=True))
                        self.polygon_plotter.plot_polygon(other_occupancy,
                                                          SituationPolygons.PolygonClass(collision=True))

        return self.polygon_plotter.next_frame()
        # This is for plotting purposes, can be ignored

    def is_true(self) -> bool:
        """
        Whether an emergency situation is occurring
        @return: True if it is occurring, False otherwise
        """
        assert self.obs is not None
        return self.emergency_situation.is_emergency

    def infos(self) -> EmergencyDescription:
        """
        @return: Emergency Description
        """
        assert self.obs is not None
        return self.emergency_situation

    def simulation_ended(self) -> None:
        """ Called when the simulation ends """
        pass

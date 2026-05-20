"""Bus pull-off parking: main road with shoulder bays (not a symmetric parking lot)."""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd
from gymnasium.envs.registration import register

from highway_env.envs.common.action import action_factory
from highway_env.envs.common.observation import KinematicsGoalObservation
from highway_env.envs.parking_env import ParkingEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle
from highway_env.vehicle.objects import Landmark, Obstacle


class BusVehicle(Vehicle):
    """Longer, slower vehicle approximating a transit bus."""

    LENGTH = 12.0
    WIDTH = 2.6
    MAX_SPEED = 12.0
    MIN_SPEED = -4.0


GOAL_FEATURES = ["x", "y", "vx", "vy", "cos_h", "sin_h"]


class KinematicsGoalTrafficObservation(KinematicsGoalObservation):
    """
    HER-compatible dict obs: ego + goal (6) for achieved/desired goals, plus
    flattened ego-relative kinematics of nearby vehicles in ``observation``.
    """

    def __init__(
        self,
        env,
        scales: list[float],
        observe_vehicles_count: int = 4,
        see_behind: bool = True,
        **kwargs,
    ) -> None:
        self.observe_vehicles_count = observe_vehicles_count
        kwargs.pop("type", None)
        features = kwargs.pop("features", GOAL_FEATURES)
        super().__init__(
            env,
            scales=scales,
            features=features,
            see_behind=see_behind,
            vehicles_count=1,
            **kwargs,
        )

    def _nearby_traffic_flat(self) -> np.ndarray:
        n_other = max(0, self.observe_vehicles_count - 1)
        size = n_other * len(self.features)
        if not self.env.road or not self.observer_vehicle or n_other == 0:
            return np.zeros(size, dtype=np.float64)

        close = self.env.road.close_objects_to(
            self.observer_vehicle,
            self.env.PERCEPTION_DISTANCE,
            count=n_other,
            see_behind=self.see_behind,
            sort=True,
            vehicles_only=True,
        )
        origin = self.observer_vehicle
        rows = [
            v.to_dict(origin, observe_intentions=False)
            for v in close[-n_other:]
        ]
        while len(rows) < n_other:
            rows.append(dict.fromkeys(self.features, 0.0))
        flat = np.ravel(pd.DataFrame(rows)[self.features].values).astype(np.float64)
        return flat / np.tile(self.scales, n_other)

    def observe(self) -> OrderedDict:
        if not self.observer_vehicle:
            n_other = max(0, self.observe_vehicles_count - 1)
            empty = np.zeros(len(self.features), dtype=np.float64)
            traffic = np.zeros(n_other * len(self.features), dtype=np.float64)
            return OrderedDict(
                [
                    ("observation", np.concatenate([empty, traffic])),
                    ("achieved_goal", empty.copy()),
                    ("desired_goal", empty.copy()),
                ]
            )

        ego = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.to_dict()])[self.features]
        )
        goal = np.ravel(
            pd.DataFrame.from_records([self.observer_vehicle.goal.to_dict()])[
                self.features
            ]
        )
        ego_scaled = ego / self.scales
        goal_scaled = goal / self.scales
        traffic = self._nearby_traffic_flat()
        return OrderedDict(
            [
                ("observation", np.concatenate([ego_scaled, traffic])),
                ("achieved_goal", ego_scaled),
                ("desired_goal", goal_scaled),
            ]
        )


class BusParkingEnv(ParkingEnv):
    """
    Bus stop style task on a road with curb-side pull-off bays.

    The ego travels on a main lane and must pull into a shoulder bay ahead.
    Other bays may hold parked vehicles. This replaces the old symmetric
    parking-lot layout (rows of slots on both sides of a center aisle).
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "observation": {
                    "type": "KinematicsGoalTraffic",
                    "features": GOAL_FEATURES,
                    "scales": [100, 100, 5, 5, 1, 1],
                    "normalize": False,
                    "observe_vehicles_count": 4,
                    "see_behind": True,
                },
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": (-4.0, 3.5),
                    "steering_range": (-np.deg2rad(40), np.deg2rad(40)),
                    "speed_range": (BusVehicle.MIN_SPEED, BusVehicle.MAX_SPEED),
                },
                "reward_weights": [1.0, 0.25, 0.15, 0.15, 0.05, 0.05],
                "success_goal_reward": 0.18,
                "collision_reward": -5,
                "time_penalty": 0.02,
                "duration": 180,
                "vehicles_count": 2,
                "spawn_speed": (2.0, 5.0),
                "spawn_distance": (25.0, 45.0),
                "add_walls": False,
                # Road + pull-off geometry
                "road_length": 140.0,
                "lane_width": 4.0,
                "pull_off_width": 6.0,
                "pull_off_length": 14.0,
                "pull_off_count": 4,
                "pull_off_spacing": 28.0,
                "first_pull_off_x": 50.0,
                "shoulder_y": -5.0,
                "randomize_spot": True,
                "goal_longitudinal": 78.0,
            }
        )
        return config

    def _pull_off_lane_indices(self) -> list[tuple[str, str, int]]:
        n = self.config["pull_off_count"]
        return [("pull", str(k), 0) for k in range(n)]

    def _select_goal_lane_index(self) -> tuple[str, str, int]:
        lanes = self._pull_off_lane_indices()
        if self.config["randomize_spot"]:
            return lanes[int(self.np_random.integers(len(lanes)))]
        goal_x = float(self.config["goal_longitudinal"])
        cfg = self.config
        k = int(round((goal_x - cfg["first_pull_off_x"]) / cfg["pull_off_spacing"]))
        k = int(np.clip(k, 0, cfg["pull_off_count"] - 1))
        return lanes[k]

    def _create_road(self, spots: int | None = None) -> None:
        del spots  # unused; kept for ParkingEnv API compatibility
        cfg = self.config
        length = cfg["road_length"]
        lane_w = cfg["lane_width"]
        po_w = cfg["pull_off_width"]
        po_len = cfg["pull_off_length"]
        y_off = cfg["shoulder_y"]
        striped = LineType.STRIPED
        continuous = LineType.CONTINUOUS

        net = RoadNetwork()
        net.add_lane(
            "road",
            "main",
            StraightLane(
                [0.0, 0.0],
                [length, 0.0],
                width=lane_w,
                line_types=(striped, continuous),
            ),
        )

        x0 = cfg["first_pull_off_x"]
        spacing = cfg["pull_off_spacing"]
        for k in range(cfg["pull_off_count"]):
            x_center = x0 + k * spacing
            net.add_lane(
                "pull",
                str(k),
                StraightLane(
                    [x_center - po_len / 2, y_off],
                    [x_center + po_len / 2, y_off],
                    width=po_w,
                    line_types=(continuous, continuous),
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        cfg = self.config
        pull_lanes = self._pull_off_lane_indices()
        goal_lane_index = self._select_goal_lane_index()
        goal_lane = self.road.network.get_lane(goal_lane_index)
        goal_x = float(goal_lane.position(goal_lane.length / 2, 0)[0])

        self.controlled_vehicles = []
        lo, hi = cfg["spawn_speed"]
        dist_lo, dist_hi = cfg["spawn_distance"]
        spawn_x = max(2.0, goal_x - self.np_random.uniform(dist_lo, dist_hi))
        heading = self.np_random.uniform(-0.1, 0.1)
        speed = self.np_random.uniform(lo, hi)
        ego = BusVehicle(self.road, [spawn_x, 0.0], heading, speed)
        ego.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        ego.goal = Landmark(
            self.road,
            goal_lane.position(goal_lane.length / 2, 0),
            heading=goal_lane.heading,
        )
        self.road.objects.append(ego.goal)

        other_lanes = [idx for idx in pull_lanes if idx != goal_lane_index]
        self.np_random.shuffle(other_lanes)
        for lane_index in other_lanes[: cfg["vehicles_count"]]:
            v = Vehicle.make_on_lane(
                self.road, lane_index, longitudinal=goal_lane.length / 2, speed=0.0
            )
            self.road.vehicles.append(v)

        if cfg["add_walls"]:
            width, height = cfg["road_length"] + 20, 40
            for y in (-height / 2, height / 2):
                obstacle = Obstacle(self.road, [cfg["road_length"] / 2, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def define_spaces(self) -> None:
        """Use traffic-aware HER observations instead of goal-only kinematics."""
        obs_cfg = dict(self.config["observation"])
        obs_cfg.pop("type", None)
        self.observation_type = KinematicsGoalTrafficObservation(self, **obs_cfg)
        self.observation_type_parking = self.observation_type
        self.observation_space = self.observation_type.space()
        self.action_type = action_factory(self, self.config["action"])
        self.action_space = self.action_type.space()

    def _reward(self, action: np.ndarray) -> float:
        reward = super()._reward(action)
        reward -= self.config["time_penalty"]
        return reward


register(
    id="bus-parking-v0",
    entry_point="bus_parking_env:BusParkingEnv",
)

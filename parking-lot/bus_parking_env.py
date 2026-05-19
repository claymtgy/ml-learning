"""Side-of-road bus stop parking on top of highway-env's ParkingEnv."""

from __future__ import annotations

import numpy as np
from gymnasium.envs.registration import register

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


class BusParkingEnv(ParkingEnv):
    """
    Parallel parking into bays beside a travel lane — bus stop style.

    The ego starts on the center lane and must pull into a random side bay
    with the correct heading, similar to a bus docking at a curb stop.
    """

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update(
            {
                "action": {
                    "type": "ContinuousAction",
                    "acceleration_range": (-4.0, 3.5),
                    "steering_range": (-np.deg2rad(40), np.deg2rad(40)),
                    "speed_range": (BusVehicle.MIN_SPEED, BusVehicle.MAX_SPEED),
                },
                # x, y, vx, vy, cos_h, sin_h — emphasize pose over speed at the bay
                "reward_weights": [1.0, 0.25, 0.15, 0.15, 0.05, 0.05],
                "success_goal_reward": 0.18,
                "collision_reward": -5,
                "time_penalty": 0.02,
                "duration": 150,
                "vehicles_count": 2,
                "spawn_speed": (2.0, 5.0),
                "parking_spots": 10,
                "bay_width": 6.0,
                "bay_offset": 12.0,
                "bay_length": 14.0,
            }
        )
        return config

    def _create_road(self, spots: int | None = None) -> None:
        spots = spots or self.config["parking_spots"]
        net = RoadNetwork()
        width = self.config["bay_width"]
        lt = (LineType.CONTINUOUS, LineType.CONTINUOUS)
        x_offset = 0
        y_offset = self.config["bay_offset"]
        length = self.config["bay_length"]

        for k in range(spots):
            x = (k + 1 - spots // 2) * (width + x_offset) - width / 2
            net.add_lane(
                "a",
                "b",
                StraightLane(
                    [x, y_offset], [x, y_offset + length], width=width, line_types=lt
                ),
            )
            net.add_lane(
                "b",
                "c",
                StraightLane(
                    [x, -y_offset],
                    [x, -y_offset - length],
                    width=width,
                    line_types=lt,
                ),
            )

        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=self.config["show_trajectories"],
        )

    def _create_vehicles(self) -> None:
        empty_spots = list(self.road.network.lanes_dict().keys())

        self.controlled_vehicles = []
        lo, hi = self.config["spawn_speed"]
        for i in range(self.config["controlled_vehicles"]):
            x0 = float(i - self.config["controlled_vehicles"] // 2) * 14.0
            # Face along the travel lane with a bit of roll — not a random spin.
            heading = self.np_random.uniform(-0.25, 0.25)
            speed = self.np_random.uniform(lo, hi)
            vehicle = BusVehicle(self.road, [x0, 0.0], heading, speed)
            vehicle.color = VehicleGraphics.EGO_COLOR
            self.road.vehicles.append(vehicle)
            self.controlled_vehicles.append(vehicle)
            empty_spots.remove(vehicle.lane_index)

        for vehicle in self.controlled_vehicles:
            lane_index = empty_spots[
                self.np_random.choice(np.arange(len(empty_spots)))
            ]
            lane = self.road.network.get_lane(lane_index)
            vehicle.goal = Landmark(
                self.road, lane.position(lane.length / 2, 0), heading=lane.heading
            )
            self.road.objects.append(vehicle.goal)
            empty_spots.remove(lane_index)

        for _ in range(self.config["vehicles_count"]):
            if not empty_spots:
                break
            lane_index = empty_spots[
                self.np_random.choice(np.arange(len(empty_spots)))
            ]
            v = Vehicle.make_on_lane(
                self.road, lane_index, longitudinal=4.0, speed=0.0
            )
            self.road.vehicles.append(v)
            empty_spots.remove(lane_index)

        if self.config["add_walls"]:
            width, height = 80, 48
            for y in [-height / 2, height / 2]:
                obstacle = Obstacle(self.road, [0, y])
                obstacle.LENGTH, obstacle.WIDTH = (width, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)
            for x in [-width / 2, width / 2]:
                obstacle = Obstacle(self.road, [x, 0], heading=np.pi / 2)
                obstacle.LENGTH, obstacle.WIDTH = (height, 1)
                obstacle.diagonal = np.sqrt(obstacle.LENGTH**2 + obstacle.WIDTH**2)
                self.road.objects.append(obstacle)

    def _reward(self, action: np.ndarray) -> float:
        reward = super()._reward(action)
        reward -= self.config["time_penalty"]
        return reward


register(
    id="bus-parking-v0",
    entry_point="bus_parking_env:BusParkingEnv",
)

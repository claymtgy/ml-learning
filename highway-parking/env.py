import numpy as np
from highway_env.envs.parking_env import ParkingEnv
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.objects import Landmark, Obstacle


class CurbsideParkingEnv(ParkingEnv):
    @classmethod
    def default_config(cls):
        cfg = super().default_config()
        cfg.update({
            "reward_weights": [1.0, 1.0, 0.2, 0.2, 0.5, 0.5],
            "success_goal_reward": 0.10,
            "collision_reward": -5,
            "duration": 120,
            "road_length": 120.0,
            "lane_width": 4.0,
            "spot_length": 11.0,
            "goal_longitudinal": 40.0,
            "randomize_spot": False,
        })
        return cfg

    def _create_road(self):
        cfg = self.config
        L = cfg["road_length"]
        w = cfg["lane_width"]
        c, s = LineType.CONTINUOUS, LineType.STRIPED

        net = RoadNetwork()
        net.add_lane("a", "b",
                     StraightLane([0, 0], [L, 0], width=w, line_types=(s, s)))
        net.add_lane("a", "b",
                     StraightLane([0, 0], [L, 0], width=w, line_types=(c, s)))
        self.road = Road(
            network=net,
            np_random=self.np_random,
            record_history=cfg["show_trajectories"],
        )

    def _create_vehicles(self):
        cfg = self.config
        w = cfg["lane_width"]
        L = cfg["road_length"]

        goal_x = (
            self.np_random.uniform(15.0, L - 15.0)
            if cfg["randomize_spot"] else cfg["goal_longitudinal"]
        )
        goal_y = -w

        self.controlled_vehicles = []
        spawn_x = max(2.0, goal_x - self.np_random.uniform(25.0, 45.0))
        spawn_heading = self.np_random.uniform(-0.1, 0.1)
        ego = self.action_type.vehicle_class(
            self.road, [spawn_x, 0.0], spawn_heading, 0.0
        )
        ego.LENGTH = 12
        ego.WIDTH = 2.55
        ego.diagonal = np.sqrt(ego.LENGTH ** 2 + ego.WIDTH ** 2)

        ego.color = VehicleGraphics.EGO_COLOR
        self.road.vehicles.append(ego)
        self.controlled_vehicles.append(ego)

        ego.goal = Landmark(self.road, [goal_x, goal_y], heading=0.0)
        self.road.objects.append(ego.goal)

        spot_len = cfg["spot_length"]
        for dx in (-spot_len, +spot_len):
            obs = Obstacle(self.road, [goal_x + dx, goal_y], heading=0.0)
            obs.LENGTH, obs.WIDTH = 5.0, 2.0
            obs.diagonal = np.sqrt(obs.LENGTH ** 2 + obs.WIDTH ** 2)
            self.road.objects.append(obs)

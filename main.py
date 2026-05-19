import gymnasium as gym
import numpy as np


def wrap_angle(angle: float):
    """wrap angle to [-pi, pi]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi


class BusParkingEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )

        # Bus params
        self.dt = 0.1
        self.max_speed = 2.0
        self.max_steps = 300

        self.accel_gain = 1.5
        self.turn_gain = 1.2

        # world space size
        self.x_min, self.x_max = -5.0, 15.0
        self.y_min, self.y_max = -5.0, 15.0

        # success params and tolerances
        self.goal_pos = np.array([10.0, 0.0], dtype=np.float32)
        self.goal_heading = 0.0
        self.pos_tolerance = 0.25
        self.heading_tolerance = np.deg2rad(10.0)
        self.speed_tolerance = 0.1

        # state of bus
        self.pos = np.zeroes(2, dtype=np.float32)
        self.heading = 0.0
        self.speed = 0.0
        self.step_count = 0

    def reset(self, seed=None):
        # Start bus some fixed distance from stop
        self.pos = np.array(
            [
                self.np_random.uniform(0.0, 3.0),
                self.np_random.uniform(7.0, 11.0)
            ], dtype=np.float32)

        self.heading = self.np_random.uniform(-0.2, 0.2)
        self.speed = 0.0
        self.step_count = 0

        obs = self._get_obs
        info = {}

        return obs, info

    def step(self, action):
        self.step_count += 1

        action = np.clip(action, self.action_space_low, self.action_space_high)
        steering, throttle = float(action[0]), float(action[1])

        # Simple kinematics
        self.speed += throttle * self.accel_gain * self.dt
        self.speed = float(
            np.clip(self.speed, -self.max_speed, self.max_speed))

        self.heading = wrap_angle(
            self.heading + steering * self.turn_gain * self.speed * self.dt)

        obs = self._get_obs()
        reward = self.compute_reward()
        terminated = self._is_done()
        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # return your 6 element state vector
        pass

    def compute_reward(self):
        # negative distance to goal for nw
        pass

    def _is_done(self):
        # Success: within 5cm. Fail: Timeout or collisoion
        pass

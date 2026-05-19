from stable_baselines3 import PPO, SAC, HerReplayBuffer
from gymnasium.wrappers import RecordVideo
import gymnasium
import register

ENV_ID = "curb-parking-v0"
ENV_CONFIG = {"reward_weights": [.5, .5, 0, 0, 0.8, 0.8],
              "success_goal_reward": 0.2,
              "collision_reward": -10}
N_ENVS = 16  # tune to your CPU; start with min(8, os.cpu_count())


def make_env(render_mode=None):
    # Note: no render_mode here — rendering inside subprocesses is wasteful
    return gymnasium.make(ENV_ID, render_mode=render_mode, config=ENV_CONFIG)


eval_env = RecordVideo(
    make_env(render_mode='rgb_array'),
    video_folder="parking_videos",
    episode_trigger=lambda e: True,
)

model = SAC.load("highway_parking_sac/model", env=eval_env)
try:
    for _ in range(10):
        done = truncated = False
        obs, info = eval_env.reset()
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
finally:
    eval_env.close()

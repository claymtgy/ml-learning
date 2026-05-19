import os
import gymnasium
from gymnasium.wrappers import RecordVideo
import register
import highway_env  # noqa: F401 — needed for the 'parking-v0' registration
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "curb-parking-v0"
ENV_CONFIG = {"reward_weights": [1.5, 1.5, 0.0, 0.0, 0.3, 0.3],
              "success_goal_reward": 0.1,
              "collision_reward": -10,
              "offscreen_rendering": True
              }
N_ENVS = 16  # tune to your CPU; start with min(8, os.cpu_count())


def make_env(render_mode=None):
    # Note: no render_mode here — rendering inside subprocesses is wasteful
    return gymnasium.make(ENV_ID, render_mode=render_mode, config=ENV_CONFIG)


def main():
    env = make_env()

    # model = SAC(
    #     "MultiInputPolicy",
    #     env,
    #     replay_buffer_class=HerReplayBuffer,
    #     replay_buffer_kwargs=dict(
    #         n_sampled_goal=4,
    #         goal_selection_strategy="future"
    #     ),
    #     learning_rate=3e-4,
    #     buffer_size=int(1e6),
    #     learning_starts=1000,
    #     batch_size=256,
    #     gamma=0.95,
    #     tau=0.05,
    #     train_freq=(64, "step"),
    #     gradient_steps=64,
    #     ent_coef="auto",
    #     policy_kwargs=dict(net_arch=[256, 256]),
    #     device="cpu",
    #     verbose=1,
    #     tensorboard_log="highway_parking_sac_her"
    # )

    model = SAC.load("highway_parking_sac/model", env=env)
    try:
        model.learn(total_timesteps=100_000)  # bump this up — see notes below
    except KeyboardInterrupt:
        model.save("highway_parking_sac/model")
    # vec_env.close()
    env.close()

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


if __name__ == "__main__":
    main()

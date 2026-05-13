import os
import gymnasium
from gymnasium.wrappers import RecordVideo
import highway_env  # noqa: F401 — needed for the 'parking-v0' registration
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "parking-v0"
ENV_CONFIG = {"reward_weights": [1, 0.2, 0.1, 0.1, 0.03, 0.03]}
N_ENVS = 16  # tune to your CPU; start with min(8, os.cpu_count())


def make_env(render_mode=None):
    # Note: no render_mode here — rendering inside subprocesses is wasteful
    return gymnasium.make(ENV_ID, render_mode=render_mode, config=ENV_CONFIG)


def main():
    # vec_env = make_vec_env(
    #     make_env,
    #     n_envs=N_ENVS,
    #     vec_env_cls=SubprocVecEnv,
    #     vec_env_kwargs={"start_method": "spawn"},  # safest on Linux too
    # )

    # model = PPO(
    #     "MultiInputPolicy",
    #     vec_env,
    #     policy_kwargs=dict(net_arch=[256, 256]),
    #     n_steps=2048,            # per env -> total rollout = N_ENVS * 2048
    #     n_epochs=10,
    #     learning_rate=5e-5,
    #     batch_size=256,          # must divide N_ENVS * n_steps
    #     gamma=0.99,
    #     verbose=1,
    #     tensorboard_log="parking_ppo/",
    # )

    env = make_env()

    model = SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future"
        ),
        learning_rate=1e-3,
        buffer_size=int(1e6),
        learning_starts=1000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=(64, "step"),
        gradient_steps=64,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",
        verbose=1,
        tensorboard_log="parking_sac_her"
    )

    model = SAC.load("parking_sac/model", env=env)
    model.learn(total_timesteps=100_000)  # bump this up — see notes below
    model.save("parking_sac/model")
    # vec_env.close()
    env.close()

    # --- evaluation / video recording: use a single env, not the SubprocVecEnv ---
    # eval_env = gymnasium.make(
    #     ENV_ID, render_mode="rgb_array", config=ENV_CONFIG)
    # eval_env = RecordVideo(
    #     eval_env,
    #     video_folder="parking_videos/",
    #     episode_trigger=lambda e: True,
    # )

    eval_env = RecordVideo(
        make_env(render_mode='rgb_array'),
        video_folder="parking_videos",
        episode_trigger=lambda e: True,
    )

    model = SAC.load("parking_sac/model", env=eval_env)
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

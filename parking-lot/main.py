import argparse
import os

import numpy as np
import gymnasium
from gymnasium.wrappers import RecordVideo
import highway_env  # noqa: F401 — registers highway-env envs
import bus_parking_env  # noqa: F401 — registers bus-parking-v0

from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv

ENV_ID = "bus-parking-v0"
MODEL_PATH = "parking_sac/model"
VIDEO_DIR = "parking_videos"
TENSORBOARD_DIR = "parking_sac_her"
N_ENVS = 8

# Override bus_parking_env defaults here if needed
ENV_CONFIG = {
    "reward_weights": [1.0, 2.0, 0.15, 0.15, 0.05, 0.05],
    "time_penalty": 0.05,
    "collision_reward": -10,
    "pull_off_width": 5.0,
}


def make_env(render_mode=None):
    return gymnasium.make(ENV_ID, render_mode=render_mode, config=ENV_CONFIG)


def _her_buffer_ready(model) -> bool:
    return isinstance(model.replay_buffer, HerReplayBuffer) and bool(
        np.any(model.replay_buffer.ep_length > 0)
    )


def prepare_her_buffer(model, env, *, resume: bool) -> None:
    """
    HER cannot sample until a full episode is stored. SAC.load() does not restore
    the replay buffer, so defer gradients until new rollouts finish episodes.
    """
    if _her_buffer_ready(model):
        return

    duration = int(env.envs[0].unwrapped.config["duration"])
    # VecEnv advances all envs each step; need duration steps per env minimum.
    min_steps = duration * N_ENVS + 64
    if resume:
        model.learning_starts = model.num_timesteps + min_steps
    else:
        model.learning_starts = max(model.learning_starts, min_steps)

    print(
        "HER replay buffer is empty (normal after --resume); "
        f"collect-only until timestep {model.learning_starts} "
        f"(~{duration} steps/env for episodes to finish)."
    )


def build_model(env, *, load_path: str | None = None):
    if load_path:
        return SAC.load(load_path, env=env)

    return SAC(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        learning_rate=3e-4,
        buffer_size=int(1e6),
        learning_starts=2000,
        batch_size=256,
        gamma=0.95,
        tau=0.05,
        train_freq=(64, "step"),
        gradient_steps=64,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        device="cpu",
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
    )


def train(timesteps: int, resume: bool, fresh: bool):
    env = make_vec_env(make_env, n_envs=N_ENVS, vec_env_cls=DummyVecEnv)
    model = None
    try:
        if fresh and os.path.exists(f"{MODEL_PATH}.zip"):
            os.remove(f"{MODEL_PATH}.zip")
            print(f"Removed old checkpoint at {MODEL_PATH} (--fresh)")
        load_path = (
            MODEL_PATH if resume and os.path.exists(
                f"{MODEL_PATH}.zip") else None
        )
        model = build_model(env, load_path=load_path)

        if isinstance(model.replay_buffer, HerReplayBuffer):
            prepare_her_buffer(model, env, resume=bool(load_path))

        if resume and load_path:
            total_timesteps = model.num_timesteps + timesteps
            reset_num_timesteps = False
        else:
            total_timesteps = timesteps
            reset_num_timesteps = True

        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        if model is not None:
            model.save(MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}")
        env.close()


def evaluate(episodes: int):
    # HER checkpoints need an env at load time (for compute_reward), but not
    # the RecordVideo wrapper — that interferes with reset/render.
    load_env = make_env()
    model = SAC.load(MODEL_PATH, env=load_env)
    load_env.close()

    base = make_env(render_mode="rgb_array")
    base.reset()
    _ = base.render()  # init pygame viewer before RecordVideo

    eval_env = RecordVideo(
        base,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda e: True,
        name_prefix="bus_park",
    )
    try:
        for ep in range(episodes):
            done = truncated = False
            obs, info = eval_env.reset()
            steps = 0
            max_speed = 0.0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                action = np.asarray(action).reshape(-1)
                obs, reward, done, truncated, info = eval_env.step(action)
                steps += 1
                v = eval_env.unwrapped.vehicle
                max_speed = max(max_speed, abs(v.speed))
            success = info.get("is_success", False)
            print(
                f"episode {ep + 1}: steps={steps} max_speed={max_speed:.2f} "
                f"success={success} last_reward={reward:.3f}"
            )
    finally:
        eval_env.close()
    print(f"Wrote videos to {VIDEO_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a bus pull-off policy (road + shoulder bays)."
    )
    parser.add_argument(
        "mode",
        choices=["train", "eval"],
        help="train: SAC+HER learning; eval: record rollout videos",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="training timesteps (default: 500000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from parking_sac/model if it exists",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete old checkpoint and train from scratch (required after env layout changes)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="evaluation episodes to record (default: 5)",
    )
    args = parser.parse_args()

    if args.mode == "train":
        train(args.timesteps, args.resume, args.fresh)
    else:
        evaluate(args.episodes)


if __name__ == "__main__":
    main()

import argparse
import os
import sys
import time

import gymnasium
import torch
from gymnasium.wrappers import RecordVideo
import highway_env  # noqa: F401 — registers highway-v0, highway-fast-v0

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ENV_ID = "highway-v0"
FAST_ENV_ID = "highway-fast-v0"
MODEL_PATH = "highway_ppo/model"
VIDEO_DIR = "highway_videos"
TENSORBOARD_DIR = "highway_ppo"
N_ENVS = 8
N_STEPS = 512  # smaller rollouts → first PPO update sooner on CPU/WSL

# Reward: stay on road, avoid crashes, keep a reasonable speed.
ENV_CONFIG = {
    "lanes_count": 4,
    "vehicles_count": 30,
    "duration": 40,
    "collision_reward": -1.0,
    "high_speed_reward": 0.4,
    "right_lane_reward": 0.1,
    "normalize_reward": True,
    "offscreen_rendering": True,
}

FAST_ENV_CONFIG = {
    **ENV_CONFIG,
    "lanes_count": 3,
    "vehicles_count": 20,
    "duration": 30,
}


class RolloutProgressCallback(BaseCallback):
    """Print progress while PPO collects its first (slow) rollout on CPU."""

    def __init__(self, n_steps: int, n_envs: int, verbose: int = 0):
        super().__init__(verbose)
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.rollout_target = n_steps * n_envs
        self._rollout_start = None
        self._last_print = 0.0

    def _on_rollout_start(self) -> None:
        self._rollout_start = time.time()
        print(
            f"Collecting rollout ({self.rollout_target} env steps) — "
            "no SB3 table until this finishes...",
            flush=True,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_print >= 500:
            elapsed = time.time() - self._rollout_start
            print(
                f"  env steps: {self.num_timesteps} "
                f"({elapsed:.0f}s elapsed)",
                flush=True,
            )
            self._last_print = self.num_timesteps
        return True


def make_env(render_mode=None, *, fast: bool = False):
    env_id = FAST_ENV_ID if fast else ENV_ID
    config = FAST_ENV_CONFIG if fast else ENV_CONFIG
    return gymnasium.make(env_id, render_mode=render_mode, config=config)


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA requested but not available. In WSL, install PyTorch with CUDA:\n"
                "  pip install torch --index-url https://download.pytorch.org/whl/cu124"
            )
        return "cuda"
    # auto
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}", flush=True)
        return "cuda"
    print("CUDA not available — using CPU for the policy network.", flush=True)
    return "cpu"


def build_model(
    env,
    *,
    device: str,
    load_path: str | None = None,
    n_steps: int = N_STEPS,
):
    if load_path:
        return PPO.load(load_path, env=env, device=device)

    return PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.8,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[256, 256]),
        device=device,
        verbose=1,
        tensorboard_log=TENSORBOARD_DIR,
    )


def train(timesteps: int, resume: bool, fresh: bool, fast: bool, device: str):
    # SubprocVecEnv parallelizes on Linux/WSL; DummyVecEnv is safer on Windows.
    vec_cls = SubprocVecEnv if sys.platform != "win32" else DummyVecEnv
    env_fn = lambda: make_env(fast=fast)
    print(f"Creating {N_ENVS} envs ({FAST_ENV_ID if fast else ENV_ID})...", flush=True)
    env = make_vec_env(env_fn, n_envs=N_ENVS, vec_env_cls=vec_cls)
    print("Envs ready. Starting PPO.", flush=True)

    model = None
    try:
        if fresh and os.path.exists(f"{MODEL_PATH}.zip"):
            os.remove(f"{MODEL_PATH}.zip")
            print(f"Removed old checkpoint at {MODEL_PATH} (--fresh)")

        load_path = (
            MODEL_PATH if resume and os.path.exists(f"{MODEL_PATH}.zip") else None
        )
        model = build_model(env, device=device, load_path=load_path)

        if resume and load_path:
            total_timesteps = model.num_timesteps + timesteps
            reset_num_timesteps = False
        else:
            total_timesteps = timesteps
            reset_num_timesteps = True

        callback = RolloutProgressCallback(N_STEPS, N_ENVS)
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=reset_num_timesteps,
            callback=callback,
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        if model is not None:
            model.save(MODEL_PATH)
            print(f"Saved model to {MODEL_PATH}")
        env.close()


def evaluate(episodes: int, fast: bool, device: str):
    model = PPO.load(MODEL_PATH, device=device)

    base = make_env(render_mode="rgb_array", fast=fast)
    base.reset()
    _ = base.render()

    eval_env = RecordVideo(
        base,
        video_folder=VIDEO_DIR,
        episode_trigger=lambda e: True,
        name_prefix="highway",
    )
    try:
        for ep in range(episodes):
            done = truncated = False
            obs, info = eval_env.reset()
            steps = 0
            total_reward = 0.0
            while not (done or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = eval_env.step(action)
                steps += 1
                total_reward += reward

            crashed = eval_env.unwrapped.vehicle.crashed
            print(
                f"episode {ep + 1}: steps={steps} reward={total_reward:.2f} "
                f"crashed={crashed}"
            )
    finally:
        eval_env.close()
    print(f"Wrote videos to {VIDEO_DIR}/")


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate a PPO policy for highway driving."
    )
    parser.add_argument(
        "mode",
        choices=["train", "eval"],
        help="train: PPO learning; eval: record rollout videos",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=200_000,
        help="training timesteps (default: 200000)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue from highway_ppo/model if it exists",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete old checkpoint and train from scratch",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="use highway-fast-v0 (fewer vehicles, faster sim — recommended on WSL/CPU)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="policy network device (default: auto — uses GPU if available)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="evaluation episodes to record (default: 5)",
    )
    args = parser.parse_args()
    device = resolve_device(args.device)

    if args.mode == "train":
        train(args.timesteps, args.resume, args.fresh, args.fast, device)
    else:
        evaluate(args.episodes, args.fast, device)


if __name__ == "__main__":
    main()

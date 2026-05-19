import gymnasium as gym
from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode="human")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

obs, _ = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

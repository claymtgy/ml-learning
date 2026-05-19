from stable_baselines3 import SAC
from main import BusParkingEnv

env = BusParkingEnv
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=200_000)
model.save("bus_parking_v1")

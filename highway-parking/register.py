# highway-parking/register.py
import gymnasium
gymnasium.register(
    id="curb-parking-v0",
    entry_point="env:CurbsideParkingEnv",
)

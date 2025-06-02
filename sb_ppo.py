import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

import numpy as np

from ctrlab.envs.hoversb import Hover
from ctrlab.utils.quaternion import euler_to_quat
from ctrlab import crazyflie

mass = crazyflie['mass']
inertia = crazyflie['inertia']
G1 = crazyflie['G1']
tau = crazyflie['tau']

env = Hover(mass=mass, 
            inertia=inertia, 
            G1 = G1,
            tau = tau,
            dt = np.float32(0.01)
            )

checkpoint_callback = CheckpointCallback(
    save_freq=100_000,
    save_path="./checkpoints/",
    name_prefix="ppo_hover"
)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=2_000_000, callback=checkpoint_callback)
model.save("ppo_hover")
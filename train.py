import numpy as np

from ctrlab import crazyflie
from ctrlab.envs.hover import Hover
from ctrlab.rl.ppo import PPO

mass = crazyflie['mass']
inertia = crazyflie['inertia']
G1 = crazyflie['G1']
tau = crazyflie['tau']

env = Hover(num_envs=256, 
            mass=mass, 
            inertia=inertia, 
            G1 = G1,
            tau = tau,
            dt = np.float32(0.01)
            )

ppo = PPO(env, 
          hidden_dim=64, 
          num_layers=2, 
          activation='tanh', 
          n_steps=500, 
          batch_size=4096, 
          n_epochs=10, 
          gamma=0.999, 
          lr=3e-4,
          policy_clip=0.1,
          gae_lambda=0.95,
          directory='tmp/ppo')

ppo.learn(total_timesteps=50000000, save_interval=2000000)
ppo.save_models()
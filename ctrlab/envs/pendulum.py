import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class SimplePendulumEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    
    def __init__(self):
        super(SimplePendulumEnv, self).__init__()
        self.dt = 0.05  # Time step
        self.g = 9.81   # Gravity
        self.l = 1.0    # Length of pendulum
        self.m = 1.0    # Mass of pendulum
        self.max_torque = 2.0
        
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=np.array([-1., -1., -8.]), high=np.array([1., 1., 8.]), shape=(3,), dtype=np.float32)
        
        self.state = None
        self.randomize_inertia()
        self.reset()
        
    def randomize_inertia(self):
        """Randomize inertia values for domain randomization."""
        self.m = random.uniform(0.8, 1.2)  # Random mass within 20% range
        self.l = random.uniform(0.8, 1.2)  # Random length within 20% range
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.randomize_inertia()  # Apply domain randomization at reset
        theta = np.random.uniform(-np.pi, np.pi)  # Random initial angle
        omega = np.random.uniform(-1, 1)          # Random initial angular velocity
        self.state = np.array([theta, omega], dtype=np.float32)
        return self._get_obs(), {}
    
    def step(self, action):
        theta, omega = self.state
        torque = np.clip(action, -self.max_torque, self.max_torque)[0]
        
        # Dynamics (simple Euler integration)
        alpha = (-self.g / self.l) * np.sin(theta) + (torque / (self.m * self.l**2))
        omega += alpha * self.dt
        theta += omega * self.dt
        
        # Keep theta within [-pi, pi]
        theta = (theta + np.pi) % (2 * np.pi) - np.pi
        
        self.state = np.array([theta, omega], dtype=np.float32)
        
        # Reward: Encourage staying upright (theta ~ 0)
        reward = - (theta**2 + 0.1 * omega**2 + 0.001 * (torque**2))
        
        return self._get_obs(), reward, False, False, {}
    
    def _get_obs(self):
        theta, omega = self.state
        return np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float32)
    
    def render(self):
        pass  # Visualization can be added using matplotlib
    
    def close(self):
        pass

# Register the environment
gym.envs.registration.register(
    id='SimplePendulum-v0',
    entry_point=SimplePendulumEnv,
    max_episode_steps=200
)
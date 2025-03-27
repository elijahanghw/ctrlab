import numpy as np
import gymnasium as gym

from ctrlab.systems.drone.drone3D import Drone3D_C, Drone3D_PY
from ctrlab.utils.quaternion import *

MAX_STEPS = 1000

class Hover(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}
    def __init__(self, 
                 mass=1, 
                 inertia=0.1*np.eye(3, dtype=np.float32), 
                 G1 = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [-25, -25, -25, -25],
                                [-2.5, -2.5, 2.5, 2.5],
                                [-2.5, 2.5, -2.5, 2.5],
                                [-1, 1, 1, -1]], dtype=np.float32),
                 dt = np.float32(0.01)
                 ):
        
        super(Hover, self).__init__()
        self.drone = Drone3D_C(mass=mass, inertia=inertia, G1=G1)
        self.drone.setup(dt=dt)
        self.state_len = self.drone.states.shape[0]
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_len,), dtype=np.float32)
        self.states = None
        
        self.num_steps = 0
        
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.num_steps = 0
        
        X = np.random.uniform(low=-2, high=2, size=(3,)).astype(np.float32)
        # E = np.random.uniform(low=-np.pi/8, high=np.pi/8, size=(3,)).astype(np.float32)
        # Q = euler_to_quat(E, dtype=np.float32)
        Q = np.array([1,0,0,0], dtype=np.float32)
        # V = np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
        V = np.array([0,0,0], dtype=np.float32)
        W = np.array([0,0,0]).astype(np.float32)
        
        self.drone.initialize_states(X=X,
                                     Q=Q,
                                     V=V,
                                     W=W)
    
        
        return self._get_obs(), {}
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        action = (action + 1)/2
        self.drone.step(action)

        X = self.drone.states[0:3]
        Q = self.drone.states[3:7]
        V = self.drone.states[7:10]
        W = self.drone.states[10:13]
        
        self.states = self._get_obs()
        
        bounds = 5
        out_of_bounds = (np.abs(X) > bounds).any()
        
        reward = -1*np.linalg.norm(X) - 0.05*np.linalg.norm(V) - 0.05*np.linalg.norm(W)

        self.num_steps += 1
        info = {}   
        if self.num_steps >= MAX_STEPS:
            truncated = True
            info['reach_max_steps'] = True
        elif out_of_bounds:
            truncated = True
            reward = -100
            info['out_of_bounds'] = True
        else:
            truncated = False
        
        if np.linalg.norm(X) < 0.1 and np.linalg.norm(V) < 0.1:
            terminated = True
            info['reached_target'] = True
            reward = 500
        else:
            terminated = False
         
         
        if terminated or truncated:
            info['terminal_state'] = self._get_obs()
            self.reset()

        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        return self.drone.states.astype(np.float32)
    
    def render(self):
        pass  # Visualization can be added using matplotlib
    
    def close(self):
        pass
    
    
gym.register(
    id="drone-hover-v0",
    entry_point=Hover,
    max_episode_steps=MAX_STEPS
)
import numpy as np
import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv

from ctrlab import crazyflie
from ctrlab.systems.drone.drone3D import Drone3D_vec, Drone3D_PY, Drone3D_C
from ctrlab.utils.quaternion import *


class Hover(gym.Env):
    def __init__(self, 
                 mass=crazyflie['mass'], 
                 inertia=crazyflie['inertia'], 
                 G1 = crazyflie['G1'],
                 tau = crazyflie['tau'],
                 dt = np.float32(0.01)
                 ):
        
        self.MAX_STEPS = 500
        self.drone = Drone3D_C(mass=mass, inertia=inertia, G1=G1, tau=tau)
        self.drone.setup(dt=dt)
        self.state_len = self.drone.states.shape[0]
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_len,), dtype=np.float32)
        
        self.dones = False
        self.num_steps = 0
        self.actions = np.zeros((4,), dtype=np.float32)
        
        self.obs = np.zeros((self.state_len,), dtype=np.float32)
        
        self.reset()
        

    def reset(self, seed=None):
        self.num_steps = 0
        self.done = False
        
        X = np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
        Q = self.random_orientations()
        V = np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
        W = np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
        w = np.random.uniform(low=0.66, high=0.66, size=(4,)).astype(np.float32)
        
        self.drone.initialize_states(X=X,
                                     Q=Q,
                                     V=V,
                                     W=W,
                                     w=w)

        
        return self._get_obs(), {}
    
    def random_orientations(self, max_tilt_angle_deg=30, dtype=np.float32):
        max_tilt_angle_rad = np.deg2rad(max_tilt_angle_deg)

        # Random tilt axis (uniform on unit sphere, orthogonal to z-axis)
        axis = np.random.randn(3).astype(dtype)
        axis[2] = 0  # project to x-y plane to make tilt only
        axis = axis / (np.linalg.norm(axis) + 1e-8)

        # Random tilt angle (uniform between 0 and max tilt)
        angle = np.random.uniform(0, max_tilt_angle_rad)

        # Convert axis-angle to quaternion
        quats = axis_angle_to_quat(axis[None, :], np.array([angle], dtype=dtype))[0]

        return quats
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        action = (action + 1)/2
        self.drone.step(action)

        X = self.drone.states[0:3]
        Q = self.drone.states[3:7]
        V = self.drone.states[7:10]
        W = self.drone.states[10:13]
        w = self.drone.states[13:17]
        
        self.states = self._get_obs()
        
        bounds = 1
        max_angular_velocity = 50
        
        distance = np.linalg.norm(X)
        velocity = np.linalg.norm(V)
        angular_velocity = np.linalg.norm(W)
        reward = -2*distance**2 - 0.005*velocity**2 -0.005*angular_velocity**2 - 0.1*(1-np.abs(Q[0])) + 0.01/(distance + 0.01)
        
        # reward = 1/(distance + 0.01) - 0.1*(1-np.abs(Q[0])) - 0.005*velocity**2 - 0.005*angular_velocity**2

        self.num_steps += 1
        
        reach_max_steps = self.num_steps >= self.MAX_STEPS
        out_of_bounds = (np.abs(X) > bounds).any() | (np.abs(W) > max_angular_velocity).any()
        reach_target = (np.linalg.norm(X) < 0.05) & (np.linalg.norm(V) < 0.02)
        
        if out_of_bounds:
            reward -= 500
        elif reach_target:
            reward += 500
        
        self.dones = reach_max_steps | out_of_bounds | reach_target
        
        terminated = out_of_bounds | reach_target
        truncated = reach_max_steps
        
        info = {}

        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        np.copyto(self.obs, self.drone.states)
        return self.obs
    
    def render(self):
        pass
    
    def close(self):
        pass
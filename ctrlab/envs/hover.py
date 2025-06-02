import numpy as np
import gymnasium as gym

from ctrlab import crazyflie
from ctrlab.systems.drone.drone3D import Drone3D_vec
from ctrlab.utils.quaternion import *


class Hover(gym.Env):
    def __init__(self,
                 num_envs=1, 
                 mass=crazyflie['mass'], 
                 inertia=crazyflie['inertia'], 
                 G1 = crazyflie['G1'],
                 tau = crazyflie['tau'],
                 dt = np.float32(0.01)
                 ):
        
        super(Hover, self).__init__()
        self.MAX_STEPS = 500
        self.num_envs = num_envs
        self.drone = Drone3D_vec(num_envs=num_envs, mass=mass, inertia=inertia, G1=G1, tau=tau, use_cuda=False)
        self.drone.setup(dt=dt)
        self.state_len = self.drone.states.shape[1]
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(self.state_len,), dtype=np.float32)
        self.dones = np.zeros(self.num_envs, dtype=bool)
        self.num_steps = np.zeros(self.num_envs, dtype=int)
        
        self.obs = np.zeros_like(self.drone.states, dtype=np.float32)
        
        self.reset(np.ones(self.num_envs, dtype=bool))
        

    def reset(self, dones):
        num_reset = dones.sum()
        self.num_steps[dones] = 0
        
        X = np.random.uniform(low=-0.5, high=0.5, size=(num_reset,3)).astype(np.float32)
        Q = self.random_orientations(num_reset, max_tilt_angle_deg=30, dtype=np.float32)
        V = np.random.uniform(low=-0.5, high=0.5, size=(num_reset,3)).astype(np.float32)
        W = np.random.uniform(low=-0.5, high=0.5, size=(num_reset,3)).astype(np.float32)
        w = np.random.uniform(low=0.66, high=0.66, size=(num_reset,4)).astype(np.float32)
        
        reset_ids = np.where(dones)[0]
        self.drone.states[reset_ids, :3] = X
        self.drone.states[reset_ids, 3:7] = Q
        self.drone.states[reset_ids, 7:10] = V
        self.drone.states[reset_ids, 10:13] = W
        self.drone.states[reset_ids, 13:17] = w
    
        self.dones[dones] = False
        
        return self._get_obs(), {}
    
    def random_orientations(self, num_envs, max_tilt_angle_deg=30, dtype=np.float32):
        max_tilt_angle_rad = np.deg2rad(max_tilt_angle_deg)

        # Random tilt axis (uniform on unit sphere, orthogonal to z-axis)
        axis = np.random.randn(num_envs, 3).astype(dtype)
        axis[:,2] = 0  # project to x-y plane to make tilt only
        axis = axis / (np.linalg.norm(axis, axis=1, keepdims=True) + 1e-8)

        # Random tilt angle (uniform between 0 and max tilt)
        angles = np.random.uniform(0, max_tilt_angle_rad, size=(num_envs,)).astype(dtype)

        # Convert axis-angle to quaternion
        quats = axis_angle_to_quat(axis, angles)

        return quats
    
    def step(self, action):
        action = np.clip(action, -1, 1)
        action = (action + 1)/2
        self.drone.step(action)

        X = self.drone.states[:,0:3]
        Q = self.drone.states[:,3:7]
        V = self.drone.states[:,7:10]
        W = self.drone.states[:,10:13]
        w = self.drone.states[:,13:17]
        
        self.states = self._get_obs()
        
        bounds = 1
        max_angular_velocity = 50
        
        distance = np.linalg.norm(X, axis=1)
        velocity = np.linalg.norm(V, axis=1)
        reward = -2*distance**2 - 0.05*velocity**2 - 2*(1-np.abs(Q[:,0]))**2 - 0.005*np.linalg.norm(action - 0.66)**2 + 2

        self.num_steps += 1
        
        reach_max_steps = self.num_steps >= self.MAX_STEPS
        out_of_bounds = (np.abs(X) > bounds).any(axis=1) | (np.abs(W) > max_angular_velocity).any(axis=1)
        reach_target = (np.linalg.norm(X, axis=1) < 0.05) & (np.linalg.norm(V, axis=1) < 0.01)
        
        reward[out_of_bounds] = -200
        reward[reach_target] = 200
        
        self.dones = reach_max_steps | out_of_bounds | reach_target
        
        terminated = out_of_bounds | reach_target
        truncated = reach_max_steps
        
        info = [{}] * self.num_envs

        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        np.copyto(self.obs, self.drone.states)
        return self.obs
    
    def render(self):
        pass
    
    def close(self):
        pass
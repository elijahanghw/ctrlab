import numpy as np
from numpy import clip

from ctrlab.utils.quaternion import *

class att_PID:
    def __init__(self, K, dt):
        
        self.rate_loop = rate_PID(K_roll=[1,0,0], 
                                  K_pitch=[1,0,0],
                                  K_yaw=[1,0,0],
                                  dt=dt)
        self.K = K
        self.dt = dt
        
        self.E = 0
        
    def initialize(self, states):
        self.rate_loop.initialize(states[10:13])
        self.E = 0
    
    def reset(self):
        self.E = 0
    
    def control(self, desired_q, current_q, current_rates):
        self.compute_error(desired_q, current_q)
        desired_rates = 2*self.K * np.sign(self.E[0]) * self.E[1:]
        delta_m = self.rate_loop.control(desired_rates, current_rates)
        
        return delta_m
    
    def compute_error(self, desired_q, current_q):
        self.E = quat_mult(quat_inv(current_q), desired_q)
    
class rate_PID:
    def __init__(self, K_roll, K_pitch, K_yaw, dt):
        self.K_roll = K_roll
        self.K_pitch = K_pitch
        self.K_yaw = K_yaw
        self.dt = dt
        
        self.Kp = np.array([K_roll[0], K_pitch[0], K_yaw[0]], dtype=np.float32)
        self.Ki = np.array([K_roll[1], K_pitch[1], K_yaw[1]], dtype=np.float32)
        self.Kd = np.array([K_roll[2], K_pitch[2], K_yaw[2]], dtype=np.float32)
        
        self.E = 0
        self.E_prev = 0
        self.E_i = 0
    
    def initialize(self, current_rates):
        self.E = 0
        self.E_prev = 0
        self.E_i = 0
        self.past_rates = current_rates
    
    def control(self, desired_rates, current_rates):
        self.compute_error(desired_rates, current_rates)
        delta_m = self.Kp*self.E + self.Ki*self.E_i + self.Kd*self.E_d
        self.E_prev = self.E.copy()
        self.past_rates = current_rates.copy()
        return delta_m
    
    def compute_error(self, desired_rates, current_rates):
        self.E = desired_rates - current_rates
        self.E_i += self.E*self.dt
        # self.E_d = (self.E - self.E_prev)/self.dt
        self.E_d = -(current_rates - self.past_rates)/self.dt
    
    
def control_mixer(delta_t, delta_m):
    u_min = 0
    u_max = 1
    dp = delta_m[0]
    dq = delta_m[1]
    dr = delta_m[2]
    dT = delta_t
    
    m1 = np.sqrt(clip(0.25*(dT - dp - dq - dr), a_min=u_min, a_max=u_max))
    m2 = np.sqrt(clip(0.25*(dT - dp + dq + dr), a_min=u_min, a_max=u_max))
    m3 = np.sqrt(clip(0.25*(dT + dp - dq + dr), a_min=u_min, a_max=u_max))
    m4 = np.sqrt(clip(0.25*(dT + dp + dq - dr), a_min=u_min, a_max=u_max))
    
    return np.array([m1, m2, m3, m4], dtype=np.float32)
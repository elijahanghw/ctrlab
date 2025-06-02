import ctypes
import numpy as np

from ctrlab.utils.quaternion import *
from ctrlab.utils.integrator import *
                

class FlyBot:
    def __init__(self, 
                 mass=1, 
                 inertia=0.1, 
                 G = np.array([[0, 0],
                               [1, 1],
                               [-0.01, 0.01]], dtype=np.float32)):
        
        self.mass = mass
        self.I = inertia
        self.G = G
        self.initialize_states()

        
    def setup(self, dt=0.01, integration='rk4'):
        if integration not in ['forward_euler', 'rk4']:
            raise Exception("Intergration scheme not supported.")
        
        self.dt = dt
        self.integration = integration
        
    def initialize_states(self,
                          X=np.array([0, 0], dtype=np.float32),
                          Q=np.array([0], dtype=np.float32),
                          V=np.array([0, 0], dtype=np.float32),
                          W=np.array([0], dtype=np.float32),
                          ):
        
        self.states = np.concatenate([X, Q, V, W])
        
    def equations_of_motion(self, x, U):
        X = x[0:2]
        Q = x[2]
        V = x[3:5]
        W = x[5]

        F = self.G @ U
        
        # Rotation matrix
        R = np.array([[np.cos(Q), -np.sin(Q)],
                      [np.sin(Q), np.cos(Q)]])
        
        # Linear acceleration
        dV = (R@F[0:2])/self.mass + np.array([0, -9.81])
        
        # Angular acceleration
        dW = F[2]/self.I
        
        # Change in position
        dX = V.copy()
        
        # Change in orientation
        dQ = W.copy()
        
        
        return np.concatenate([dX, dQ, dV, dW]) 
        
        
    def step(self, U, ground_collision=False):

        f = lambda x: self.equations_of_motion(x, U)
        
        if self.integration == 'rk4':
            self.states = RK4(self.states, f, self.dt)
            
        elif self.integration == 'forward_euler':
            self.states = forward_euler(self.states, f, self.dt)

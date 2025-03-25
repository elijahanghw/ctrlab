import ctypes
import numpy as np

from ctrlab.utils.quaternion import *
from ctrlab.utils.integrator import *


# Load the shared library
dronelib = ctypes.CDLL('./lib/dronelib.so')


dronelib.integrate_rk4.argtypes = [
    ctypes.c_float, # mass
    ctypes.POINTER(ctypes.c_float),  # I
    ctypes.POINTER(ctypes.c_float),  # I_inv
    ctypes.POINTER(ctypes.c_float),  # G1
    ctypes.POINTER(ctypes.c_float),  # states
    ctypes.POINTER(ctypes.c_float),  # U
    ctypes.c_float                   # dt
]

dronelib.integrate_euler.argtypes = [
    ctypes.c_float, # mass
    ctypes.POINTER(ctypes.c_float),  # I
    ctypes.POINTER(ctypes.c_float),  # I_inv
    ctypes.POINTER(ctypes.c_float),  # G1
    ctypes.POINTER(ctypes.c_float),  # states
    ctypes.POINTER(ctypes.c_float),  # U
    ctypes.c_float                   # dt
]


class Drone3D_C:
    def __init__(self, 
                 mass=1, 
                 inertia=0.1*np.eye(3, dtype=np.float32), 
                 G1 = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [-25, -25, -25, -25],
                                [-2.5, -2.5, 2.5, 2.5],
                                [-2.5, 2.5, -2.5, 2.5],
                                [-1, 1, 1, -1]], dtype=np.float32)):
        self.mass = mass
        self.I = inertia
        self.I_inv = np.linalg.inv(inertia)
        self.G1 = G1
        self.initialize_states()

        
    def setup(self, dt=0.01, integration='rk4'):
        if integration not in ['forward_euler', 'rk4']:
            raise Exception("Intergration scheme not supported.")
        
        self.dt = dt
        self.integration = integration
        
    def initialize_states(self,
                          X=np.array([0, 0, 0]),
                          Q=np.array([1, 0, 0, 0]),
                          V=np.array([0, 0, 0]),
                          W=np.array([0, 0, 0])
                          ):
        
        self.states = np.concatenate([X, Q, V, W])
        
    def step(self, U, ground_collision=False):
        derivatives = np.zeros(13, dtype=np.float32)
        
        # Convert the matrix and vector to pointers to pass to the C function
        c_I= self.I.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_I_inv = self.I_inv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_G1 = self.G1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_states = self.states.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_U = U.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the C function
        if self.integration == 'rk4':
            dronelib.integrate_rk4(self.mass, c_I, c_I_inv, c_G1, c_states, c_U, self.dt)
            # self.states[3:7] = quat_normalize(self.states[3:7])
            
        elif self.integration == 'forward_euler':
            dronelib.integrate_euler(self.mass, c_I, c_I_inv, c_G1, c_states, c_U, self.dt)
            # self.states[3:7] = quat_normalize(self.states[3:7])

        if ground_collision:
            if self.states[2] > 0:
                self.states[2] = 0
                self.states[9] = 0
                
                

class Drone3D_PY:
    def __init__(self, 
                 mass=1, 
                 inertia=0.1*np.eye(3, dtype=np.float32), 
                 G1 = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [-25, -25, -25, -25],
                                [-2.5, -2.5, 2.5, 2.5],
                                [-2.5, 2.5, -2.5, 2.5],
                                [-1, 1, 1, -1]], dtype=np.float32)):
        self.mass = mass
        self.I = inertia
        self.I_inv = np.linalg.inv(inertia)
        self.G1 = G1
        self.initialize_states()

        
    def setup(self, dt=0.01, integration='rk4'):
        if integration not in ['forward_euler', 'rk4']:
            raise Exception("Intergration scheme not supported.")
        
        self.dt = dt
        self.integration = integration
        
    def initialize_states(self,
                          X=np.array([0, 0, 0], dtype=np.float32),
                          Q=np.array([1, 0, 0, 0], dtype=np.float32),
                          V=np.array([0, 0, 0], dtype=np.float32),
                          W=np.array([0, 0, 0], dtype=np.float32)
                          ):
        
        self.states = np.concatenate([X, Q, V, W])
        
    def equations_of_motion(self, x, U):
        X = x[0:3]
        Q = x[3:7]
        V = x[7:10]
        W = x[10:13]
        
        # Forces
        F = self.G1 @ U
        
        # Rotation matrix
        R = quat_to_rot(Q)
        
        # Linear acceleration
        dV = F[0:3]/self.mass - np.cross(W, V) + R @ np.array([0, 0, 9.81])
        
        # Angular acceleration
        dW = self.I_inv @ (F[3:6] - np.cross(W, self.I@W))
        
        # Change in position
        dX = R.T @ V
        
        # Change in quaternion
        dQ = quat_derivative(Q, W)
        
        return np.concatenate([dX, dQ, dV, dW]) 
        
        
    def step(self, U, ground_collision=False):

        f = lambda x: self.equations_of_motion(x, U)
        
        # Call the C function
        if self.integration == 'rk4':
            self.states = RK4(self.states, f, self.dt)
            self.states[3:7] = quat_normalize(self.states[3:7])
            
        elif self.integration == 'forward_euler':
            self.states = forward_euler(self.states, f, self.dt)
            self.states[3:7] = quat_normalize(self.states[3:7])

        if ground_collision:
            if self.states[2] > 0:
                self.states[2] = 0
                self.states[9] = 0

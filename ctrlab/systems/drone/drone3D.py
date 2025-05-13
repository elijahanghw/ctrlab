import ctypes
import numpy as np

from ctrlab.utils.quaternion import *
from ctrlab.utils.integrator import *


# Load the shared library
# dronelib = ctypes.CDLL('./lib/dronelib.so')
LIB_PATH_CPU = './lib/dronelib.so'
LIB_PATH_CUDA = './lib/dronelibcuda.so'

class Drone3D_C:
    def __init__(self, 
                 mass=1, 
                 inertia=0.1*np.eye(3, dtype=np.float32), 
                 G1 = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [-25, -25, -25, -25],
                                [-2.5, -2.5, 2.5, 2.5],
                                [-2.5, 2.5, -2.5, 2.5],
                                [-1, 1, 1, -1]], dtype=np.float32),
                                tau = np.float32(0.01)):
        
        # Load the shared library
        self.dronelib = ctypes.CDLL('./lib/dronelib.so')
        self.dronelib.integrate_rk4.argtypes = [
            ctypes.c_float, # mass
            ctypes.POINTER(ctypes.c_float),  # I
            ctypes.POINTER(ctypes.c_float),  # I_inv
            ctypes.POINTER(ctypes.c_float),  # G1
            ctypes.c_float,  # tau
            ctypes.POINTER(ctypes.c_float),  # states
            ctypes.POINTER(ctypes.c_float),  # U
            ctypes.c_float                   # dt
        ]
        
        self.dronelib.integrate_euler.argtypes = [
            ctypes.c_float, # mass
            ctypes.POINTER(ctypes.c_float),  # I
            ctypes.POINTER(ctypes.c_float),  # I_inv
            ctypes.POINTER(ctypes.c_float),  # G1
            ctypes.c_float,  # tau
            ctypes.POINTER(ctypes.c_float),  # states
            ctypes.POINTER(ctypes.c_float),  # U
            ctypes.c_float                   # dt
        ]
                
        self.mass = mass
        self.I = inertia
        self.I_inv = np.linalg.inv(inertia)
        self.G1 = G1
        self.tau = tau
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
                          W=np.array([0, 0, 0]),
                          w=np.array([0, 0, 0, 0])
                          ):
        
        self.states = np.concatenate([X, Q, V, W, w])
        
    def step(self, U, ground_collision=False):
        # Convert the matrix and vector to pointers to pass to the C function
        c_I= self.I.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_I_inv = self.I_inv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_G1 = self.G1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_states = self.states.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_U = U.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the C function
        if self.integration == 'rk4':
            self.dronelib.integrate_rk4(self.mass, c_I, c_I_inv, c_G1, self.tau, c_states, c_U, self.dt)
            # self.states[3:7] = quat_normalize(self.states[3:7])
            
        elif self.integration == 'forward_euler':
            self.dronelib.integrate_euler(self.mass, c_I, c_I_inv, c_G1, self.tau, c_states, c_U, self.dt)
            # self.states[3:7] = quat_normalize(self.states[3:7])

        if ground_collision:
            if self.states[2] > 0:
                self.states[2] = 0
                self.states[9] = 0
                
                
class Drone3D_vec:
    def __init__(self, 
                 num_envs=1,
                 mass=1, 
                 inertia=0.1*np.eye(3, dtype=np.float32), 
                 G1 = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [-25, -25, -25, -25],
                                [-2.5, -2.5, 2.5, 2.5],
                                [-2.5, 2.5, -2.5, 2.5],
                                [-1, 1, 1, -1]], dtype=np.float32),
                 tau = np.float32(0.01),
                 use_cuda=False):
        
        self.dronelib = ctypes.CDLL(LIB_PATH_CUDA if use_cuda else LIB_PATH_CPU)
            
        self.dronelib.vec_integrate_rk4.argtypes = [
            ctypes.c_int,                    # num_envs
            ctypes.c_float,                  # mass
            ctypes.POINTER(ctypes.c_float),  # I
            ctypes.POINTER(ctypes.c_float),  # I_inv
            ctypes.POINTER(ctypes.c_float),  # G1
            ctypes.c_float,                  # tau
            ctypes.POINTER(ctypes.c_float),  # states
            ctypes.POINTER(ctypes.c_float),  # U
            ctypes.c_float                   # dt
        ]
        
        self.dronelib.vec_integrate_euler.argtypes = [
            ctypes.c_int,                    # num_envs
            ctypes.c_float,                  # mass
            ctypes.POINTER(ctypes.c_float),  # I
            ctypes.POINTER(ctypes.c_float),  # I_inv
            ctypes.POINTER(ctypes.c_float),  # G1
            ctypes.c_float,                  # tau
            ctypes.POINTER(ctypes.c_float),  # states
            ctypes.POINTER(ctypes.c_float),  # U
            ctypes.c_float                   # dt
        ]
        
        self.num_envs = num_envs
        self.mass = mass
        self.I = inertia
        self.I_inv = np.linalg.inv(inertia)
        self.G1 = G1
        self.tau = tau
        
        self.states = np.zeros((num_envs, 13), dtype=np.float32)
        
        self.initialize_states(X=np.array([[0, 0, 0]]*self.num_envs),
                               Q=np.array([[1, 0, 0, 0]]*self.num_envs),
                               V=np.array([[0, 0, 0]]*self.num_envs),
                               W=np.array([[0, 0, 0]]*self.num_envs),
                               w=np.array([[0, 0, 0, 0]]*self.num_envs)
                               )

        
    def setup(self, dt=0.01, integration='rk4'):
        if integration not in ['forward_euler', 'rk4']:
            raise Exception("Intergration scheme not supported.")
        
        self.dt = dt
        self.integration = integration
        
    def initialize_states(self, X, Q, V, W, w):
            self.states = np.concatenate([X, Q, V, W, w], axis=1).astype(np.float32)
        
    def step(self, U, ground_collision=False):
        # Convert the matrix and vector to pointers to pass to the C function
        c_I= self.I.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_I_inv = self.I_inv.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_G1 = self.G1.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_states = self.states.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_U = U.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Call the C function
        if self.integration == 'rk4':
            self.dronelib.vec_integrate_rk4(self.num_envs, self.mass, c_I, c_I_inv, c_G1, self.tau, c_states, c_U, self.dt)
            # self.states[3:7] = quat_normalize(self.states[3:7])
            
        elif self.integration == 'forward_euler':
            self.dronelib.vec_integrate_euler(self.num_envs, self.mass, c_I, c_I_inv, c_G1, self.tau, c_states, c_U, self.dt)
            # self.states[3:7] = quat_normalize(self.states[3:7])

        if ground_collision:
            hit_ground = self.states[:, 2] > 0
            self.states[hit_ground, 2] = 0
            self.states[hit_ground, 9] = 0
                
                

class Drone3D_PY:
    def __init__(self, 
                 mass=1, 
                 inertia=0.1*np.eye(3, dtype=np.float32), 
                 G1 = np.array([[0, 0, 0, 0],
                                [0, 0, 0, 0],
                                [-25, -25, -25, -25],
                                [-2.5, -2.5, 2.5, 2.5],
                                [-2.5, 2.5, -2.5, 2.5],
                                [-1, 1, 1, -1]], dtype=np.float32),
                 tau = np.float32(0.01)):
        
        self.mass = mass
        self.I = inertia
        self.I_inv = np.linalg.inv(inertia)
        self.G1 = G1
        self.tau = tau
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
                          W=np.array([0, 0, 0], dtype=np.float32),
                          w=np.array([0, 0, 0, 0], dtype=np.float32)
                          ):
        
        self.states = np.concatenate([X, Q, V, W, w])
        
    def equations_of_motion(self, x, U):
        X = x[0:3]
        Q = x[3:7]
        V = x[7:10]
        W = x[10:13]
        w = x[13:17]
        
        # Forces
        rpm = w**2
        F = self.G1 @ rpm
        
        # Rotation matrix
        R = quat_to_rot(Q)
        
        # Linear acceleration
        dV = F[0:3]/self.mass - np.cross(W, V) + R.T @ np.array([0, 0, 9.81])
        
        # Angular acceleration
        dW = self.I_inv @ (F[3:6] - np.cross(W, self.I@W))
        
        # Change in position
        dX = R @ V
        
        # Change in quaternion
        dQ = quat_derivative(Q, W)
        
        # Change in propeller speed
        dw = (U - w)/self.tau
        
        return np.concatenate([dX, dQ, dV, dW, dw]) 
        
        
    def step(self, U, ground_collision=False):

        f = lambda x: self.equations_of_motion(x, U)
        
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

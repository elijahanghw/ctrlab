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
        # self.states = np.zeros((17,), dtype=np.float32)
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
        U = U.astype(np.float32)
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
        
        self.use_cuda = use_cuda
        self.num_envs = num_envs
        self.mass = mass
        self.I = inertia
        self.I_inv = np.linalg.inv(inertia)
        self.G1 = G1
        self.tau = tau
        
        self.states = np.zeros((num_envs, 17), dtype=np.float32)
        
        self.dronelib = ctypes.CDLL(LIB_PATH_CUDA if use_cuda else LIB_PATH_CPU)

        if use_cuda:
            self._setup_cuda()
        else:
            self._setup_cpu()
        
        
        self.initialize_states(
            X=np.zeros((num_envs, 3), dtype=np.float32),
            Q=np.tile(np.array([[1, 0, 0, 0]], dtype=np.float32), (num_envs, 1)),
            V=np.zeros((num_envs, 3), dtype=np.float32),
            W=np.zeros((num_envs, 3), dtype=np.float32),
            w=np.zeros((num_envs, 4), dtype=np.float32)
        )

    
    def _setup_cpu(self):
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
        
        self.dronelib.vec_integrate_euler.argtypes = self.dronelib.vec_integrate_rk4.argtypes
    
    def _setup_cuda(self):
        self.sim = ctypes.c_void_p()

        self.dronelib.drone_sim_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        self.dronelib.drone_sim_free.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

        self.dronelib.drone_sim_step_rk4.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_float]
        self.dronelib.drone_sim_step_euler.argtypes = self.dronelib.drone_sim_step_rk4.argtypes
        self.dronelib.drone_sim_copy_states_to_host.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.dronelib.drone_sim_copy_inputs_from_host.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]
        self.dronelib.drone_sim_set_constants.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float)
        ]
        self.dronelib.drone_sim_set_states.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_float)]

        self.dronelib.drone_sim_init(ctypes.byref(self.sim), self.num_envs)

        # Upload constants with sim pointer directly
        c_I = self.I.astype(np.float32).flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_I_inv = self.I_inv.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        c_G1 = self.G1.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.dronelib.drone_sim_set_constants(self.sim, c_I, c_I_inv, c_G1)
    
    def setup(self, dt=0.01, integration='rk4'):
        if integration not in ['forward_euler', 'rk4']:
            raise Exception("Intergration scheme not supported.")
        
        self.dt = np.float32(dt)
        self.integration = integration
        
    def initialize_states(self, X, Q, V, W, w):
            self.states = np.concatenate([X, Q, V, W, w], axis=1).astype(np.float32)
            
    def _upload_states_to_gpu(self):
        c_states = self.states.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.dronelib.drone_sim_set_states(self.sim, c_states)

    def _upload_inputs_to_gpu(self, U):
        c_U = U.flatten().astype(np.float32).ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.dronelib.drone_sim_copy_inputs_from_host(self.sim, c_U)

    def _download_states_from_gpu(self):
        c_states = self.states.flatten().ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        self.dronelib.drone_sim_copy_states_to_host(self.sim, c_states)
        
    def step(self, U, ground_collision=False):
        if self.use_cuda:
            self._upload_inputs_to_gpu(U)

            if self.integration == 'rk4':
                self.dronelib.drone_sim_step_rk4(self.sim, self.mass, self.tau, self.dt)
            elif self.integration == 'forward_euler':
                self.dronelib.drone_sim_step_euler(self.sim, self.mass, self.tau, self.dt)

            # self._download_states_from_gpu()
            
        else:
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
            
    def download(self):
        if self.use_cuda:
            self.dronelib.download_states(self.sim_ptr, self.states.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return self.states.copy()

    def free(self):
        if self.use_cuda and self.sim_ptr:
            self.dronelib.destroy_sim(self.sim_ptr)
            self.sim_ptr = None

    def __del__(self):
        self.free()
                

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
        Q = quat_normalize(x[3:7])
        V = x[7:10]
        W = x[10:13]
        w = x[13:17]
        
        # Forces
        rpm = w**2

        F = self.G1 @ rpm
        
        # Rotation matrix
        R = quat_to_rot(Q)
        
        # Linear acceleration
        dV = (R@F[0:3])/self.mass + np.array([0, 0, 9.81])
        
        # Angular acceleration
        dW = self.I_inv @ (F[3:6] - np.cross(W, self.I@W))
        
        # Change in position
        dX = V.copy()
        
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

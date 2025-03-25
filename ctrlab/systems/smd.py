import numpy as np

from ctrlab.math.integrator import *

class SpringMassDamper:
    def __init__(self, 
                 mass=1, 
                 stiffness=0.1, 
                 damping=0.0):
        
        self.M = mass
        self.K = stiffness
        self.C = damping
        
    def equations_of_motion(self, states, input):
        A_sys = np.array([[0, 1],
                          [-self.K/self.M, -self.C/self.M]])
        
        B_sys = np.array([[0],[1/self.M]])
        
        d_states = A_sys @ states + B_sys @ input
        
        return d_states
    
    def setup(self, dt=0.01, integration='rk4'):
        self.dt = dt
        if integration == 'forward_euler':
            self.integrate = forward_euler
        elif integration == 'rk4':
            self.integrate = RK4
        else:
            raise Exception("Intergration scheme not supported.")
        
    def initialize_states(self,
                          x1=0,
                          x2=0):
        
        self.states = np.array([x1, x2])
        
    def step(self, U):
        f = lambda x: self.equations_of_motion(x, U)
        self.states = self.integrate(self.states, f, self.dt)
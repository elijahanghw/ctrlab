from time import time
import numpy as np
import matplotlib.pyplot as plt

from ctrlab.systems.drone.drone3D import Drone3D_C, Drone3D_PY
from ctrlab.utils.quaternion import euler_to_quat

numsteps = 1000
dt=np.float32(0.01)

X=np.array([0, 0, 0], dtype=np.float32)
Q=euler_to_quat([0, 0, 0], dtype=np.float32)
V=np.array([0, 0, 0], dtype=np.float32)
W=np.array([0, 0, 0], dtype=np.float32)

U = np.array([0.4, 0.2, 0.5, 0.0], dtype=np.float32)

# Simulator in C
simc = Drone3D_C(mass=1, inertia=0.1*np.eye(3, dtype=np.float32))

simc.setup(dt=dt)
simc.initialize_states(X=X,
                      Q=Q,
                      V=V,
                      W=W)

statesc = np.zeros((numsteps+1, 13))

statesc[0,:] = simc.states

start_time = time()     
for i in range(1, numsteps+1): 
    
    simc.step(U, ground_collision=False)
    statesc[i,:] = simc.states
    
print(f"Computation time for C : {time() - start_time}")

# Simulator in python
simpy = Drone3D_PY(mass=1, inertia=0.1*np.eye(3, dtype=np.float32))

simpy.setup(dt=dt)
simpy.initialize_states(X=X,
                      Q=Q,
                      V=V,
                      W=W)

statespy = np.zeros((numsteps+1, 13))

statespy[0,:] = simpy.states

start_time = time()     
for i in range(1, numsteps+1): 
    
    simpy.step(U, ground_collision=False)
    statespy[i,:] = simpy.states
    
print(f"Computation time for python : {time() - start_time}")

rms = np.sqrt(np.mean((statesc - statespy)**2))
print(f"RMS : {rms}")

plt.plot(statesc[:,0], statesc[:,2])
plt.plot(statespy[:,0], statespy[:,2])
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.gca().invert_yaxis()
plt.savefig("drone3D.png")
# plt.show()

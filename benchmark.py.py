from time import time
import numpy as np
import matplotlib.pyplot as plt

from ctrlab.systems.drone.drone3D import *
from ctrlab.utils.quaternion import euler_to_quat

numsteps = 1000
dt=np.float32(0.01)

X=np.array([0, 0, 0], dtype=np.float32)
Q=euler_to_quat([0, 0, 0], dtype=np.float32)
V=np.array([0, 0, 0], dtype=np.float32)
W=np.array([0, 0, 0], dtype=np.float32)

U = np.array([0.5, 0.3, 0.5, 0.3], dtype=np.float32)

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
    
print(f"FPS for C : {numsteps/(time() - start_time)}")


# Simulator in vectorized C
num_envs = 1000
U_vec = np.array([U] * num_envs, dtype=np.float32)
simvec = Drone3D_vec(num_envs=num_envs, mass=1, inertia=0.1*np.eye(3, dtype=np.float32))

simvec.setup(dt=dt)
simvec.initialize_states(X=np.array([X] * num_envs),
                         Q=np.array([Q] * num_envs),
                         V=np.array([V] * num_envs),
                         W=np.array([W] * num_envs))

statesvec = np.zeros((numsteps+1, 13))

statesvec[0,:] = simvec.states[10]

start_time = time()     
for i in range(1, numsteps+1): 
    simvec.step(U_vec, ground_collision=False)
    statesvec[i,:] = simvec.states[10]
    
print(f"FPS for vectorized C ({num_envs} environments) : {numsteps*num_envs/(time() - start_time)}")


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
    
print(f"FPS for python : {numsteps/(time() - start_time)}")

c_py_rms = np.sqrt(np.mean((statesc - statespy)**2))
print(f"RMS (C-PY) : {c_py_rms}")

vec_py_rms = np.sqrt(np.mean((statesvec - statespy)**2))
print(f"RMS (VEC-PY) : {vec_py_rms}")

c_vec_rms = np.sqrt(np.mean((statesc - statesvec)**2))
print(f"RMS (C-VEC) : {c_vec_rms}")

plt.plot(statesc[:,0], statesc[:,2])
plt.plot(statespy[:,0], statespy[:,2])
plt.plot(statesvec[:,0], statesvec[:,2])
plt.xlim([-10, 10])
plt.ylim([-10, 10])
plt.xlabel("x [m]")
plt.ylabel("z [m]")
plt.gca().invert_yaxis()
plt.savefig("figures/benchmark.png")

plt.figure()
plt.plot(statesc[:,3])
plt.plot(statespy[:,3])
plt.plot(statesvec[:,3])
plt.savefig("figures/attitude.png")
# plt.show()

from time import time
import numpy as np
import matplotlib.pyplot as plt

from ctrlab.systems.drone.drone3D import Drone3D_C
from ctrlab.systems.drone.controller import att_PID, rate_PID, control_mixer
from ctrlab.utils.quaternion import euler_to_quat

# Simulation parameters
numsteps = 300
dt=np.float32(0.01)

# Drone setup
drone = Drone3D_C(mass=1, inertia=0.1*np.eye(3, dtype=np.float32))
drone.setup(dt=dt)
drone.initialize_states(X=np.array([0, 0, 0], dtype=np.float32),
                        Q=euler_to_quat([np.pi/8, np.pi/8, 0], dtype=np.float32),
                        V=np.array([0, 0, 0], dtype=np.float32),
                        # W=np.random.uniform(low=-0.5, high=0.5, size=(3,)).astype(np.float32)
                        W=np.array([0, 0, 0], dtype=np.float32)
                      )

states = np.zeros((numsteps+1, 13))
states[0,:] = drone.states

# Controller setup
attitude_controller = att_PID(K=5, dt=dt)

attitude_controller.initialize(drone.states)
desired_attitude = np.array([1, 0, 0, 0], dtype=np.float32)
desired_thrust = np.array([0, 0, 0.5], dtype=np.float32)
cmd = np.zeros((numsteps, 4))

start_time = time()     
for i in range(1, numsteps+1): 
    delta_m = attitude_controller.control(desired_attitude, drone.states[3:7], drone.states[10:13])
    U = control_mixer(0.5, delta_m)
    cmd[i-1,:] = U
    drone.step(U, ground_collision=False)
    states[i,:] = drone.states


plt.figure()
plt.plot(states[:,3], label="q0", color="red")
plt.plot(states[:,4], label="q1", color="blue")
plt.plot(states[:,5], label="q2", color="black")
plt.plot(states[:,6], label="q3", color="green")

plt.plot(desired_attitude[0]*np.ones(numsteps+1), label="Desired q0", color="red", linestyle="--")
plt.plot(desired_attitude[1]*np.ones(numsteps+1), label="Desired q1", color="blue", linestyle="--")
plt.plot(desired_attitude[2]*np.ones(numsteps+1), label="Desired q2", color="black", linestyle="--")
plt.plot(desired_attitude[3]*np.ones(numsteps+1), label="Desired q3", color="green", linestyle="--")
plt.xlabel("Time")
plt.ylabel("q")
plt.legend()
plt.grid()
plt.savefig("attitude_drone.png")

plt.figure()
plt.plot(cmd[:,0], label="M1", color="red")
plt.plot(cmd[:,1], label="M2", color="blue")
plt.plot(cmd[:,2], label="M3", color="black")
plt.plot(cmd[:,3], label="M4", color="green")
plt.xlabel("Time")
plt.ylabel("Motor Command")
plt.legend()
plt.grid()
plt.savefig("cmd_drone.png")


plt.show()

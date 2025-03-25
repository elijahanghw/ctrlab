import numpy as np
import matplotlib.pyplot as plt

from ctrlab.systems.smd import SpringMassDamper
from ctrlab.controllers.pid import PID

numsteps = 500
dt = 0.02

sim = SpringMassDamper(mass=1, stiffness=1, damping=0.0)
sim.setup(dt=dt)
sim.initialize_states(x1=0, x2=0)

controller = PID(P=5,
                 I=2,
                 D=3,
                 dt=dt)

ref = 1
controller.initialize_error(ref=ref, state=sim.states[0])

states = np.zeros((numsteps+1, 2))

states[0,:] = sim.states

input = np.zeros((numsteps, 1))

for i in range(1, numsteps+1):
    U = controller.control(ref=ref, state=sim.states[0])
    sim.step(np.array([U]))
    states[i,:] = sim.states
    input[i-1,:] = U

fig = plt.figure()
ax1 = fig.add_subplot(131)    
ax1.plot([i*dt for i in range(numsteps+1)], states[:,0])
ax1.plot([i*dt for i in range(numsteps+1)], ref*np.ones_like([i*dt for i in range(numsteps+1)]), color='grey', linestyle='--')
ax1.set_xlabel("t [s]")
ax1.set_ylabel("x1 [m]")
ax1.grid()
ax1.set_title("Position")

ax2 = fig.add_subplot(132)  
ax2.plot([i*dt for i in range(numsteps+1)], states[:,1])
ax2.set_xlabel("t [s]")
ax2.set_ylabel("x2 [m/s]")
ax2.grid()
ax2.set_title("Velocity")

ax3 = fig.add_subplot(133)      
ax3.plot([i*dt for i in range(numsteps)], input[:,0])
ax3.set_xlabel("t [s]")
ax3.set_ylabel("F [N]")
ax3.grid()
ax3.set_title("Force input")

plt.show()
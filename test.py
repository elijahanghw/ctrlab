import numpy as np
from numpy import pi
from ctrlab.utils.quaternion import *

g = np.array([0, 0, 9.81], dtype=np.float32)

T = np.array([0, 0, -1])

euler = [pi/2, 0, 0]

q = euler_to_quat(euler)

R_q = quat_to_rot(q)

R_e = euler_to_rot(euler)

print(R_q@T)

print(R_e@T)
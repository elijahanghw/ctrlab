import numpy as np

def quat_to_rot(q):
    q0, q1, q2, q3 = quat_normalize(q)  # Normalize the quaternion

    R = np.array([
        [1 - 2 * (q2**2 + q3**2), 2 * (q1*q2 - q0*q3), 2 * (q1*q3 + q0*q2)],
        [2 * (q1*q2 + q0*q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2*q3 - q0*q1)],
        [2 * (q1*q3 - q0*q2), 2 * (q2*q3 + q0*q1), 1 - 2 * (q1**2 + q2**2)]
    ], dtype=np.float32)

    return R

def quat_normalize(q):
    return q / np.linalg.norm(q) 

def quat_derivative(q, w):
    q = quat_normalize(q)

    # Construct Omega matrix
    Omega = np.array([
        [0, -w[0], -w[1], -w[2]],
        [w[0], 0, w[2], -w[1]],
        [w[1], -w[2], 0, w[0]],
        [w[2], w[1], -w[0], 0]
    ])

    # Compute quaternion derivative
    dq = 0.5 * Omega @ q
    return dq.astype(np.float32)

def euler_to_quat(angle, dtype=np.float32):
    phi = angle[0]
    theta = angle[1]
    psi = angle[2]
    
    cy = np.cos(psi * 0.5)
    sy = np.sin(psi * 0.5)
    cp = np.cos(theta * 0.5)
    sp = np.sin(theta * 0.5)
    cr = np.cos(phi * 0.5)
    sr = np.sin(phi * 0.5)

    q0 = cr * cp * cy + sr * sp * sy
    q1 = sr * cp * cy - cr * sp * sy
    q2 = cr * sp * cy + sr * cp * sy
    q3 = cr * cp * sy - sr * sp * cy

    return quat_normalize(np.array([q0, q1, q2, q3], dtype=dtype))

def quat_to_euler(q):
    q0, q1, q2, q3 = quat_normalize(q)  # Normalize quaternion

    # Roll (X-axis rotation)
    sinr_cosp = 2 * (q0 * q1 + q2 * q3)
    cosr_cosp = 1 - 2 * (q1**2 + q2**2)
    phi = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (Y-axis rotation)
    sinp = 2 * (q0 * q2 - q3 * q1)
    if np.abs(sinp) >= 1:
        theta = np.sign(sinp) * (np.pi / 2)  # Clamp to 90 degrees
    else:
        theta = np.arcsin(sinp)

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (q0 * q3 + q1 * q2)
    cosy_cosp = 1 - 2 * (q2**2 + q3**2)
    psi = np.arctan2(siny_cosp, cosy_cosp)

    return phi, theta, psi

def quat_mult(q1, q2):
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)

    q_0 = q1[0]*q2[0] - q1[1]*q2[1] - q1[2]*q2[2] - q1[3]*q2[3]
    q_1 = q1[0]*q2[1] + q1[1]*q2[0] + q1[2]*q2[3] - q1[3]*q2[2]
    q_2 = q1[0]*q2[2] - q1[1]*q2[3] + q1[2]*q2[0] + q1[3]*q2[1]
    q_3 = q1[0]*q2[3] + q1[1]*q2[2] - q1[2]*q2[1] + q1[3]*q2[0]

    return np.array([q_0, q_1, q_2, q_3], dtype=np.float32)

def quat_inv(q):
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


from numpy import sin, cos
def euler_to_rot(angle):
    phi = angle[0]
    theta = angle[1]
    psi = angle[2]
    
    Rx = np.array([[1, 0, 0], [0, cos(phi), -sin(phi)], [0, sin(phi), cos(phi)]])
    Ry = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]])
    Rz = np.array([[cos(psi), -sin(psi), 0], [sin(psi), cos(psi), 0], [0, 0, 1]])
    R = Rz@Ry@Rx
    
    return R

def axis_angle_to_quat(axis, angle):
    # axis: [N, 3], angle: [N]
    half_angle = angle / 2
    sin_half = np.sin(half_angle)
    q = np.zeros((axis.shape[0], 4), dtype=axis.dtype)
    q[:,0] = np.cos(half_angle)             # w
    q[:,1:] = axis * sin_half[:,None]       # x, y, z
    return q

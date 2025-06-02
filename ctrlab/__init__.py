from . import *
import numpy as np

crazyflie = {'mass': np.float32(0.027),
            'inertia': np.diag([3.85e-6, 3.85e-6, 5.9675e-6]).astype(np.float32),
            'G1': np.array([[0, 0, 0, 0],
                            [0, 0, 0, 0],
                            [-0.15, -0.15, -0.15, -0.15],
                            [-4.2e-3, -4.2e-3, 4.2e-3, 4.2e-3],
                            [-4.2e-3, 4.2e-3, -4.2e-3, 4.2e-3],
                            [-8.9e-4, 8.9e-4, 8.9e-4, -8.9e-4]], dtype=np.float32),
            'tau': np.float32(0.15),
            }
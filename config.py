import numpy as np
import time
import matplotlib.pyplot as plt

DRONE_R = 0.3
SENSOR_R = 3.0
EPSILON = 0.1
NUM_UAV = 3
DT = 0.05

W_u   = 0.1
W_sep = 4.0
W_dir = 2.0
W_nav = 1.0
W_obs = 1.0

HEIGHT_BOUNDS = np.array([2.0, 20.0])

CONTROL_BOUNDS = np.array([[-2.0, 2.0],
                           [-2.0, 2.0],
                           [-1.0, 1.0]])
AMAX = np.linalg.norm(CONTROL_BOUNDS[:, 1])


VREF = 1.0
UREF = np.array([1,0,0])
VELO_BOUNDS = np.array([[-2.0, 2.0],
                        [-2.0, 2.0],
                        [-0.5, 0.5]])
VMAX = 2

MAX_STEP_D = VMAX * DT
DREF = 1.5
# Obstacle x, y, r
OBSTACLES = np.array([[4.0, 2.0, 0.1],
                      [5.0,-1.0, 0.1],
                      [6.0, 0.0, 0.1],
                      [6.0, 1.0, 0.1],
                      [6.0, 4.0, 0.2],
                      [8.0,-1.0, 0.1],
                      [8.0, 2.0, 0.1]])

# OBSTACLES = np.array([[6.0, -0.6, 0.1],
#                       [6.0,  0.0, 0.1],
#                       [6.0,  0.6, 0.1],
#                       [6.0,  1.2, 0.1],
#                       [6.0,  1.8, 0.1]])

# OBSTACLES = np.array([[4.0, 2.0, 0.1],
#                       [5.0,-1.0, 0.1],
#                       [6.0, 1.0, 0.1],
#                       [6.0, 4.0, 0.2],
#                       [8.0,-1.0, 0.1],
#                       [8.0, 2.0, 0.1],
#                       [9.0, 0.0, 0.1],])

                     

# Position x, y ,z
INIT_POSITIONS = np.array([[1.5, 0.3, 5.0],
                           [0.0, 0.1, 5.0],
                           [0.4, 1.2, 5.0]])

# Velocity u_x, u_y, u_z
INIT_VELOS = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])

INIT_STATES = np.concatenate([INIT_POSITIONS, INIT_VELOS], axis=1)


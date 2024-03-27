import numpy as np
import time
import matplotlib.pyplot as plt

DRONE_R = 0.3
SENSOR_R = 3.0
EPSILON = 0.1
N_UAV = 5
N_NEIGHBOR = 3
DT = 0.05

X_GOAL = 12

HEIGHT_BOUNDS = np.array([2.0, 20.0])

CONTROL_BOUNDS = np.array([[-2.0, 2.0],
                           [-2.0, 2.0],
                           [-1.0, 1.0]])
AMAX = np.linalg.norm(CONTROL_BOUNDS[:, 1])

VREF = 1
UREF = np.array([1,0,0])
VELO_BOUNDS = np.array([[-2.0, 2.0],
                        [-2.0, 2.0],
                        [-0.5, 0.5]])
VMAX = 2
MAX_STEP_D = VMAX * DT
DREF = 1.5
# Obstacle x, y, r
# #CASE1
# OBSTACLES = np.array([[4.0, 2.0, 0.1],
#                       [5.0,-1.0, 0.1],
#                       [6.0, 1.0, 0.1],
#                       [6.0, 4.0, 0.2],
#                       [8.0,-1.0, 0.1],
#                       [8.0, 2.0, 0.1],
#                       [9.0, 0.0, 0.1],])

# #CASE2
# OBSTACLES = np.array([[4.0, 0.0, 0.1],
#                       [4.0,-2.0, 0.1],
#                       [6.0,-1.5, 0.1],
#                       [6.0, 1.0, 0.1],
#                       [6.0, 2.0, 0.1],
#                       [7.0, 3.0, 0.1],
#                       [8.0, 4.5, 0.1],
#                       [9.0, 6.0, 0.1],])

# #CASE3
# OBSTACLES = np.array([[4.0,-2.0, 0.1],
#                       [4.0, 1.0, 0.1],
#                       [4.0, 4.0, 0.1],
#                       [5.0,-3.0, 0.1],
#                       [5.0, 2.0, 0.1],
#                       [6.0, 6.0, 0.1],
#                       [7.0,-1.0, 0.1],
#                       [8.0, 0.0, 0.1]])

#CASE4
OBSTACLES = np.array([[4.0,-1.0, 0.2],
                      [6.5, 0.5, 0.2],
                      [8.0,-1.0, 0.2],
                      [8.0, 2.0, 0.2],
                      [4.0, 2.0, 0.2],
                      [6.0, 3.0, 0.2],
                      [2.0, 0.0, 0.2],
                      [9.0, 1.0, 0.2]])

# Position x, y ,z
INIT_POSITIONS = np.array([[1.0, 0.0, 4.9],
                           [0.0, 0.1, 5.0],
                           [0.4, 1.2, 5.1],
                           [1.7, 1.2, 4.8],
                           [0.5, 0.6, 4.2]])

# INIT_POSITIONS = np.array([[1.5, 0.3, 4.9],
#                            [0.0, 0.1, 5.0],
#                            [0.4, 1.2, 5.1]])

# Velocity v_x, v_y, v_z
INIT_VELOS = np.array([[0.0, 0.0, 0.0] for i in range(N_UAV)])

INIT_STATES = np.concatenate([INIT_POSITIONS, INIT_VELOS], axis=1)


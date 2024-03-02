import numpy as np
import time
import matplotlib.pyplot as plt

ROBOT_RADIUS = 0.3
EPSILON = 0.1
NUM_UAV = 3
DT = 0.1


CONTROL_BOUNDS = np.array([[-2.0, 2.0],
                           [-2.0, 2.0],
                           [-1.0, 1.0]])
AMAX = np.linalg.norm(CONTROL_BOUNDS[:2, 1])


VREF = 1.0
UREF = np.array([1,0,0])
VELO_BOUNDS = np.array([[-1.4, 1.4],
                        [-1.4, 1.4],
                        [-0.5, 0.5]])
VMAX = np.linalg.norm(VELO_BOUNDS[:2, 1])
MAX_STEP_D = VMAX * DT

DREF = 1.0


W_sep = 3.0
W_dir = 1.0
W_nav = 5.0
W_u = 1e-1
# W_obs = 1e4

# Obstacle x, y, r
OBSTACLES = np.array([[6.0, 1.0, 0.1],
                      [6.0, 0.0, 0.1],
                      [8.0,-1.0, 0.1],
                      [8.0, 2.0, 0.1],
                      [4.0, 2.0, 0.1]])

INIT_POSITIONS = np.array([[1.2, 0.0, 5.0],
                           [0, 0.6, 5.0],
                           [0.6, 0.2, 5.0]])

INIT_VELOS = np.array([[0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0]])

INIT_STATES = np.concatenate([INIT_POSITIONS, INIT_VELOS], axis=1)


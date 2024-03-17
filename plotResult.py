from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from config_init import *
from config_plot import *




BC_paths = []
MPC_paths = []

for i in range(N_UAV):
    BC_paths.append(np.load(PATH_FILE_NAME.format(method="BC", index=i)))
    MPC_paths.append(np.load(PATH_FILE_NAME.format(method="MPC", index=i)))
#Plot XY paths
plotXYPaths(BC_paths, MPC_paths)

# ax = plt.axes(projection="3d")

# Plot Speed
plotSpeeds(BC_paths, MPC_paths)

# Plot distance
plotDistances(BC_paths, MPC_paths)

# Plot order
plotOrder(BC_paths, MPC_paths)
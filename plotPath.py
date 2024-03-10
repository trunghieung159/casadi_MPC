from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from config import *

def getCircle(x,y,r):
    theta = np.linspace( 0 , 2 * np.pi , 150 )   
    a = x + r * np.cos( theta )
    b = y + r * np.sin( theta )
    return a, b

path0 = np.load("path_0.npy")
path1 = np.load("path_1.npy")
path2 = np.load("path_2.npy")

# ax = plt.axes(projection="3d")
plt.figure("MPC_XY_path")
ax = plt.axes()
ax.set_title("Drone path")
ax.grid(True)
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.axis('equal')

# Plot obstacles
for j in range(OBSTACLES.shape[0]):
    x, y, r = OBSTACLES[j,:]
    a, b = getCircle(x, y, r)
    ax.plot(a, b, '-k')
    a, b = getCircle(x, y, r+DRONE_R-0.05)
    ax.plot(a, b, '--k')

# Plot path
ax.plot(path0[:,1], path0[:,2], label="Drone 0")
ax.plot(path1[:,1], path1[:,2], label="Drone 1")
ax.plot(path2[:,1], path2[:,2], label="Drone 2")
plt.legend()
plt.title("MPC: Motion XY paths")

# Plot Speed
plt.figure(num="MPC_speed")
speeds = np.array([np.linalg.norm(path0[:,4:7], axis=1),
                   np.linalg.norm(path1[:,4:7], axis=1),
                   np.linalg.norm(path2[:,4:7], axis=1)]).T
plt.fill_between(path0[:,0], np.min(speeds,axis=1), np.max(speeds,axis=1), color="#1f77b4", label="Max/Min", alpha=0.3)
plt.plot(path0[:,0], np.mean(speeds,axis=1), 'b-', label="Average")
plt.plot([path0[0,0], path0[-1,0]], [VREF, VREF], 'g--', label="VREF") 
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.xlim([0, path0[-1,0]])
plt.legend()
plt.grid(True)
plt.title("MPC: Drone speed")
# Plot distance
plt.figure(num="MPC_distance")
distances = np.array([np.linalg.norm(path0[:,1:4]-path1[:,1:4], axis=1),
                      np.linalg.norm(path1[:,1:4]-path2[:,1:4], axis=1),
                      np.linalg.norm(path2[:,1:4]-path0[:,1:4], axis=1)]).T
plt.fill_between(path0[:,0], np.min(distances,axis=1), np.max(distances,axis=1), color="#1f77b4", label="Max/Min", alpha=0.3)
plt.plot(path0[:,0], np.mean(distances,axis=1), 'b-', label="Average")
plt.plot([path0[0,0], path0[-1,0]], [DREF, DREF], 'g--', label='DREF')
plt.plot([path0[0,0], path0[-1,0]], [2*DRONE_R, 2*DRONE_R], 'k--', label="Safety radius")
plt.xlabel("Time (s)")
plt.ylabel("Inter-agent distance (m)")
plt.xlim([0, path0[-1,0]])
plt.legend()
plt.grid(True)
plt.title("MPC: Inter-agent distances")
# Plot order
plt.figure(num="MPC_order")
headings = []
for i in range(1,len(path0)):
    heading = path0[i,4:6]/np.linalg.norm(path0[i,4:6]) \
            + path1[i,4:6]/np.linalg.norm(path1[i,4:6]) \
            + path2[i,4:6]/np.linalg.norm(path2[i,4:6])
    headings.append(np.linalg.norm(heading)/NUM_UAV)
plt.plot(path0[1:,0], headings)
plt.xlabel("Time (s)")
plt.ylabel("Order")
plt.xlim([0, path0[-1,0]])
plt.grid(True)
plt.title("MPC: Drone orders")
plt.show()
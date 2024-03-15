from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from config_init import *

def getCircle(x,y,r):
    theta = np.linspace( 0 , 2 * np.pi , 150 )   
    a = x + r * np.cos( theta )
    b = y + r * np.sin( theta )
    return a, b

path0_mpc = np.load("MPC_path_0.npy")
path1_mpc = np.load("MPC_path_1.npy")
path2_mpc = np.load("MPC_path_2.npy")

path0_bc = np.load("BC_path_0.npy")
path1_bc = np.load("BC_path_1.npy")
path2_bc = np.load("BC_path_2.npy")
# ax = plt.axes(projection="3d")
plt.figure("MPC_vs_BC_XY_path")
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
ax.plot(path0_mpc[:,1], path0_mpc[:,2], 'c', label="MPC: Drone 0", linewidth =0.7)
ax.plot(path1_mpc[:,1], path1_mpc[:,2], 'm', label="MPC: Drone 1", linewidth =0.7)
ax.plot(path2_mpc[:,1], path2_mpc[:,2], 'y', label="MPC: Drone 2", linewidth =0.7)

ax.plot(path0_bc[:,1], path0_bc[:,2], 'c--', label="BC: Drone 0", linewidth =0.7)
ax.plot(path1_bc[:,1], path1_bc[:,2], 'm--', label="BC: Drone 1", linewidth =0.7)
ax.plot(path2_bc[:,1], path2_bc[:,2], 'y--', label="BC: Drone 2", linewidth =0.7)
plt.legend()
plt.title("MPC vs BC: Motion XY paths")

# Plot Speed
plt.figure(num="MPC_vs_BC_speed")
speeds_mpc = np.array([np.linalg.norm(path0_mpc[:,4:7], axis=1),
                   np.linalg.norm(path1_mpc[:,4:7], axis=1),
                   np.linalg.norm(path2_mpc[:,4:7], axis=1)]).T
speeds_bc = np.array([np.linalg.norm(path0_bc[:,4:7], axis=1),
                   np.linalg.norm(path1_bc[:,4:7], axis=1),
                   np.linalg.norm(path2_bc[:,4:7], axis=1)]).T
plt.fill_between(path0_mpc[:,0], np.min(speeds_mpc,axis=1), np.max(speeds_mpc,axis=1), color="#1f77b4", label="MPC Max/Min", alpha=0.3)
plt.fill_between(path0_bc[:,0], np.min(speeds_bc,axis=1), np.max(speeds_bc,axis=1), color="#e9828c", label="BC Max/Min", alpha=0.3)

plt.plot(path0_mpc[:,0], np.mean(speeds_mpc,axis=1), 'b-', label="MPC Average")
plt.plot(path0_bc[:,0], np.mean(speeds_bc,axis=1), 'r-', label="BC Average")
plt.plot([path0_mpc[0,0], max(path0_mpc[-1,0], path0_bc[-1,0])], [VREF, VREF], 'g--', label="VREF") 
plt.xlabel("Time (s)")
plt.ylabel("Speed (m/s)")
plt.xlim([0, max(path0_mpc[-1,0], path0_bc[-1,0])])
plt.legend()
plt.grid(True)
plt.title("MPC vs BC: Drone speed")
# Plot distance
plt.figure(num="MPC_vs_BC_distance")
distances_mpc = np.array([np.linalg.norm(path0_mpc[:,1:4]-path1_mpc[:,1:4], axis=1),
                      np.linalg.norm(path1_mpc[:,1:4]-path2_mpc[:,1:4], axis=1),
                      np.linalg.norm(path2_mpc[:,1:4]-path0_mpc[:,1:4], axis=1)]).T
distances_bc = np.array([np.linalg.norm(path0_bc[:,1:4]-path1_bc[:,1:4], axis=1),
                      np.linalg.norm(path1_bc[:,1:4]-path2_bc[:,1:4], axis=1),
                      np.linalg.norm(path2_bc[:,1:4]-path0_bc[:,1:4], axis=1)]).T
plt.fill_between(path0_mpc[:,0], np.min(distances_mpc,axis=1), np.max(distances_mpc,axis=1), color="#1f77b4", label="MPC Max/Min", alpha=0.3)
plt.fill_between(path0_bc[:,0], np.min(distances_bc,axis=1), np.max(distances_bc,axis=1), color="#e9828c", label="BC Max/Min", alpha=0.3)

plt.plot(path0_mpc[:,0], np.mean(distances_mpc,axis=1), 'b-', label="MPC Average")
plt.plot(path0_bc[:,0], np.mean(distances_bc,axis=1), 'r-', label="BC Average")

plt.plot([path0_mpc[0,0], max(path0_mpc[-1,0], path0_bc[-1,0])], [DREF, DREF], 'g--', label='DREF')
# plt.plot([path0_mpc[0,0], max(path0_mpc[-1,0], path0_bc[-1,0])], [2*DRONE_R, 2*DRONE_R], 'k--', label="Safety radius")
plt.xlabel("Time (s)")
plt.ylabel("Inter-agent distance (m)")
plt.xlim([0, max(path0_mpc[-1,0], path0_bc[-1,0])])
plt.legend()
plt.grid(True)
plt.title("MPC vs BC: Inter-agent distances")
# Plot order
plt.figure(num="MPC_vs_BC_order")
headings_mpc = []
for i in range(1,len(path0_mpc)):
    heading = path0_mpc[i,4:6]/np.linalg.norm(path0_mpc[i,4:6]) \
            + path1_mpc[i,4:6]/np.linalg.norm(path1_mpc[i,4:6]) \
            + path2_mpc[i,4:6]/np.linalg.norm(path2_mpc[i,4:6])
    headings_mpc.append(np.linalg.norm(heading)/N_UAV)
headings_bc = []
for i in range(1,len(path0_bc)):
    heading = path0_bc[i,4:6]/np.linalg.norm(path0_bc[i,4:6]) \
            + path1_bc[i,4:6]/np.linalg.norm(path1_bc[i,4:6]) \
            + path2_bc[i,4:6]/np.linalg.norm(path2_bc[i,4:6])
    headings_bc.append(np.linalg.norm(heading)/N_UAV)

plt.plot(path0_mpc[1:,0], headings_mpc, 'b', label="MPC")
plt.plot(path0_bc[1:,0], headings_bc, 'r', label="BC")

plt.xlabel("Time (s)")
plt.ylabel("Order")
plt.xlim([0, max(path0_mpc[-1,0], path0_bc[-1,0])])
plt.legend()
plt.grid(True)
plt.title("MPC vs BC: Drone orders")
plt.show()
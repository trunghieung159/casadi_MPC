import matplotlib.pyplot as plt
from config_init import *

PATH_FILE_NAME = "path_{method}_{index}.npy"
TIMES_FILE_NAME = "process_time_{method}.npy"

def getCircle(x,y,r):
    theta = np.linspace( 0 , 2 * np.pi , 150 )   
    a = x + r * np.cos( theta )
    b = y + r * np.sin( theta )
    return a, b

def plotObs(ax):
    for j in range(OBSTACLES.shape[0]):
        x, y, r = OBSTACLES[j,:]
        a, b = getCircle(x, y, r)
        ax.plot(a, b, '-k')
        a, b = getCircle(x, y, r+DRONE_R-0.05)
        ax.plot(a, b, '--k')

def plotXYpath(ax, paths, method):
    plotObs(ax)
    ax.set_title("{method}: drone XY paths".format(method=method))
    ax.grid(True)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.axis('equal')
    for i in range(N_UAV):
        ax.plot(paths[i][:,1], paths[i][:,2], 
                label="drone {index}".format(index = i), 
                linewidth =0.7)

def plotXYPaths(BC_paths, MPC_paths):
    fig, axs = plt.subplots(1, 2, num="MPC vs BC: XY paths")
    plotXYpath(axs[0], BC_paths, "BC")
    plotXYpath(axs[1], MPC_paths, "MPC")
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels)

def plotSpeeds(BC_paths, MPC_paths):
    plt.figure(num="MPC_vs_BC_speed")
    speeds_mpc = np.array([np.linalg.norm(MPC_paths[i][:,4:7], axis=1) for i in range(N_UAV)]).T
    speeds_bc = np.array([np.linalg.norm(BC_paths[i][:,4:7], axis=1) for i in range(N_UAV)]).T

    plt.fill_between(MPC_paths[0][:,0], np.min(speeds_mpc,axis=1),
                    np.max(speeds_mpc,axis=1), color="#1f77b4", 
                    label="MPC Max/Min", alpha=0.3)
    plt.fill_between(BC_paths[0][:,0], np.min(speeds_bc,axis=1), 
                    np.max(speeds_bc,axis=1), color="#e9828c", 
                    label="BC Max/Min", alpha=0.3)

    plt.plot(MPC_paths[0][:,0], np.mean(speeds_mpc,axis=1), 'b', label="MPC Average")
    plt.plot(BC_paths[0][:,0], np.mean(speeds_bc,axis=1), 'r', label="BC Average")
    plt.plot([MPC_paths[0][0,0], max(BC_paths[0][-1,0], BC_paths[0][-1,0])], 
            [VREF, VREF], 'g--', label="VREF") 
    plt.xlabel("Time (s)")
    plt.ylabel("Speed (m/s)")
    plt.xlim([0, max(MPC_paths[0][-1,0], BC_paths[0][-1,0])])
    plt.legend()
    plt.grid(True)
    plt.title("MPC vs BC: Drone speed")

def plotDistances(BC_paths, MPC_paths):
    plt.figure(num="MPC_vs_BC_distance")
    mpc_distances_array = []
    bc_distances_array = []
    for i in range(N_UAV):
        for j in range(i+1, N_UAV):
            mpc_distances_array.append(np.linalg.norm(MPC_paths[i][:,1:4] 
                                                    - MPC_paths[j][:,1:4], axis=1))
            bc_distances_array.append(np.linalg.norm(BC_paths[i][:,1:4] 
                                                    - BC_paths[j][:,1:4], axis=1))

    distances_mpc = np.array(mpc_distances_array).T
    distances_bc = np.array(bc_distances_array).T

    plt.fill_between(MPC_paths[0][:,0], np.min(distances_mpc,axis=1), 
                    np.max(distances_mpc,axis=1), color="#1f77b4", 
                    label="MPC Max/Min", alpha=0.3)
    plt.fill_between(BC_paths[0][:,0], np.min(distances_bc,axis=1), 
                    np.max(distances_bc,axis=1), color="#e9828c", 
                    label="BC Max/Min", alpha=0.3)

    plt.plot(MPC_paths[0][:,0], 
            np.mean(distances_mpc,axis=1), 
            'b-', label="MPC Average")
    plt.plot(BC_paths[0][:,0], 
            np.mean(distances_bc,axis=1), 
            'r-', label="BC Average")

    plt.plot([MPC_paths[0][0,0], max(MPC_paths[0][-1,0], BC_paths[0][-1,0])], 
            [DREF, DREF], 'g--', label='DREF')
    # plt.plot([MPC_paths[0][0,0], max(MPC_paths[0][-1,0], BC_paths[0][-1,0])], 
    #          [2*DRONE_R, 2*DRONE_R], 'k--', label="Safety radius")
    plt.xlabel("Time (s)")
    plt.ylabel("Inter-agent distance (m)")
    plt.xlim([0, max(MPC_paths[0][-1,0], BC_paths[0][-1,0])])
    plt.legend()
    plt.grid(True)
    plt.title("MPC vs BC: Inter-agent distances")

def plotOrder(MPC_paths, BC_paths):
    plt.figure(num="MPC_vs_BC_order")
    headings_mpc = []
    for i in range(1,len(MPC_paths[0])):
        heading = np.zeros(2)
        for j in range(N_UAV):
            if np.linalg.norm(MPC_paths[j][i,4:6]) != 0:
                heading += MPC_paths[j][i,4:6]/np.linalg.norm(MPC_paths[j][i,4:6]) 
        headings_mpc.append(np.linalg.norm(heading)/N_UAV)
        
    headings_bc = []
    for i in range(1,len(BC_paths[0])):
        heading = np.zeros(2)
        for j in range(N_UAV):
            if np.linalg.norm(BC_paths[j][i,4:6]) != 0:
                heading += BC_paths[j][i,4:6]/np.linalg.norm(BC_paths[j][i,4:6]) 
        headings_bc.append(np.linalg.norm(heading)/N_UAV)
        
    plt.plot(MPC_paths[0][1:,0], headings_mpc, 'b', label="MPC")
    plt.plot(BC_paths[0][1:,0], headings_bc, 'r', label="BC")

    plt.xlabel("Time (s)")
    plt.ylabel("Order")
    plt.xlim([0, max(MPC_paths[0][-1,0], BC_paths[0][-1,0])])
    plt.legend()
    plt.grid(True)
    plt.title("MPC vs BC: Drone orders")
    plt.show()
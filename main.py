

from DroneMPC import DroneMPC
from DroneBC import DroneBC
from setup import *

import numpy as np
import time
import matplotlib.pyplot as plt
if __name__ == "__main__":
    drones = []
    known_obs = set()
    # Initialize Drone
    for i in range(N_UAV):
        drone = DroneBC(i, INIT_STATES[i, :], known_obs)
        drones.append(drone)
    for i in range(N_UAV):
        drones[i].setupController()
    
    compute_times = []
    iter = 0
    try:
        print("[INFO] Start")
        run = True
        while run:
            times = []
            controls = []
            for i in range(N_UAV):
                # compute velocity using nmpc
                start = time.time()
                controls.append(drones[i].computeControlSignal(drones, known_obs))
                times.append(time.time()-start)
            compute_times.append(times)
            known_obs = set()
            for i in range(N_UAV):
                drones[i].updateState(controls[i], known_obs)
            iter += 1
            if iter % 10 == 0:
                print("Iteration {}".format(iter))
            #Reach terminal condition
            count = 0
            for i in range(N_UAV):
                if drones[i].state[0] > X_GOAL:
                    count += 1
            run = count < N_UAV
    finally:
        print("[INFO] Saving")
        # Saving
        for i in range(N_UAV):
            drone = drones[i]
            path = np.array(drone.path)
            np.save("path_{}.npy".format(drone.index), path)
            np.save("process_time.npy", compute_times)

        compute_times = np.array(compute_times)
        print("Average time: {:.6}s".format(compute_times.mean()))
        print("Max time: {:.6}s".format(compute_times.max()))   
        print("Min time: {:.6}s".format(compute_times.min()))

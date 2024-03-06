

from Drone import Drone
from config import *

import numpy as np
import time
import matplotlib.pyplot as plt
import warnings
if __name__ == "__main__":
    warnings.simplefilter('ignore', np.RankWarning)
    drones = []
    known_obs = set()
    # Initialize Drone
    for i in range(NUM_UAV):
        drone = Drone(i, INIT_STATES[i, :], 10, known_obs)
        drones.append(drone)
    for i in range(NUM_UAV):
        drones[i].setupController(drones)
    
    compute_times = []
    iter = 0
    try:
        print("[INFO] Start")
        run = True
        while run:
            times = []
            for i in range(NUM_UAV):
                # compute velocity using nmpc
                start = time.time()
                control = drones[i].computeControlSignal(drones, known_obs)
                times.append(time.time()-start)
                drones[i].updateState(control, known_obs)

            compute_times.append(times)
            iter += 1
            if iter % 10 == 0:
                print("Iteration {}".format(iter))
            
            # if iter > 90:
            #     break
            #Reach terminal condition
            count = 0
            for i in range(NUM_UAV):
                if drones[i].state[0] > 15:
                    count += 1
            run = count < NUM_UAV
    finally:
        print("[INFO] Saving")
        # Saving
        for i in range(NUM_UAV):
            drone = drones[i]
            path = np.array(drone.path)
            np.save("path_{}.npy".format(drone.index), path)
            np.save("process_time.npy", compute_times)

        compute_times = np.array(compute_times)
        print("Average time: {:.6}s".format(compute_times.mean()))
        print("Max time: {:.6}s".format(compute_times.max()))   
        print("Min time: {:.6}s".format(compute_times.min()))

from DroneMPC import DroneMPC
from DroneBC import DroneBC
from config_init import *

def runAndSave(method : str):
    drones = []
    known_obs = set()
    # Initialize Drone
    if method == "BC":
        for i in range(N_UAV):
            drone = DroneBC(i, INIT_STATES[i, :], known_obs)
            drones.append(drone)
    elif method == "MPC":
        for i in range(N_UAV):
            drone = DroneMPC(i, INIT_STATES[i, :], known_obs)
            drones.append(drone)
    else:
        print("Invalid control method. Exit now")
        return
    #Set up controller
    for i in range(N_UAV):
        drones[i].setupController()
    
    compute_times = []
    iter = 0
    try:
        print(method, "Computing")
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
        print(method ,"Saving")
        # Saving
        path_file_name = "{method}_path_{index}.npy"
        times_file_name = "{method}_process_time.npy"
        for i in range(N_UAV):
            path = np.array(drones[i].path)
            np.save(path_file_name.format(method = method, index =i), path)
        np.save(times_file_name.format(method = method), compute_times)
        compute_times = np.array(compute_times)
        print("Average time: {:.6}s".format(compute_times.mean()))
        print("Max time: {:.6}s".format(compute_times.max()))   
        print("Min time: {:.6}s".format(compute_times.min()))
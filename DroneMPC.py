import numpy as np
import math
import time
import casadi as ca
import warnings
from config_init import *
from config_mpc import *

class DroneMPC:
    def __init__(self, index:int, state:np.array, known_obs):
        self.index = index
        
        # Drone state and control
        self.time_stamp = 0.0
        self.step = 0
        self.state = state
        self.control = np.array([0.0, 0.0, 0.0]) 

        self.n_state = 6
        self.n_control = 3

        #Update known obs
        self.__update_known_obs(known_obs)

        # State predictions
        self.state_predicts = np.zeros((N_PREDICT+1, self.n_state))
        for i in range(N_PREDICT + 1):
            self.state_predicts[i, :] = self.state
        self.control_predicts = np.zeros((N_PREDICT, self.n_control))

        # Store drone path
        self.path = [np.concatenate([[self.time_stamp], self.state, self.control])]

    def updateState(self, control:np.array, known_obs):
        """
        Computes the states of drone after applying control signals
        Update state predictions, known obstacles
        """
        # Update state
        position = self.state[:3]
        velocity = self.state[3:]

        next_velocity = velocity + control*DT
        avg_velo = velocity + control*DT/2
        next_position = position + avg_velo*DT

        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp += DT
        self.step += 1
        
        # Store
        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control]))

        #Update known obstacles
        self.__update_known_obs(known_obs)

        #Update state and control predicts
        self.state_predicts[:N_PREDICT, :] = self.state_predicts_next[1:, :]
        self.control_predicts[:N_PREDICT-1, :] = self.control_predicts_next[1:, :]

        self.control_predicts[N_PREDICT - 1, :] = np.array([0, 0, 0]) 
        self.state_predicts[N_PREDICT, :] = compute_next_state(self.state_predicts[N_PREDICT-1, :],
                                                               self.control_predicts[N_PREDICT - 1, :],
                                                               "np")
        

    def setupController(self):
        '''Set up init state, constrains for predictions'''
        # Predictive length
        self.opti = ca.Opti()
        # states and controls variable 
        self.opt_controls = self.opti.variable(N_PREDICT, self.n_control)
        self.opt_states = self.opti.variable(N_PREDICT+1, self.n_state)
        
        # initial condition
        self.opt_start = self.opti.parameter(self.n_state)

        #control constrains
        self.opti.subject_to(self.opti.bounded(CONTROL_BOUNDS[0, 0], 
                                               self.opt_controls[:,0], 
                                               CONTROL_BOUNDS[0, 1]))
        self.opti.subject_to(self.opti.bounded(CONTROL_BOUNDS[1, 0],
                                               self.opt_controls[:,1],
                                               CONTROL_BOUNDS[1, 1]))
        self.opti.subject_to(self.opti.bounded(CONTROL_BOUNDS[2, 0], 
                                               self.opt_controls[:,2], 
                                               CONTROL_BOUNDS[2, 1]))

        # height constrains
        self.opti.subject_to(self.opti.bounded(HEIGHT_BOUNDS[0], 
                                               self.opt_states[:, 2], 
                                               HEIGHT_BOUNDS[1]))
    
        #velocity constrains
        self.opti.subject_to(self.opti.bounded(VELO_BOUNDS[0, 0], 
                                               self.opt_states[:, 3], 
                                               VELO_BOUNDS[0, 1]))
        
        self.opti.subject_to(self.opti.bounded(VELO_BOUNDS[1, 0], 
                                               self.opt_states[:, 4], 
                                               VELO_BOUNDS[1, 1]))
        
        self.opti.subject_to(self.opti.bounded(VELO_BOUNDS[2, 0], 
                                               self.opt_states[:, 5], 
                                               VELO_BOUNDS[2, 1]))
        velo_sqr = self.opt_states[:, 3]**2 + self.opt_states[:, 4]**2 + self.opt_states[:, 5]**2
        self.opti.subject_to(velo_sqr <= VMAX**2)

        #step-to-step constrains
        self.opti.subject_to(self.opt_states[0, :] == self.opt_start.T)
        for i in range(N_PREDICT):
            next_state = compute_next_state(self.opt_states[i, :],
                                             self.opt_controls[i, :],
                                             "casadi") 
            self.opti.subject_to(self.opt_states[i+1, :] == next_state)

        # add drone collision constraints
        self.neighbor_states = [self.opti.parameter(N_PREDICT+1, self.n_state) for _ in range(N_NEIGHBOR)]
        # for idx in range(N_NEIGHBOR):
        #     neighbor_state = self.neighbor_states[idx]
        #     for i in range(N_PREDICT+1):
        #         distance = ca.norm_2(self.opt_states[i,:3]-neighbor_state[i,:3])
        #         self.opti.subject_to(distance > 2*DRONE_R)
        

        # #drone-to-drone distance constrains:
        # self.drones_appr_pos = [self.opti.parameter(N_PREDICT, 3) 
        #                         for i in range(N_UAV - 1)]
        # covariance = np.array([0.5 * AMAX * (i*DT)**2 
        #                        for i in range(1, N_PREDICT+1)]) 
        # for i in range(N_UAV - 1):
        #     for j in range(0, N_PREDICT):
        #         distance = ca.norm_fro(self.opt_states[j+1, :3] 
        #                                - self.drones_appr_pos[i][j, :])
        #         self.opti.subject_to(distance > 2 * DRONE_R 
        #                                      + covariance[j] + MAX_STEP_D)

        # #obstacle-distance constrains
        # for i in range(N_PREDICT+1):
        #     for j in range(OBSTACLES.shape[0]):
        #         distance = ca.norm_fro(self.opt_states[i, :2].T - OBSTACLES[j, :2])
        #         self.opti.subject_to(distance > DRONE_R + 
        #                              OBSTACLES[j,2])
        opts_setting = {'ipopt.max_iter': 1e8,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-1,
                        'ipopt.acceptable_obj_change_tol': 1e-1}
        self.opti.solver('ipopt', opts_setting)  

    def computeControlSignal(self, drones, known_obs):
        """
        Computes control signal for drones
        """
        # cost function
        obj = self.costFunction(self.opt_states, self.opt_controls, drones, known_obs)
        self.opti.minimize(obj)
        
        #initials for predictions
        self.opti.set_value(self.opt_start, self.state)

        # set parameter, here only update predictive state of neighbors
        self.neighbors_distance = np.array([np.linalg.norm(self.state[:3]-drones[i].state[:3]) for i in self.neighbor_indices])
        for i, idx in enumerate(self.neighbor_indices):
            self.opti.set_value(self.neighbor_states[i], drones[idx].state_predicts)

        # #set drones approximate position for drone-to-drone distance constrains
        # index = 0
        # for i in range(N_UAV):
        #     if i == self.index:
        #         continue
        #     for j in range(N_PREDICT):
        #         self.opti.set_value(self.drones_appr_pos[index][j, :], 
        #                             drones[i].state[:3] + 
        #                             drones[i].state[3:] * (j+1) * DT)
        #     index += 1
        #     if index > N_UAV-2:
        #         break

        # provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.state_predicts)
        self.opti.set_initial(self.opt_controls, self.control_predicts)
        
        # solve the problem
        sol = self.opti.solve()
        ## obtain the control input
        self.control_predicts_next = sol.value(self.opt_controls)
        self.state_predicts_next = sol.value(self.opt_states)
        return sol.value(self.opt_controls)[0,:]

    def costFunction(self, opt_states, opt_controls, drones, known_obs):

        c_u = self.costControl(opt_controls)
        c_sep = self.costSeparation(opt_states, drones)
        c_spe = self.costSpeed(opt_states)
        # c_nav = self.costNavigation(opt_states)
        c_dir = self.costDirection(opt_states)
        c_obs = self.costObstacle(opt_states, known_obs)
        c_dip = self.costDisplacement(opt_states)
        total = W_sep*c_sep + W_spe*c_spe + W_dip*c_dip + W_obs*c_obs + W_u*c_u
        # total = W_sep*c_sep + W_dir*c_dir  + W_spe*c_spe + W_obs*c_obs + W_u*c_u
        return total

    # Minimal control signal
    def costControl(self, u):
        cost_u = 0
        for i in range(N_PREDICT):
            control = u[i,:]
            cost_u += ca.dot(control, control)
        # print("u: ", cost_u.shape)
        return cost_u/N_PREDICT

    def costSeparation(self, traj, drones):
        cost_sep = 0
        for j in self.neighbor_indices:
            for i in range(1, N_PREDICT + 1): 
                pos_rel = drones[j].state_predicts[i,:3] - traj[i,:3].T
                cost = (ca.dot(pos_rel, pos_rel) - DREF**2)**2
                cost_sep += cost
        # print("sep: ", cost_sep.shape)
        return cost_sep / N_NEIGHBOR / N_PREDICT

    # def costNavigation(self, traj):
    #     cost_nav = 0
    #     for i in range(1, N_PREDICT + 1):
    #         vel = traj[i,3:]
    #         v_ref = VREF*UREF
    #         cost_nav += ca.norm_fro(vel.T - v_ref)**2
    #     # print("nav: ", cost_dir.shape)
    #     return cost_nav / N_PREDICT
    
    def costDirection(self, traj):
        cost_dir = 0
        for i in range(1, N_PREDICT + 1):
            vel = traj[i,3:].T
            # print(vel.shape)
            vel_norm = ca.norm_2(vel)
            d = ca.dot(vel, UREF)
            cost_dir += (vel_norm**3 - d**3)**2
            # print("dir: ", cost_dir.shape)
        return cost_dir / N_PREDICT
    
    def costDisplacement(self, traj):
        cost_dis = 0
        for i in range(1, N_PREDICT + 1):
            vel = traj[i,3:]
            d = ca.dot(vel.T, UREF)
            cost_dis += (d**3 - VREF**3)**2
        return cost_dis / N_PREDICT
    
    def costSpeed(self, traj):
        cost_spe = 0
        for i in range(1, N_PREDICT + 1):
            vel = traj[i,3:]
            velo_sqr = ca.dot(vel, vel)
            cost = (velo_sqr**2 - VREF**4)**2
            cost_spe += cost
        # print("nav: ", cost_nav.shape)
        return cost_spe / N_PREDICT

    def costObstacle(self, traj, known_obs):
        cost_obs = 0
        for i in range(1, N_PREDICT + 1):
            for j in known_obs:
                drone_to_centre = OBSTACLES[j, :2] - traj[i, :2].T
                drone_to_centre_norm = ca.norm_fro(drone_to_centre)
                drone_to_edge_norm = drone_to_centre_norm -  OBSTACLES[j, 2] 
                cost = 1 / (drone_to_edge_norm**2 - DRONE_R*2)**2   
                # cost = 1 / (drone_to_edge_norm - DRONE_R)**2   
                cost_obs += cost
        if len(known_obs) == 0:
            return 0
        return cost_obs / N_PREDICT / len(known_obs)
    
    def update_neighbors(self, drones):
        '''
        Set N_NEIGHBOR closest drones as neighbors 
        '''
        distance_list = []
        for i in range(N_UAV):
            dis = np.linalg.norm(self.state[:3] - drones[i].state[:3])
            distance_list.append(dis)
        distance_list = np.array(distance_list)
        neighbor_set = np.argsort(distance_list)[1:1+N_NEIGHBOR]
        self.neighbor_indices = neighbor_set

    def __update_known_obs(self, known_obs):
        '''Update known obstacles by sensor'''
        for i in range(OBSTACLES.shape[0]):
            distance = np.linalg.norm(self.state[:2] - OBSTACLES[i, :2])
            if distance - OBSTACLES[i, 2] < SENSOR_R:
                known_obs.add(i) 

            
def compute_next_state(last_state, control, return_type = "np"):
    if return_type == "np":
        next_state = np.zeros(6)
        next_state[:3] = last_state[:3] + last_state[3:]*DT + 0.5*(DT**2)*control 
        #Height constrain
        next_state[2] = max(min(next_state[2], HEIGHT_BOUNDS[1]), HEIGHT_BOUNDS[0])

        next_state[3:] = last_state[3:] +  DT*control
        #Velocity constrains
        next_state[3:] = np.minimum(np.minimum(next_state[3:], VELO_BOUNDS[:,1]), VELO_BOUNDS[:, 0])
    elif return_type == "casadi":
        next_state =  ca.horzcat(*[
                last_state[:3] + last_state[3:]*DT + 0.5*control*(DT**2),
                last_state[3:] + control*DT
                ])
    else:
        next_state = None 
        print("Compute next state: Invalid return type")
    return next_state
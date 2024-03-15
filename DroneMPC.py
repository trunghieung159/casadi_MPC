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
        self.state_predicts[N_PREDICT, :] = self.state_predicts[N_PREDICT - 1, :] + \
                                                np.concatenate([self.state_predicts[N_PREDICT -1, 3:],
                                                self.control_predicts[N_PREDICT - 1, :]])*DT
        

    def setupController(self):
        '''Set up init state, constrains for predictions'''
        # Predictive length
        self.opti = ca.Opti()
        # states and controls variable 
        self.opt_controls = self.opti.variable(N_PREDICT, self.n_control)
        self.opt_states = self.opti.variable(N_PREDICT+1, self.n_state)
        f = lambda x_, u_: ca.horzcat(*[
            x_[3:] + u_ * DT / 2,
            u_
        ])

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
            next_state = self.opt_states[i, :] \
                    + f(self.opt_states[i, :], self.opt_controls[i, :])*DT
            self.opti.subject_to(self.opt_states[i+1, :] == next_state)

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
        opts_setting = {'ipopt.max_iter': 1e5,
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
        c_nav = self.costNavigation(opt_states)
        c_spe = self.costSpeed(opt_states)
        c_obs = self.costObstacle(opt_states, known_obs)
        total = W_sep*c_sep + W_spe*c_spe + W_nav*c_nav + W_obs*c_obs + W_u*c_u
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
        for j in range(N_UAV):
            if j == self.index:
                continue
            for i in range(1, N_PREDICT + 1): 
                pos_rel = drones[j].state_predicts[i,:3] - traj[i,:3].T
                cost = (ca.mtimes([pos_rel.T,pos_rel]) - DREF**2)**2
                cost_sep += cost
        # print("sep: ", cost_sep.shape)
        return cost_sep / (N_UAV-1) / N_PREDICT

    def costNavigation(self, traj):
        cost_nav = 0
        for i in range(1, N_PREDICT + 1):
            vel = traj[i,3:]
            d = ca.dot(vel.T, UREF)
            cost = (d**3-VREF**3)**2
            cost_nav += cost 
        # print("nav: ", cost_dir.shape)
        return cost_nav / N_PREDICT
    
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
                cost = 1 / (drone_to_edge_norm**2 - DRONE_R**2)   
                cost_obs += cost
        return cost_obs / N_PREDICT
    
    def __update_known_obs(self, known_obs):
        '''Update known obstacles by sensor'''
        for i in range(OBSTACLES.shape[0]):
            distance = np.linalg.norm(self.state[:2] - OBSTACLES[i, :2])
            if distance - OBSTACLES[i, 2] < SENSOR_R:
                known_obs.add(i) 


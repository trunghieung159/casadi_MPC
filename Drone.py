import numpy as np
import math
import time
import casadi as ca

from config import *

class Drone:
    def __init__(self, index:int, state:np.array, n_predict:int, radius):
        self.index = index
        
        # Drone state and control
        self.time_stamp = 0.0
        self.step = 0
        self.state = state
        self.control = np.array([0.0, 0.0, 0.0])

        self.n_state = 6
        self.n_control = 3

        # Drone radius
        self.radius = radius

        # Drone control bounds
        self.control_max = np.array([2.0, 2.0, 1.0])
        self.control_min = np.array([-2.0,-2.0,-1.0])
        
        # State predictions
        # History predictive steps
        self.n_predict = n_predict
        self.state_predicts = np.zeros((self.n_predict+1, self.n_state))
        for i in range(self.n_predict + 1):
            self.state_predicts[i, :] = self.state

        self.controls_prediction = np.zeros((n_predict, self.n_control))
        # Store drone path
        self.path = [np.concatenate([[self.time_stamp], self.state, self.control])]

    def updateState(self, control:np.array, DT:float):
        """
        Computes the states of drone after applying control signals
        Update state predictions
        """
        
        # Update
        position = self.state[:3]
        velocity = self.state[3:]

        avg_velo = velocity + control*DT/2
        next_position = position + avg_velo*DT

        # next_position = position + velocity*DT
        next_velocity = velocity + control*DT

        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp += DT
        self.step += 1
        
        # Store
        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control]))

        #Update state predict
        for i in range(self.n_predict):
            self.state_predicts[i, :] = self.state_predicts[i+1, :] 
        for i in range(self.n_predict - 1):
            self.controls_prediction[i, :] = self.controls_prediction[i+1, :]
        self.controls_prediction[self.n_predict - 1, :] = np.array([0, 0, 0]) 

    def setupController(self, drones):
       
        # Predictive length
        self.opti = ca.Opti()
        # states and controls variable 
        self.opt_controls = self.opti.variable(self.n_predict, self.n_control)
        self.opt_states = self.opti.variable(self.n_predict+1, self.n_state)

        f = lambda x_, u_: ca.horzcat(*[
            x_[3:] + u_ * DT / 2,
            u_
        ])

        # initial condition
        self.opt_start = self.opti.parameter(self.n_state)

        #step-to-step constrains
        self.opti.subject_to(self.opt_states[0, :] == self.opt_start.T)
        for i in range(self.n_predict):
            x_next = self.opt_states[i, :] \
                    + f(self.opt_states[i, :], self.opt_controls[i, :])*DT
            self.opti.subject_to(self.opt_states[i+1, :] == x_next)
        
        #velocity constrain
        self.opti.subject_to(self.opti.bounded(VELO_BOUNDS[0, 0], self.opt_states[:, 3], VELO_BOUNDS[0, 1]))
        self.opti.subject_to(self.opti.bounded(VELO_BOUNDS[1, 0], self.opt_states[:, 4], VELO_BOUNDS[1, 1]))
        self.opti.subject_to(self.opti.bounded(VELO_BOUNDS[2, 0], self.opt_states[:, 5], VELO_BOUNDS[2, 1]))

        # drone-to-drone distance constrains:
        self.drones_appr_x = self.opti.parameter(self.n_predict, NUM_UAV-1)
        self.drones_appr_y = self.opti.parameter(self.n_predict, NUM_UAV-1)
        index = 0
        for i in range(NUM_UAV - 1):
            if i == self.index:
                continue
            distance = ca.sqrt((self.opt_states[1:,1] - self.drones_appr_x[:, index])**2 + \
                               (self.opt_states[1:,2]- self.drones_appr_y[:, index])**2)
            covariance = 0.5 * ((i*DT)**2) * np.full(self.n_predict, AMAX)
            self.opti.subject_to(distance > 2 * ROBOT_RADIUS + covariance)
               
        #obstacle-distance constrains
        for i in range(self.n_predict+1):
            for j in range(OBSTACLES.shape[0]):
                distance = ca.sqrt((self.opt_states[i,0]-OBSTACLES[j,0])**2 + \
                                    (self.opt_states[i,1]-OBSTACLES[j,1])**2) 
                self.opti.subject_to(distance > ROBOT_RADIUS + OBSTACLES[j,2])
        
        self.opti.subject_to(self.opti.bounded(CONTROL_BOUNDS[0, 0], self.opt_controls[:,0], CONTROL_BOUNDS[0, 1]))
        self.opti.subject_to(self.opti.bounded(CONTROL_BOUNDS[1, 0], self.opt_controls[:,1], CONTROL_BOUNDS[1, 1]))
        self.opti.subject_to(self.opti.bounded(CONTROL_BOUNDS[2, 0], self.opt_controls[:,2], CONTROL_BOUNDS[2, 1]))

        opts_setting = {'ipopt.max_iter': 1e5,
                        'ipopt.print_level': 0,
                        'print_time': 0,
                        'ipopt.acceptable_tol': 1e-1,
                        'ipopt.acceptable_obj_change_tol': 1e-1}
        self.opti.solver('ipopt', opts_setting)  

    def computeControlSignal(self, drones):
        """
        Computes control signal for drones
        """
        # cost function
        obj = self.costFunction(self.opt_states, self.opt_controls, drones)
        self.opti.minimize(obj)

        #initials for predictions
        self.opti.set_value(self.opt_start, self.state)

        #set drones approximate position for drone-to-drone distance constrains
        index = 0
        for i in range(NUM_UAV):
            if i == self.index:
                continue
            for j in range(self.n_predict):
                self.opti.set_value(self.drones_appr_x[j, index], drones[i].state[0] + (j+1)*drones[i].state[3] * DT)
                self.opti.set_value(self.drones_appr_y[j, index], drones[i].state[1] + (j+1)*drones[i].state[4] * DT)
            index += 1
            if index > NUM_UAV-2:
                break

        # provide the initial guess of the optimization targets
        self.opti.set_initial(self.opt_states, self.state_predicts)
        self.opti.set_initial(self.opt_controls, self.controls_prediction)

        # solve the problem
        sol = self.opti.solve()
        
        ## obtain the control input
        self.controls_prediction = sol.value(self.opt_controls)
        self.state_predicts = sol.value(self.opt_states)
        return sol.value(self.opt_controls)[0,:]

    def costFunction(self, opt_states, opt_controls, drones):

        c_u = self.costControl(opt_controls)
        c_sep = self.costSeparation(opt_states, drones)
        c_dir = self.costDirection(opt_states)
        c_nav = self.costNavigation(opt_states)
        total = W_sep*c_sep + W_dir*c_dir + W_nav*c_nav + W_u*c_u

        return total

    # Minimal control signal
    def costControl(self, u):
        cost_u = 0
        for i in range(self.n_predict):
            control = u[i,:]
            cost_u += ca.mtimes([control, control.T])
        # print("u: ", cost_u.shape)
        return cost_u

    def costSeparation(self, traj, drones):
        cost_sep = 0
        for j in range(NUM_UAV):
            if j == self.index:
                continue
            for i in range(self.n_predict + 1): 
                pos_rel = drones[j].state_predicts[i,:3] - traj[i,:3].T
                cost_sep += (ca.mtimes([pos_rel.T,pos_rel]) - DREF**2)**2
        # print("sep: ", cost_sep.shape)
        return cost_sep/(NUM_UAV-1)

    def costDirection(self, traj):
        cost_dir = 0
        for i in range(self.n_predict + 1):
            vel = traj[i,3:]
            cost_dir += (VMAX - ca.mtimes(vel,UREF) / np.linalg.norm(UREF))#**2/(ca.mtimes([vel,vel.T])+1e-5)**2
        # print("dir: ", cost_dir.shape)
        return cost_dir
    
    def costNavigation(self, traj):
        cost_nav = 0
        for i in range(self.n_predict + 1):
            vel = traj[i,3:]
            cost_nav += (ca.mtimes([vel,vel.T]) - VREF**2)**2
        # print("nav: ", cost_nav.shape)
        return cost_nav
    
    # def costObstacle(...):
    #     return

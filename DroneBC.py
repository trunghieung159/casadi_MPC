import numpy as np
import math
import time

from config_init import *
from config_bc import *
class DroneBC:
    def __init__(self, index:int, state:np.array, known_obs):
        self.index = index
        
        # Drone state and control
        self.time_stamp = 0.0
        self.state = state
        self.control = np.array([0, 0, 0])
        self.step = 0

        self.n_state = 6
        self.n_control = 3
        
        # Store drone path
        self.path = [np.concatenate([[self.time_stamp], self.state, self.control])]
        self.__update_known_obs(known_obs)

    def setupController(self):
        pass

    def updateState(self, control:np.array, known_obs):
        """
        Computes the states of drone after applying control signals
        """
        self.step += 1
        # Update
        position = self.state[:3]
        velocity = self.state[3:]
        
        next_velocity = velocity + control*DT
        avg_velo = velocity + 0.5 * control *DT
        # Control signal alignment
        next_position = position + avg_velo*DT

        self.state = np.concatenate([next_position, next_velocity])
        self.control = control
        self.time_stamp = self.time_stamp + DT

        self.path.append(np.concatenate([[self.time_stamp], self.state, self.control]))
        self.__update_known_obs(known_obs)

    def computeControlSignal(self, drones, known_obs):
        """
        Computes control velocity of the copter
        """
        # state = self.state.copy()
        observed_obstacles = self.observerObstacles(known_obs)

        v_m = self.behaviorMigration()
        v_f = self.behaviorSeparation(drones)
        v_o = self.behaviorObstacle(observed_obstacles)
        v_c = self.behaviorCollision(drones)
        vel = W_nav*v_m + W_sep*v_f + W_obs*v_o + W_col*v_c

        #Heigh constrain
        next_height = self.state[2] + (self.state[5] + vel[2]*DT)*DT
        if next_height < HEIGHT_BOUNDS[0]:
            vel[2] = ((HEIGHT_BOUNDS[0] - self.state[2]) / DT - self.state[5])/DT
        elif next_height > HEIGHT_BOUNDS[1]:
            vel[2] = ((HEIGHT_BOUNDS[1] - self.state[2]) / DT - self.state[5])/DT

        if np.linalg.norm(vel) > VMAX:
            vel = vel * VMAX / np.linalg.norm(vel)
        control = 2*(vel - self.state[3:])/DT
        constrained_control = self.__apply_control_constrains(control)
        # print(constrained_control)
        return constrained_control

    def __update_known_obs(self, known_obs):
        '''Update known obstacles by sensor'''
        for i in range(OBSTACLES.shape[0]):
            distance = np.linalg.norm(self.state[:2] - OBSTACLES[i, :2])
            if distance - OBSTACLES[i, 2] < SENSOR_R:
                known_obs.add(i) 

    def __apply_control_constrains(self, control:np.array):
        smaller = control < CONTROL_BOUNDS[:, 0]
        greater = control > CONTROL_BOUNDS[:, 1]
        constrained_control = np.zeros(self.n_control)
        for i in range(self.n_control):
            if smaller[i]:
                constrained_control[i] = CONTROL_BOUNDS[i, 0]
            elif greater[i]:
                constrained_control[i] = CONTROL_BOUNDS[i, 1]
            else:
                constrained_control[i] = control[i]
        return constrained_control
        
    @staticmethod
    def behaviorMigration():
        return VREF*UREF
    
    def behaviorSeparation(self, drones):
        v_s = np.zeros(self.n_control)
        for i in self.neighbor_indices:
            if i == self.index:
                continue
            pos_rel = drones[i].state[:3]-self.state[:3]
            dir = pos_rel/np.linalg.norm(pos_rel)
            # v_s += (drones[i].state[:3]-self.state[:3]) - DREF*dir
            v_s += (np.linalg.norm(pos_rel)-DREF)*dir
        return v_s

    def behaviorObstacle(self, obstacles):
        vel_obs = np.zeros(self.n_control)
        if obstacles.shape[0] == 0:
            return vel_obs
        for i in range(obstacles.shape[0]):
            obs = obstacles[i]
            obs_rel = self.state[:2] - obs[:2]
            rs = np.linalg.norm(obs_rel)
            dir = np.concatenate([obs_rel,[0]])/np.linalg.norm(rs)
            # vel_obs += dir*np.exp(-BETA*(rs-DRONE_R-obs[2]))/(rs-DRONE_R-obs[2])
            vel_obs += math.exp(rs - (DRONE_R+obs[2]))/(rs - (DRONE_R+obs[2]))*dir
        return vel_obs/obstacles.shape[0]
    
    def behaviorCollision(self, drones):
        vel_col = np.zeros(self.n_control)
        for j in self.neighbor_indices:
            if j == self.index:
                continue
            pos_rel = drones[j].state[:3] - self.state[:3]
            rs = np.linalg.norm(pos_rel)
            if rs >= DREF:
                continue
            dir = -pos_rel/rs
            # vel_col += dir*(1 - np.tanh(BETA*(rs - 2*DRONE_R)))/2
            # vel_col += -dir*np.exp(-BETA*(rs-2*DRONE_R))/(rs-2*DRONE_R)
            vel_col += math.exp(rs - 2*DRONE_R)/(rs - 2*DRONE_R)*dir
        vel_col /= (N_UAV-1)
        return vel_col

    def update_neighbors(self, drones):
        distance_list = []
        for i in range(N_UAV):
            dis = np.linalg.norm(self.state[:3] - drones[i].state[:3])
            distance_list.append(dis)
        distance_list = np.array(distance_list)
        neighbor_set = np.argsort(distance_list)[1:1+N_NEIGHBOR]
        self.neighbor_indices = neighbor_set

    def observerObstacles(self, known_obs):
        observed_obstacles = []
        for j in known_obs:
            obs = OBSTACLES[j,:]
            if np.linalg.norm(self.state[:2]-obs[:2]) <= SENSOR_R:
                observed_obstacles.append(obs)
            # observed_obstacles.append(obs)
        return np.array(observed_obstacles)
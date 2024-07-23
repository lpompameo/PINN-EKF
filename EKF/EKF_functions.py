'''
EKF functions
'''
#%% Import libraries
import numpy as np
from filterpy.common import Q_discrete_white_noise
#%% Variables
class variables:
    def __init__(self) -> None:
        pass
    def create_dictionary_from_tuple(self, tup):
        # Ensure that the tuple contains at least one set of strings as its first element
        if not isinstance(tup, tuple) or len(tup) == 0 or not isinstance(tup[0], list) or not all(isinstance(s, str) for s in tup[0]):
            raise ValueError("La tupla deve contenere un set di stringhe come primo elemento.")

        # Create an empty dictionary
        result_dict = {}

        # Iterate over the list of strings and add each string as a key to the dictionary
        for index, name in enumerate(tup[0]):
            result_dict[name] = index

        return result_dict

    def state(self, *kwargs) -> dict:
        return self.create_dictionary_from_tuple(kwargs)

    def measure(self, *kwargs) -> dict:
        return self.create_dictionary_from_tuple(kwargs)    
    

#%% Filter functions
class filter_functions:

    def __init__(self, sV, mV, fs) -> None:

        self.dt = 1/fs
        self.var = variables()
        self.x = self.var.state(sV)
        self.nS = len(self.x)
        self.z = self.var.measure(mV)
        self.nM = len(self.z)


    def f(self, prevState) -> np.array:

        curState = np.zeros((self.nS, 1))

        curState[self.x['ax']] = prevState[self.x['ax']]
        curState[self.x['ay']] = prevState[self.x['ay']]
        curState[self.x['az']] = prevState[self.x['az']]
        curState[self.x['px']] = prevState[self.x['px']] + self.dt * prevState[self.x['vx']] +\
                                    0.5 * (self.dt **2) * prevState[self.x['ax']]
        curState[self.x['py']] = prevState[self.x['py']] + self.dt * prevState[self.x['vy']] +\
                                    0.5 * (self.dt **2) * prevState[self.x['ay']]
        curState[self.x['pz']] = prevState[self.x['pz']] + self.dt * prevState[self.x['vz']] +\
                                    0.5 * (self.dt **2) * prevState[self.x['az']]
        curState[self.x['vx']] = prevState[self.x['vx']] + self.dt * prevState[self.x['ax']]
        curState[self.x['vy']] = prevState[self.x['vy']] + self.dt * prevState[self.x['ay']]
        curState[self.x['vz']] = prevState[self.x['vz']] + self.dt * prevState[self.x['az']]
        curState[self.x['phi_dot']] = prevState[self.x['phi_dot']]
        curState[self.x['theta_dot']] = prevState[self.x['theta_dot']]
        curState[self.x['psi_dot']] = prevState[self.x['psi_dot']]
        curState[self.x['phi']] = prevState[self.x['phi']] + self.dt * prevState[self.x['phi_dot']]
        curState[self.x['theta']] = prevState[self.x['theta']] + self.dt * prevState[self.x['theta_dot']]
        curState[self.x['psi']] = prevState[self.x['psi']] + self.dt * prevState[self.x['psi_dot']]

        return curState
    
    def F(self) -> np.array:

        M = np.eye(self.nS, self.nS)

        M[self.x['px'], self.x['vx']] = self.dt
        M[self.x['px'], self.x['ax']] = 0.5 * (self.dt **2)

        M[self.x['py'], self.x['vy']] = self.dt
        M[self.x['py'], self.x['ay']] = 0.5 * (self.dt **2)

        M[self.x['pz'], self.x['vz']] = self.dt
        M[self.x['pz'], self.x['az']] = 0.5 * (self.dt **2)

        M[self.x['vx'], self.x['ax']] = self.dt
        M[self.x['vy'], self.x['ay']] = self.dt
        M[self.x['vz'], self.x['az']] = self.dt

        M[self.x['phi'], self.x['phi_dot']] = self.dt
        M[self.x['theta'], self.x['theta_dot']] = self.dt
        M[self.x['psi'], self.x['psi_dot']] = self.dt

        return M

    def Q(self, sigma_acc, sigma_gyro) -> np.array:

        M = np.zeros((self.nS, self.nS))

        q5 = Q_discrete_white_noise(dim = 3, dt = self.dt, 
                                    var = sigma_acc[0] **2, block_size = 1)        
        #py, vy, ay
        q6 = Q_discrete_white_noise(dim = 3, dt = self.dt, 
                                    var = sigma_acc[1] **2, block_size = 1)
        #pz, vz, az
        q1 = Q_discrete_white_noise(dim = 3, dt = self.dt, 
                                    var = sigma_acc[2] **2, block_size = 1)
        #phi, phi_dot
        q2 = Q_discrete_white_noise(dim = 2, dt = self.dt, 
                                    var = sigma_gyro[0], block_size = 1)
        #theta, theta_dot
        q3 = Q_discrete_white_noise(dim = 2, dt = self.dt, 
                                    var = sigma_gyro[1], block_size = 1)
        #psi, psi_dot
        q4 = Q_discrete_white_noise(dim = 2, dt = self.dt, 
                                    var = sigma_gyro[2], block_size = 1)
        
        M[self.x['px'], self.x['px']] = q5[0,0] * (self.dt **4) * 0.25
        M[self.x['vx'], self.x['vx']] = q5[1,1] * (self.dt **2)
        M[self.x['ax'], self.x['ax']] = q5[2,2]

        M[self.x['px'], self.x['vx']] = q5[0,1] * self.dt
        M[self.x['vx'], self.x['px']] = q5[1,0] * self.dt
        M[self.x['px'], self.x['ax']] = q5[0,2] * (self.dt **2) * 0.5
        M[self.x['ax'], self.x['px']] = q5[2,0] * (self.dt **2) * 0.5
        M[self.x['vx'], self.x['ax']] = q5[1,2] * self.dt
        M[self.x['ax'], self.x['vx']] = q5[2,1] * self.dt

        M[self.x['py'], self.x['py']] = q6[0,0] * (self.dt **4) * 0.25
        M[self.x['vy'], self.x['vy']] = q6[1,1] * (self.dt **2)
        M[self.x['ay'], self.x['ay']] = q6[2,2]

        M[self.x['py'], self.x['vy']] = q6[0,1] * self.dt
        M[self.x['vy'], self.x['py']] = q6[1,0] * self.dt
        M[self.x['py'], self.x['ay']] = q6[0,2] * (self.dt **2) * 0.5
        M[self.x['ay'], self.x['py']] = q6[2,0] * (self.dt **2) * 0.5
        M[self.x['vy'], self.x['ay']] = q6[1,2] * self.dt
        M[self.x['ay'], self.x['vy']] = q6[2,1] * self.dt

        M[self.x['pz'], self.x['pz']] = q1[0,0] * (self.dt **4) * 0.25
        M[self.x['vz'], self.x['vz']] = q1[1,1] * (self.dt **2)
        M[self.x['az'], self.x['az']] = q1[2,2]
        
        M[self.x['pz'], self.x['vz']] = q1[0,1] * self.dt
        M[self.x['vz'], self.x['pz']] = q1[1,0] * self.dt
        M[self.x['pz'], self.x['az']] = q1[0,2] * (self.dt **2) * 0.5
        M[self.x['az'], self.x['pz']] = q1[2,0] * (self.dt **2) * 0.5
        M[self.x['vz'], self.x['az']] = q1[1,2] * self.dt
        M[self.x['az'], self.x['vz']] = q1[2,1] * self.dt        
        M[self.x['phi'], self.x['phi']] = q2[0,0]*(self.dt**2)
        M[self.x['phi_dot'], self.x['phi_dot']] = q2[1,1]
        M[self.x['theta'], self.x['theta']] = q3[0,0]*(self.dt**2)
        M[self.x['theta_dot'], self.x['theta_dot']] = q3[1,1]
        M[self.x['psi'], self.x['psi']] = q4[0,0]*(self.dt**2)
        M[self.x['psi_dot'], self.x['psi_dot']] = q4[1,1]

        M[self.x['phi'], self.x['phi_dot']] = q2[0,1]*self.dt
        M[self.x['phi_dot'], self.x['phi']] = q2[1,0]*self.dt
        M[self.x['theta'], self.x['theta_dot']] = q3[0,1]*self.dt
        M[self.x['theta_dot'], self.x['theta']] = q3[1,0]*self.dt
        M[self.x['psi'], self.x['psi_dot']] = q4[0,1]*self.dt
        M[self.x['psi_dot'], self.x['psi']] = q4[1,0]*self.dt

        return M
    
    def h(self, predictState, prevState) -> np.array:

        curState = np.zeros((self.nM, 1))

        curState[self.z['px_m']] = prevState[self.x['px']] + prevState[self.x['vx']] * self.dt +\
                                    prevState[self.x['ax']] * (self.dt **2) * 0.5
        curState[self.z['py_m']] = prevState[self.x['py']] + prevState[self.x['vy']] * self.dt +\
                                    prevState[self.x['ay']] * (self.dt **2) * 0.5
        curState[self.z['pz_m']] = prevState[self.x['pz']] + prevState[self.x['vz']] * self.dt +\
                                    prevState[self.x['az']] * (self.dt **2) * 0.5 
        curState[self.z['vx_m']] = (predictState[self.x['vx']] - prevState[self.x['vx']]) / self.dt
        curState[self.z['vy_m']] = (predictState[self.x['vy']] - prevState[self.x['vy']]) / self.dt
        curState[self.z['vz_m']] = (predictState[self.x['vz']] - prevState[self.x['vz']]) / self.dt 
        curState[self.z['ax_m']] = (predictState[self.x['vx']] - prevState[self.x['vx']]) / self.dt
        curState[self.z['ay_m']] = (predictState[self.x['vy']] - prevState[self.x['vy']]) / self.dt
        curState[self.z['az_m']] = (predictState[self.x['vz']] - prevState[self.x['vz']]) / self.dt 
        curState[self.z['phi_dot_m']] = (predictState[self.x['phi']] - prevState[self.x['phi']]) / self.dt
        curState[self.z['theta_dot_m']] = (predictState[self.x['theta']] - prevState[self.x['theta']]) / self.dt
        curState[self.z['psi_dot_m']] = (predictState[self.x['psi']] - prevState[self.x['psi']]) / self.dt
        
        return curState
    
    def H(self) -> np.array:

        M = np.zeros((self.nM, self.nS))

        M[self.z['px_m'], self.x['px']] = 1
        M[self.z['py_m'], self.x['py']] = 1
        M[self.z['pz_m'], self.x['pz']] = 1
        M[self.z['px_m'], self.x['vx']] = self.dt
        M[self.z['py_m'], self.x['vy']] = self.dt
        M[self.z['pz_m'], self.x['vz']] = self.dt
        M[self.z['px_m'], self.x['ax']] = (self.dt **2) * 0.5
        M[self.z['py_m'], self.x['ay']] = (self.dt **2) * 0.5
        M[self.z['pz_m'], self.x['az']] = (self.dt **2) * 0.5
        M[self.z['vx_m'], self.x['px']] = 1 / self.dt
        M[self.z['vy_m'], self.x['py']] = 1 / self.dt
        M[self.z['vz_m'], self.x['pz']] = 1 / self.dt
        M[self.z['ax_m'], self.x['vx']] = 1 / self.dt
        M[self.z['ay_m'], self.x['vy']] = 1 / self.dt
        M[self.z['az_m'], self.x['vz']] = 1 / self.dt
        M[self.z['phi_dot_m'], self.x['phi']] = 1 / self.dt
        M[self.z['theta_dot_m'], self.x['theta']] = 1 / self.dt
        M[self.z['psi_dot_m'], self.x['psi']] = 1 / self.dt

        return M

    def R(self, sigma_pos, sigma_vel, sigma_acc, sigma_gyro) -> np.array:

        M = np.zeros((self.nM, self.nM))

        M[self.z['px_m'], self.z['px_m']] = sigma_pos[0]
        M[self.z['py_m'], self.z['py_m']] = sigma_pos[1]
        M[self.z['pz_m'], self.z['pz_m']] = sigma_pos[2]
        M[self.z['vx_m'], self.z['vx_m']] = sigma_vel[0]
        M[self.z['vy_m'], self.z['vy_m']] = sigma_vel[1]
        M[self.z['vz_m'], self.z['vz_m']] = sigma_vel[2]
        M[self.z['ax_m'], self.z['ax_m']] = sigma_acc[0]
        M[self.z['ay_m'], self.z['ay_m']] = sigma_acc[1]
        M[self.z['az_m'], self.z['az_m']] = sigma_acc[2]
        M[self.z['phi_dot_m'], self.z['phi_dot_m']] = sigma_gyro[0]
        M[self.z['theta_dot_m'], self.z['theta_dot_m']] = sigma_gyro[1]
        M[self.z['psi_dot_m'], self.z['psi_dot_m']] = sigma_gyro[2]

        return M

#%% EKF functions
class EKF:

    def __init__(self, stateVariables, measureVariables, fs) -> None:
        
        self.ff = filter_functions(stateVariables, measureVariables, fs)
        self.dt = 1/fs 
        self.P = np.eye(self.ff.nS) * 1000
        return

 
    def measure(self, predictState, pos, vel, acc, gyro) -> np.array:

        measure = np.zeros(self.ff.nM)

        rotation_matrix = np.array(
                            [[np.cos(predictState[self.ff.x['theta']]) * np.cos(predictState[self.ff.x['psi']]), 
                            np.cos(predictState[self.ff.x['psi']]) * np.sin(predictState[self.ff.x['phi']]) *\
                            np.sin(predictState[self.ff.x['theta']]) - np.cos(predictState[self.ff.x['phi']]) *\
                            np.sin(predictState[self.ff.x['psi']]),
                            np.sin(predictState[self.ff.x['phi']]) * np.sin(predictState[self.ff.x['psi']]) +\
                            np.cos(predictState[self.ff.x['phi']]) * np.cos(predictState[self.ff.x['psi']]) *\
                            np.sin(predictState[self.ff.x['theta']])],
                            [np.cos(predictState[self.ff.x['theta']]) * np.sin(predictState[self.ff.x['psi']]), 
                            np.cos(predictState[self.ff.x['phi']]) * np.cos(predictState[self.ff.x['psi']]) +\
                            np.sin(predictState[self.ff.x['phi']]) * np.sin(predictState[self.ff.x['psi']]) *\
                            np.sin(predictState[self.ff.x['theta']]),
                            np.cos(predictState[self.ff.x['phi']]) * np.sin(predictState[self.ff.x['psi']]) *\
                            np.sin(predictState[self.ff.x['theta']]) - np.cos(predictState[self.ff.x['psi']]) *\
                            np.sin(predictState[self.ff.x['phi']])],
                            [-np.sin(predictState[self.ff.x['theta']]), 
                            np.sin(predictState[self.ff.x['phi']]) * np.cos(predictState[self.ff.x['theta']]),
                            np.cos(predictState[self.ff.x['phi']]) * np.cos(predictState[self.ff.x['theta']])]]).reshape((3,3))
        
        acc_rotated = np.dot(rotation_matrix, np.array(acc).T).T

        measure[self.ff.z['px_m']] = pos[0]
        measure[self.ff.z['py_m']] = pos[1]
        measure[self.ff.z['pz_m']] = pos[2]
        measure[self.ff.z['vx_m']] = vel[0]
        measure[self.ff.z['vy_m']] = vel[1]
        measure[self.ff.z['vz_m']] = vel[2]
        measure[self.ff.z['ax_m']] = acc_rotated[0]
        measure[self.ff.z['ay_m']] = acc_rotated[1]
        measure[self.ff.z['az_m']] = acc_rotated[2]

        measure[self.ff.z["phi_dot_m"]] = gyro[0] +\
                                        gyro[1] * np.sin(predictState[self.ff.x['phi']]) *\
                                        np.tan(predictState[self.ff.x['theta']]) + gyro[2] *\
                                        np.cos(predictState[self.ff.x['phi']]) * np.tan(predictState[self.ff.x['theta']])
        measure[self.ff.z['theta_dot_m']] = gyro[1] * np.cos(predictState[self.ff.x['phi']]) -\
                                          gyro[2] * np.sin(predictState[self.ff.x['phi']])
        measure[self.ff.z['psi_dot_m']] = gyro[1] * np.sin(predictState[self.ff.x['phi']]) /\
                                        np.cos(predictState[self.ff.x['theta']]) + gyro[2] *\
                                        np.cos(predictState[self.ff.x['phi']]) / np.cos(predictState[self.ff.x['theta']])
                
        return measure


    def predict(self, prevState, sigma_acc, sigma_gyro) -> np.array:

        predictState = self.ff.f(prevState)
        F = self.ff.F()
        self.P = np.dot(np.dot(F, self.P), F.T) + self.ff.Q(sigma_acc, sigma_gyro)

        return predictState


    def correct(self, predictState, prevState, z_measure, sigma_pos, sigma_vel, sigma_acc, sigma_gyro) -> np.array:
        
        H = self.ff.H()

        h_x = self.ff.h(predictState, prevState)
        y = np.subtract(z_measure, h_x)
        
        S = np.dot(H, np.dot(self.P, H.T)) + self.ff.R(sigma_pos, sigma_vel, sigma_acc, sigma_gyro)
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))

        curState = predictState + np.dot(K, y)

        I = np.eye(H.shape[1])
        self.P = np.dot(np.dot((I - np.dot(K, H)), self.P), (I - np.dot(K, H)).T) +\
                 np.dot(np.dot(K, self.ff.R(sigma_pos, sigma_vel, sigma_acc, sigma_gyro)), K.T)

        return curState
    
    def rotate(self, curState) -> np.array:
        
        rotatedState = curState

        accel = np.array([[curState[self.ff.x['ax']], curState[self.ff.x['ay']], 
                          curState[self.ff.x['az']]]]).reshape((3,1))
        
        rotation_matrix = np.array(
                            [[np.cos(curState[self.ff.x['theta']]) * np.cos(curState[self.ff.x['psi']]), 
                            np.cos(curState[self.ff.x['psi']]) * np.sin(curState[self.ff.x['phi']]) *\
                            np.sin(curState[self.ff.x['theta']]) - np.cos(curState[self.ff.x['phi']]) *\
                            np.sin(curState[self.ff.x['psi']]),
                            np.sin(curState[self.ff.x['phi']]) * np.sin(curState[self.ff.x['psi']]) +\
                            np.cos(curState[self.ff.x['phi']]) * np.cos(curState[self.ff.x['psi']]) *\
                            np.sin(curState[self.ff.x['theta']])],
                            [np.cos(curState[self.ff.x['theta']]) * np.sin(curState[self.ff.x['psi']]), 
                            np.cos(curState[self.ff.x['phi']]) * np.cos(curState[self.ff.x['psi']]) +\
                            np.sin(curState[self.ff.x['phi']]) * np.sin(curState[self.ff.x['psi']]) *\
                            np.sin(curState[self.ff.x['theta']]),
                            np.cos(curState[self.ff.x['phi']]) * np.sin(curState[self.ff.x['psi']]) *\
                            np.sin(curState[self.ff.x['theta']]) - np.cos(curState[self.ff.x['psi']]) *\
                            np.sin(curState[self.ff.x['phi']])],
                            [-np.sin(curState[self.ff.x['theta']]), 
                            np.sin(curState[self.ff.x['phi']]) * np.cos(curState[self.ff.x['theta']]),
                            np.cos(curState[self.ff.x['phi']]) * np.cos(curState[self.ff.x['theta']])]]).reshape((3,3))
        
        accel_rotated = np.dot(rotation_matrix.T, accel).T #1x3

        rotatedState[self.ff.x['ax']] = accel_rotated[:,0]
        rotatedState[self.ff.x['ay']] = accel_rotated[:,1]
        rotatedState[self.ff.x['az']] = accel_rotated[:,2]

        return rotatedState
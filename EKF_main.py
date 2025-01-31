'''
Import of the PINN output and execution of the EKF
@authors: Mariapia De Rosa, Laura Pompameo
'''
#%% Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.signal import *
from EKF_functions import *

#%% Geometry data
# Path configuration
geometry_data = pd.read_excel(
    Path(os.getenv(
        "path_flegrea") + "/Test" + "/Kalman Filter" +\
                "/Tabulati Licola - Montesanto.xlsx"))
geometry_data = geometry_data.loc[::-1].reset_index(drop=True)

# Data collection
longitude = np.array([float(i.replace("° E", "")) for i in geometry_data["Longitudine [°]"]])
latitude = np.array([float(i.replace("° N", "")) for i in geometry_data["Latitudine [°]"]])
curvature = geometry_data["Curvatura [1/km]"] * 1e-3 #now in 1/m
trasv_level = geometry_data["Liv.Trasv. [mm]"] * 1e-3 #now in m

#%% Inizializzo variabili di stato e di misura
var = variables()
ekf = EKF(stateVariables = ['px', 'py', 'pz', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 
                            'phi_dot', 'theta_dot', 'psi_dot', 
                            'phi', 'theta', 'psi'], 
        measureVariables = ['px_m', 'py_m', 'pz_m',
                            'vx_m', 'vy_m', 'vz_m',
                            'ax_m', 'ay_m', 'az_m', 
                            'phi_dot_m', 'theta_dot_m', 'psi_dot_m'], 
        fs = 25)

#%% Load data from IMU and from your PINN output
ekf_inputs_df = pd.read_pickle(Path(os.getenv(
            "path_flegrea") + "/pos_vel.pkl"))

#%% Change name of columns to distinguish PINN data from GPS data
ekf_inputs_df.rename(
    columns = {'gps_x_vel': 'pinn_x_vel', 
               'gps_y_vel': 'pinn_y_vel',
               'gps_z_vel': 'pinn_z_vel',
               'gps_x_pos': 'pinn_x_pos',
               'gps_y_pos': 'pinn_y_pos',
               'gps_z_pos': 'pinn_z_pos'},
    inplace = True,
)

#%% # Cut data based on latitude and longitude
start_geometrydata = int(np.percentile(np.where(
    (np.abs(longitude - ekf_inputs_df['gps_longitude'][ekf_inputs_df.index[0]]) < 1e-4) &
    (np.abs(latitude - ekf_inputs_df['gps_latitude'][ekf_inputs_df.index[0]]) < 1e-4))[0], 50))
stop_geometrydata = int(np.percentile(np.where(
    (np.abs(longitude - ekf_inputs_df['gps_longitude'][ekf_inputs_df.index[-1]]) < 1e-4) &
    (np.abs(latitude - ekf_inputs_df['gps_latitude'][ekf_inputs_df.index[-1]]) < 1e-4))[0], 50))

#%% State Initializzation
nbSamples = len(ekf_inputs_df) - 1
state = np.zeros((nbSamples, ekf.ff.nS)) 

state[0, ekf.ff.x['ax']] = ekf_inputs_df['ax_data'][0]
state[0, ekf.ff.x['ay']] = ekf_inputs_df['ay_data'][0]
state[0, ekf.ff.x['az']] = ekf_inputs_df['az_data'][0]

state[0, ekf.ff.x['vx']] = 0.
state[0, ekf.ff.x['vy']] = 0.
state[0, ekf.ff.x['vz']] = 0.

state[0, ekf.ff.x['px']] = 0.
state[0, ekf.ff.x['py']] = 0.
state[0, ekf.ff.x['pz']] = 0.

state[0, ekf.ff.x['phi']] = np.arctan(np.mean(ekf_inputs_df['ay_data']) /\
                                    np.mean(ekf_inputs_df['az_data']))
state[0, ekf.ff.x['theta']] = np.arctan(np.mean(ekf_inputs_df['ax_data']) /\
                                        (np.sqrt((np.mean(ekf_inputs_df['ay_data']) **2) +\
                                        (np.mean(ekf_inputs_df['az_data']) **2))))
state[0, ekf.ff.x['psi']] = 0
state[0, ekf.ff.x['phi_dot']] = ekf_inputs_df['gyrox_data'][0] +\
                                ekf_inputs_df['gyroy_data'][0] * np.sin(state[0, ekf.ff.x['phi']]) *\
                                np.tan(state[0, ekf.ff.x['theta']]) + ekf_inputs_df['gyroz_data'][0] *\
                                np.cos(state[0, ekf.ff.x['phi']]) * np.tan(state[0, ekf.ff.x['theta']])
state[0, ekf.ff.x['theta_dot']] = ekf_inputs_df['gyroy_data'][0] * np.cos(state[0, ekf.ff.x['phi']]) -\
                                ekf_inputs_df['gyroz_data'][0] * np.sin(state[0, ekf.ff.x['phi']])
state[0, ekf.ff.x['psi_dot']] = ekf_inputs_df['gyroy_data'][0] * np.sin(state[0, ekf.ff.x['phi']]) /\
                                np.cos(state[0, ekf.ff.x['theta']]) + ekf_inputs_df['gyroz_data'][0] *\
                                np.cos(state[0, ekf.ff.x['phi']]) / np.cos(state[0, ekf.ff.x['theta']])

#%% Iteration on the samples
for k in range(1, nbSamples):

    #Predict
    predictedState = (ekf.predict(prevState = state[[k-1][:]].T,
                                  sigma_acc = np.std([ekf_inputs_df['ax_data'], ekf_inputs_df['ay_data'], ekf_inputs_df['az_data']], axis=1), 
                                  sigma_gyro = np.std([ekf_inputs_df['gyrox_data'], ekf_inputs_df['gyroy_data'], ekf_inputs_df['gyroz_data']], axis=1)
                                  )).T

    #Correct
    z_measure = np.array(ekf.measure(predictState = predictedState.T, 
                                     pos = [ekf_inputs_df['pinn_x_pos'][k], ekf_inputs_df['pinn_y_pos'][k], ekf_inputs_df['pinn_z_pos'][k]],
                                     vel = [ekf_inputs_df['pinn_x_vel'][k], ekf_inputs_df['pinn_y_vel'][k], ekf_inputs_df['pinn_z_vel'][k]],
                                     acc = [ekf_inputs_df['ax_data'][k], ekf_inputs_df['ay_data'][k], ekf_inputs_df['az_data'][k]], 
                                     gyro = [ekf_inputs_df['gyrox_data'][k], ekf_inputs_df['gyroy_data'][k], ekf_inputs_df['gyroz_data'][k]]
                                     )).reshape((ekf.ff.nM,1))

    state[[k][:]] = (ekf.correct(predictState = predictedState.T, 
                                 prevState = state[[k-1][:]].T, 
                                 z_measure = z_measure,
                                 sigma_pos = np.std([ekf_inputs_df['pinn_x_pos'], ekf_inputs_df['pinn_y_pos'], ekf_inputs_df['pinn_z_pos']], axis=1),
                                 sigma_vel = np.std([ekf_inputs_df['pinn_x_vel'], ekf_inputs_df['pinn_y_vel'], ekf_inputs_df['pinn_z_vel']], axis=1),
                                 sigma_acc = np.std([ekf_inputs_df['ax_data'], ekf_inputs_df['ay_data'], ekf_inputs_df['az_data']], axis=1), 
                                 sigma_gyro = np.std([ekf_inputs_df['gyrox_data'], ekf_inputs_df['gyroy_data'], ekf_inputs_df['gyroz_data']], axis=1)
                                 )).T

#%% Rotate acc, vel and pos from inertial frame to body frame
### in order to compare them with the IMU and GPS data
for k in range(0,nbSamples):

    state[[k][:]] = (ekf.rotate(state[[k][:]].T)).T

#%% Comparison data
# Cross level
geometrydata_livtr = np.array(trasv_level[start_geometrydata:stop_geometrydata])
livtr = np.interp(np.linspace(0, stop_geometrydata - start_geometrydata - 1, 
                                len(ekf_inputs_df)-1), 
                np.arange(len(geometrydata_livtr)), geometrydata_livtr)

#Curvature
geodata_curvatura = np.array(curvature[start_geometrydata:stop_geometrydata])
curve = np.interp(np.linspace(0, stop_geometrydata - start_geometrydata - 1,
                                len(ekf_inputs_df)-1),
                np.arange(len(geodata_curvatura)), geodata_curvatura)

#%% Read gps data pickle file
gps_df = pd.read_pickle(Path(os.getenv(
            "path_flegrea") + "/gps_data.pkl"))

#%% Plot the results

#Position
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title('Position x-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), gps_df['gps_x_pos'], label = 'px GPS')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['pinn_x_pos'], label = 'px PINN')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.subplot(1, 3, 2) 
plt.title('Position y-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), gps_df['gps_y_pos'], label = 'py GPS')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['pinn_y_pos'], label = 'py PINN')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.subplot(1, 3, 3)
plt.title('Position z-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), gps_df['gps_z_pos'], label = 'pz GPS')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['pinn_z_pos'], label = 'pz PINN')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.tight_layout()  
plt.savefig('Position.png', dpi = 300)
plt.show()

#Speed
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title('Speed x-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), gps_df['gps_x_vel'], label = 'vx GPS')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['pinn_x_vel'], label = 'vx PINN')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Speed [$m/s$]')
plt.subplot(1, 3, 2) 
plt.title('Speed y-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), gps_df['gps_y_vel'], label = 'vy GPS')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['pinn_y_vel'], label = 'vy PINN')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Speed [$m/s$]')
plt.subplot(1, 3, 3)
plt.title('Speed z-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), gps_df['gps_z_vel'], label = 'vz GPS')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['pinn_z_vel'], label = 'vz PINN')
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Speed [$m/s$]')
plt.tight_layout()  
plt.savefig('Speed.png', dpi = 300)
plt.show()

#%%Acceleration
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.title('Acceleration x-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), state[:, ekf.ff.x['ax']], label = 'ax EKF')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['ax_data'], label = 'ax IMU data', zorder = 1)
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [$m/s^2$]')
plt.subplot(1, 3, 2) 
plt.title('Acceleration y-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), state[:, ekf.ff.x['ay']], label = 'ay EKF')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['ay_data'], label = 'ay IMU data', zorder = 1)
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [$m/s^2$]')
plt.subplot(1, 3, 3)
plt.title('Acceleration z-axis', fontsize = 14)
plt.plot((gps_df['tsImu'].values), state[:, ekf.ff.x['az']], label = 'az EKF')
plt.plot((gps_df['tsImu'].values), ekf_inputs_df['az_data'], label = 'az IMU data', zorder = 1)
plt.grid()
plt.legend()
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [$m/s^2$]')
plt.tight_layout()  
plt.savefig('Acceleration.png', dpi = 300)
plt.show()

#Roll, pitch, yaw
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title("Roll")
plt.plot(state[:, ekf.ff.x['phi']], label = 'phi')
plt.legend()
plt.subplot(1, 3, 2) 
plt.title("Pitch")
plt.plot(state[:, ekf.ff.x['theta']], label = 'theta')
plt.legend()
plt.subplot(1, 3, 3)
plt.title("Yaw ")
plt.plot(state[:, ekf.ff.x['psi']], label = 'psi')
plt.legend()
plt.tight_layout()  
plt.grid()
plt.show()

#Roll rate, pitch rate, yaw rate
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title("Roll rate")
plt.plot(state[:, ekf.ff.x['phi_dot']], label = 'phi_dot filtrata')
plt.plot(ekf_inputs_df['gyrox_data'], label = 'gyrox IMU', zorder = 1)
plt.legend()
plt.subplot(1, 3, 2) 
plt.title("Pitch rate")
plt.plot(state[:, ekf.ff.x['theta_dot']], label = 'theta_dot filtrata')
plt.plot(ekf_inputs_df['gyrox_data'], label = 'gyroy IMU', zorder = 1)
plt.legend()
plt.subplot(1, 3, 3)
plt.title("Yaw rate")
plt.plot(state[:, ekf.ff.x['psi_dot']], label = 'psi_dot filtrata')
plt.plot(ekf_inputs_df['gyrox_data'], label = 'gyroz IMU', zorder = 1)
plt.legend()
plt.tight_layout()  
plt.grid()
plt.show()
# %%

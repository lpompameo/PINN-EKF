'''
GPS data visualization
@authors: Mariapia De Rosa, Laura Pompameo
'''
#%% Import libraries
import os
import numpy as np
import pandas as pd
import pymap3d as pm
import tensorflow as tf
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

#%% IMU and GPS data
# Path configuration
folder = 'T1FLAV202208171759'
imufileName = folder + "S6_imu.parquet" #for Flegrea
gpsfileName = folder + "S6_gps.parquet" #for Flegrea
dataPath = Path(os.getenv("path_flegrea") + "/Test" +\
                            "/Kalman Filter" + "/dati grezzi")

# Data collection
imudata = pd.read_parquet(dataPath / f"{folder}" / f"{imufileName}")
gpsdata = pd.read_parquet(dataPath / f"{folder}" / f"{gpsfileName}")

# Timestamps for the two sensors
tsImu = imudata["timestamp_ns"]
tsGps = gpsdata["timestamp_ns"]

#%% Fix heading angle to make it continuous 
# (this should not be the case for everyone 
# if you have well-collected data)
indexNOTFixed = np.where(gpsdata['gpsFixOK'] == 0)[0]
indexFixed = np.where(gpsdata['gpsFixOK'] == 1)[0]
if (indexNOTFixed[0]).size > 0:
    index_start = np.max([indexNOTFixed[0]-30, 0])
    index_stop = np.min([indexNOTFixed[-1] + 1 + 30, 
                         len(gpsdata['gpsFixOK'])])
    index_good_start = np.max([indexFixed[0], 0])
    index_good_stop = np.min([indexFixed[-1] + 1, 
                               len(gpsdata['gpsFixOK'])]) 
do_not_change = np.concatenate(
                    [np.arange(index_good_start, index_start), 
                    np.arange(index_stop, index_good_stop)],
                    axis=0)
i = gpsdata.index[index_start]
j = gpsdata.index[index_stop]
gpsdata['headingOfMotion_deg'].loc[i:j] = np.interp(range(i, gpsdata.index[index_stop+1]),
                                                    gpsdata.index[do_not_change], 
                                                    gpsdata['headingOfMotion_deg'].loc[gpsdata.index[do_not_change]]
                                                    )
idx_heading = np.argmin(np.deg2rad(gpsdata['headingOfMotion_deg']))
gps_heading_fixed = np.deg2rad(gpsdata['headingOfMotion_deg'])
gps_heading_fixed[idx_heading:] = gps_heading_fixed[idx_heading:] + 2 * np.pi 

#%% Interpolate GPS data with the same sampling rate
gps_longitude = np.interp(tsImu, tsGps, gpsdata['longitude']).reshape((len(tsImu), 1))
gps_latitude = np.interp(tsImu, tsGps, gpsdata['latitude']).reshape((len(tsImu), 1))
gps_altitude = np.interp(tsImu, tsGps, gpsdata['heightMSL_mm'] * 1e-3).reshape((len(tsImu), 1))
gpsFixOK = np.interp(tsImu, tsGps, gpsdata["gpsFixOK"]).reshape((len(tsImu), 1))
gps_north_vel = np.interp(tsImu, tsGps, gpsdata['northVelocity_mm_s'] * 1e-3).reshape((len(tsImu), 1))
gps_east_vel = np.interp(tsImu, tsGps, gpsdata['eastVelocity_mm_s'] * 1e-3).reshape((len(tsImu), 1))
gps_up_vel = np.interp(tsImu, tsGps, -gpsdata['downVelocity_mm_s'] * 1e-3).reshape((len(tsImu), 1))       
gps_heading = np.interp(tsImu, tsGps, gps_heading_fixed).reshape((len(tsImu), 1))

#%% Normalize tsImu
times_Imu = np.reshape(
                np.array((tsImu - tsImu[tsImu.index[0]]) * 1e-9),
                newshape = (len(tsImu),1)
                )

#%% Create GPS dataframe
gps_df = pd.DataFrame(
    data = np.concatenate([times_Imu, gpsFixOK,
                            gps_longitude, gps_latitude, gps_altitude, 
                            gps_north_vel, gps_east_vel, gps_up_vel,
                            gps_heading
                            ], axis = 1), 
    columns = ['tsImu', 'gpsFixOK', 
                'gps_longitude', 'gps_latitude', 'gps_altitude', 
                'gps_north_vel', 'gps_east_vel', 'gps_up_vel', 'gps_heading']
    )

#%% Fix GPS data
indexNOTFixed = np.where(gps_df['gpsFixOK'] == 0)[0]
indexFixed = np.where(gps_df['gpsFixOK'] == 1)[0]
if (indexNOTFixed[0]).size > 0:
    index_start = np.max([indexNOTFixed[0]-300, 0])
    index_stop = np.min([indexNOTFixed[-1] + 1 + 300, len(gps_df['gpsFixOK'])])
    index_buono_start = np.max([indexFixed[0], 0])
    index_buono_stop = np.min([indexFixed[-1] + 1, len(gps_df['gpsFixOK'])]) 
    do_not_change = np.concatenate(
                [np.arange(index_buono_start, index_start), 
                    np.arange(index_stop, index_buono_stop)],
                    axis=0
                    )
    gps_df['gps_longitude'][index_start-300:index_stop+300] = np.interp(
                    range(index_start-300, index_stop+300),
                    do_not_change, 
                    gps_df['gps_longitude'][do_not_change])
    gps_df['gps_latitude'][index_start-300:index_stop+300] = np.interp(
                    range(index_start-300, index_stop+300),
                    do_not_change, 
                    gps_df['gps_latitude'][do_not_change])
    gps_df['gps_altitude'][index_start-300:index_stop+300] = np.interp(
                    range(index_start-300, index_stop+300),
                    do_not_change, 
                    gps_df['gps_altitude'][do_not_change])
    gps_df['gps_north_vel'][index_start-300:index_stop+300] = np.interp(
                    range(index_start-300, index_stop+300),
                    do_not_change, 
                    gps_df['gps_north_vel'][do_not_change])
    gps_df['gps_east_vel'][index_start-300:index_stop+300] = np.interp(
                    range(index_start-300, index_stop+300),
                    do_not_change, 
                    gps_df['gps_east_vel'][do_not_change])
    gps_df['gps_up_vel'][index_start-300:index_stop+300] = np.interp(
                    range(index_start-300, index_stop+300),
                    do_not_change, 
                    gps_df['gps_up_vel'][do_not_change])
    gps_df['gps_heading'][index_start-300:index_stop+300] = np.interp(
                    range(index_start-300, index_stop+300),
                    do_not_change, 
                    gps_df['gps_heading'][do_not_change])

# Compute geodetic coordinates
gps_east_pos, gps_north_pos, gps_up_pos = pm.geodetic2enu(
            lat = gps_df['gps_latitude'], 
            lon = gps_df['gps_longitude'], 
            h = gps_df['gps_altitude'],
            lat0 = gps_df['gps_latitude'].iloc[0], 
            lon0 = gps_df['gps_longitude'].iloc[0], 
            h0 = gps_df['gps_altitude'].iloc[0]
            ) 
gps_df['gps_north_pos'] = gps_north_pos
gps_df['gps_east_pos'] = gps_east_pos
gps_df['gps_up_pos'] = gps_up_pos

# Rotate pos, vel of heading angle
rot_matrix = tf.transpose(tf.stack([[tf.math.cos(gps_df['gps_heading']), 
                                    -tf.math.sin(gps_df['gps_heading']), 
                                    tf.zeros_like(gps_df['gps_heading'])],
                                    [tf.math.sin(gps_df['gps_heading']), 
                                    tf.math.cos(gps_df['gps_heading']),  
                                    tf.zeros_like(gps_df['gps_heading'])],
                                    [tf.zeros_like(gps_df['gps_heading']),  
                                    tf.zeros_like(gps_df['gps_heading']),  
                                    -tf.ones_like(gps_df['gps_heading'])]]))
pos_rotated = tf.matmul(rot_matrix, 
                        tf.stack([gps_df['gps_north_pos'], gps_df['gps_east_pos'], 
                                gps_df['gps_up_pos']]))[0,...]
vel_rotated = tf.matmul(rot_matrix, 
                        tf.stack([gps_df['gps_north_vel'], gps_df['gps_east_vel'], 
                                gps_df['gps_up_vel']]))[0,...]
gps_df['gps_x_pos'] = pos_rotated[0,...] 
gps_df['gps_y_pos'] = pos_rotated[1,...]          
gps_df['gps_z_pos'] = pos_rotated[2,...] 

gps_df['gps_x_vel'] = vel_rotated[0,...] 
gps_df['gps_y_vel'] = vel_rotated[1,...]    
gps_df['gps_z_vel'] = vel_rotated[2,...] 

#Split dataframe
gps_df_final = gps_df.loc[:, ['tsImu', 'gpsFixOK', 
                        'gps_longitude', 'gps_latitude', 'gps_altitude',
                        'gps_x_vel', 'gps_y_vel', 'gps_z_vel', 'gps_heading',
                        'gps_x_pos', 'gps_y_pos', 'gps_z_pos']]

gps_df_utilities = gps_df.loc[:, ['gps_north_vel', 'gps_east_vel', 'gps_up_vel', 
                        'gps_north_pos', 'gps_east_pos', 'gps_up_pos']]
            
#%% Plot 2D route
plt.figure()
plt.title('2D route')
plt.plot(gps_df_utilities['gps_east_pos'], gps_df_utilities['gps_north_pos'])
plt.show()

#%% Plot X, Y with maps under
img = plt.imread('Images/maps_without_route.png')
fig, ax = plt.subplots()
ax.plot(gps_df_utilities['gps_east_pos'], gps_df_utilities['gps_north_pos'])
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_title('Train route in ENU coordinates')
ax.imshow(img, extent=[xmin, xmax, ymin, ymax+100])

#%%

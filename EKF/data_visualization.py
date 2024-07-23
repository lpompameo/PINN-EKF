'''
GPS data visualization
'''
#%% Import libraries
import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import *
import matplotlib.pyplot as plt
try:
    from src.EKF.EKF_functions import *
    from src.EKF.preprocessing import * 
except ModuleNotFoundError:
    from preprocessing import * 
    from EKF_functions import *

#%% Import classes
fix = fixGPS_class()

#%% Dati generali
fsIMU = 25
line = "Flegrea"
linestr = "T1FLAV202"
#%% Configurazione dei path FLEGREA
dataPath = Path(os.getenv("path_flegrea") + "/Test" +\
                            "/Kalman Filter" + "/dati grezzi") 
folderList = os.listdir(os.getenv("pathFolder"))
if line == "Flegrea":
    geometry_data = pd.read_excel(
        Path(os.getenv(
            "path_flegrea") + "/Test" + "/Kalman Filter" +\
                    "/Tabulati Licola - Montesanto.xlsx"))
    geometry_data = geometry_data.loc[::-1].reset_index(drop=True)
else:
    geometry_data = pd.read_excel(
        Path(os.getenv(
            "path_flegrea") + "/Test" + "/Kalman Filter" +\
              f"/{line} 00+100-19+600.xlsx"))

#%% Geometry data
longitude = np.array([float(i.replace("° E", "")) for i in geometry_data["Longitudine [°]"]])
latitude = np.array([float(i.replace("° N", "")) for i in geometry_data["Latitudine [°]"]])
curvatura = geometry_data["Curvatura [1/km]"] * 1e-3 #ora è in 1/metri
livello_trasversale = geometry_data["Liv.Trasv. [mm]"] * 1e-3 #ora è in metri

#%% Folder selection
folder = 'T1FLAV202208171759' #cartella prefe di pia
imufileName = folder + "S6_imu.parquet" #for Flegrea
gpsfileName = folder + "S6_gps.parquet" #for Flegrea
imudata = pd.read_parquet(dataPath / f"{folder}" / f"{imufileName}")
tsImu = imudata["timestamp_ns"]
gpsdata = pd.read_parquet(dataPath / f"{folder}" / f"{gpsfileName}")
tsGps = gpsdata["timestamp_ns"]

#%% Fix heading to make it continuous
indexNOTFixed = np.where(gpsdata['gpsFixOK'] == 0)[0]
indexFixed = np.where(gpsdata['gpsFixOK'] == 1)[0]
if (indexNOTFixed[0]).size > 0:
    index_start = np.max([indexNOTFixed[0]-30, 0])
    index_stop = np.min([indexNOTFixed[-1] + 1 + 30, len(gpsdata['gpsFixOK'])])
    index_buono_start = np.max([indexFixed[0], 0])
    index_buono_stop = np.min([indexFixed[-1] + 1, len(gpsdata['gpsFixOK'])]) 

do_not_change = np.concatenate(
                    [np.arange(index_buono_start, index_start), 
                    np.arange(index_stop, index_buono_stop)],
                    axis=0)
gpsdata['headingOfMotion_deg'].loc[gpsdata.index[index_start]:gpsdata.index[index_stop]] = np.interp(
                            range(gpsdata.index[index_start], gpsdata.index[index_stop+1]),
                            gpsdata.index[do_not_change], 
                            gpsdata['headingOfMotion_deg'].loc[gpsdata.index[do_not_change]])

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
times_Imu = np.reshape(np.array((tsImu - tsImu[tsImu.index[0]]) * 1e-9),
                        (len(tsImu),1))

#%% Create gps dataframe
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

gps_df, gps_df_utilities = fix.fixGPS(gps_df)

# %% Plot GPS data after preprocessing
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title('Speed x fixed')
plt.plot(gps_df['gps_x_vel'])
plt.legend()
plt.subplot(1, 3, 2) 
plt.title('Speed y fixed')
plt.plot(gps_df['gps_y_vel'])
plt.legend()
plt.subplot(1, 3, 3)
plt.title('Speed z fixed')
plt.plot(gps_df['gps_z_vel'])
plt.legend()
plt.tight_layout()  
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title('Position x fixed')
plt.plot(gps_df['gps_x_pos'])
plt.legend()
plt.subplot(1, 3, 2) 
plt.title('Position y fixed')
plt.plot(gps_df['gps_y_pos'])
plt.legend()
plt.subplot(1, 3, 3)
plt.title('Position z fixed')
plt.plot(gps_df['gps_z_pos'])
plt.legend()
plt.tight_layout()  
plt.show()

#%% Save gps data in pickle file
# gps_df.to_pickle(Path(os.getenv(
#             "path_flegrea") + 'gps_data.pkl'), 
#             compression='infer', protocol=5, storage_options=None)

#%% Plot X,Y
plt.figure()
plt.title('2D route')
plt.plot(gps_df_utilities['gps_east_pos'], gps_df_utilities['gps_north_pos'])
plt.show()

#%% Plot X, Y with maps under
img = plt.imread(Path(os.getenv(
            "path_flegrea") + "/maps_without_route.png"))
fig, ax = plt.subplots()
ax.plot(gps_df_utilities['gps_east_pos'], gps_df_utilities['gps_north_pos'])
xmin, xmax = ax.get_xlim()
ymin, ymax = ax.get_ylim()
ax.set_title('Train route in ENU coordinates')
ax.imshow(img, extent=[xmin, xmax, ymin, ymax+100])

#%%

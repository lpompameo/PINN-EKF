'''
Initial file reading + execution of the PINN 
'''
#%% Import the libraries
import os
import sys
import socket 
import random 
import numpy as np
import tensorflow as tf
from pathlib import Path
from scipy.constants import g
import matplotlib.pyplot as plt
from fastcore.all import dict2obj 
from sklearn.preprocessing import MinMaxScaler
try:
    from PINN_functions import *
except ModuleNotFoundError:
    from src.PINN.PINN_functions import *

sys.path.append(os.path.abspath(os.path.dirname(
    os.path.abspath(os.path.dirname(
    os.path.abspath(os.path.dirname("preprocessing.py")))
    ))
))

try:
    from src.EKF.preprocessing import * 
except ModuleNotFoundError:
    from EKF.preprocessing import * 
#%% Set the type of float
DTYPE = tf.float32
# #%% GPUs

server = socket.gethostname()
print(server)
if server in ['aorus', 'modal-calc0']:
    CVD = '0'
else:
    CVD = '0'

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['CUDA_VISIBLE_DEVICES'] = CVD

SEED = 69
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)#

if CVD!='-1':
    which_gpu = f'/GPU:{CVD}'
else:
    which_gpu = '/CPU:0'

print(tf.config.get_visible_devices())

#%% Set the parameters
config = dict(
    lr = 1e-3,
    ker_reg = None,
    epochs = 5000,
    patience = 7000,
    opt = 'Adam',  #'L-BFGS-B'1
    save = False,
    print_epoch = 1000,
    plot_epoch = 10000,
    plot_intermedio = 5000,
    config_name = '12_gennaio',
    load_model = True
)

cf = dict2obj(config)

#%% Fix GPS
fix = fixGPS_class()

#%% Calibration data accelerometer in g 
calibrazione_acc = {'a_yz' : -0.00525382,
                    'a_xz' : -0.00345133,
                    'a_zy' : 0.00170025,
                    's_x'  : 1.00472521,
                    's_y'  : 0.99568991,
                    's_z'  : 0.98699466,
                    'b_x'  : 0.02389066,
                    'b_y'  : 0.02803369,
                    'b_z'  : -0.03951457}

# Calibration data gyroscope
calibrazione_gyro = {'g_yz' : 0.10000001267198236,                      
                    'g_zy' : 0.09999999053747313,                      
                    'g_xz' : 0.10000000651023393,                      
                    'g_zx' : 0.00999998266125302,                      
                    'g_xy' : 0.10000000182610527,                      
                    'g_yx' : 0.10000001014122448,
                    's_x' : 0.9999999801730876,
                    's_y' : 0.9999999782882735,
                    's_z' : 1.0000000404745477,
                    'b_x'  : -0.0028411208757018512,
                    'b_y'  : 0.010861831117082284,
                    'b_z'  : -0.007396392532099494}

#%% Calibration matrices
T_a = np.diag(np.ones(3))
T_a[0,1] = - calibrazione_acc["a_yz"]
T_a[0,2] = calibrazione_acc["a_zy"]
T_a[1,2] = - calibrazione_acc["a_xz"]

K_a = np.diag([calibrazione_acc["s_x"],
                calibrazione_acc["s_y"], 
                calibrazione_acc["s_z"]])

b_a = [calibrazione_acc["b_x"], 
       calibrazione_acc["b_y"], 
       calibrazione_acc["b_z"]]

T_g = np.diag(np.ones(3))
T_g[0,1] = -calibrazione_gyro["g_yz"]
T_g[0,2] = calibrazione_gyro["g_zy"]
T_g[1,0] = calibrazione_gyro["g_xz"]
T_g[1,2] = -calibrazione_gyro["g_zx"]
T_g[2,0] = -calibrazione_gyro["g_xy"]
T_g[2,1] = calibrazione_gyro["g_yx"]

K_g = np.diag([calibrazione_gyro["s_x"], 
               calibrazione_gyro["s_y"], 
               calibrazione_gyro["s_z"]])

b_g = [calibrazione_gyro["b_x"], 
       calibrazione_gyro["b_y"], 
       calibrazione_gyro["b_z"]]

#%% Global variables
fsIMU = 25
line = "Flegrea"
linestr = "T1FLAV202"

#%% Data path 
dataPath = Path(os.getenv("path_flegrea") + "/Test" +\
                            "/Kalman Filter" + "/dati grezzi") 
folderList = os.listdir(os.getenv("pathFolder"))
if line == "Flegrea":
    geometry_data = pd.read_excel(
        Path(os.getenv(
            "path_flegrea") + "/Test" + "/Kalman Filter" +\
                    "/Tabulati Licola - Montesanto.xlsx"))
else:
    geometry_data = pd.read_excel(
        Path(os.getenv(
            "path_analyst") + "/Test" + "/Kalman Filter" + f"/{line} 00+100-19+600.xlsx"))
    

#%% Geometry data
longitude = np.array([float(i.replace("° E", "")) for i in geometry_data["Longitudine [°]"]])
latitude = np.array([float(i.replace("° N", "")) for i in geometry_data["Latitudine [°]"]])
curvatura = geometry_data["Curvatura [1/km]"] * 1e-3 #1/m
livello_trasversale = geometry_data["Liv.Trasv. [mm]"] * 1e-3 #m

# Folder selection
folder = 'T1FLAV202208171759' 

# File selection
imufileName = folder + "S6_imu.parquet"
gpsfileName = folder + "S6_gps.parquet"

# Imudata and gpsdata
imudata = pd.read_parquet(dataPath / f"{folder}" / f"{imufileName}")
gpsdata = pd.read_parquet(dataPath / f"{folder}" / f"{gpsfileName}")

# Time selection
tsImu = imudata["timestamp_ns"]
tsGps = gpsdata["timestamp_ns"]
# Calibrate imudata
acc = np.vstack((imudata.accelerometer_g_x, imudata.accelerometer_g_y,
                                imudata.accelerometer_g_z)).T

gyro = np.vstack((imudata.gyroscope_deg_s_x, imudata.gyroscope_deg_s_y, 
                    imudata.gyroscope_deg_s_z)).T
gyro = np.deg2rad(gyro)

# Calibration accelerometer data
Ta = T_a + np.zeros((len(acc), 3, 3))
Ka = K_a + np.zeros((len(acc), 3, 3))
acc = np.einsum('ijk,ik->ij', Ta, np.einsum('ijk,ik->ij',
                    Ka, (np.array(acc) + np.array(b_a))))

# Calibration gyroscope data
Tg = T_g + np.zeros((len(gyro), 3, 3))
Kg = K_g + np.zeros((len(gyro), 3, 3))
gyro = np.einsum('ijk,ik->ij', Tg, np.einsum('ijk,ik->ij', 
                    Kg, (np.array(gyro) - np.array(b_g))))

# Fixing acceleration data and removing g from az
acc[:, 1] = acc[:, 1] - acc[0, 1]
acc[:, 2] = acc[:, 2] + 1
acc[:, 2] = acc[:, 2] - acc[0, 2]

# Conversion from g to m/s^2
acc = acc * -g
# acc[:, 2] = -acc[:, 2] 

# Fixing gyroscope data
gyro[:,0] = gyro[:,0] + gyro[0,0]
gyro[:,1] = gyro[:,1] + gyro[0,1]
gyro[:,2] = gyro[:,2] - gyro[0,2]
gyro = -gyro

# Normalize tsImu
times_Imu = np.reshape(np.array((tsImu - tsImu[tsImu.index[0]]) * 1e-9),
                       (len(tsImu),1)) 

# Create dataframe of imudata
imu_df = pd.DataFrame(
            data = np.concatenate([times_Imu, 
                                  acc[:, 0].reshape((len(times_Imu), 1)), 
                                  acc[:, 1].reshape((len(times_Imu), 1)), 
                                  acc[:, 2].reshape((len(times_Imu), 1)),
                                  gyro[:, 0].reshape((len(times_Imu), 1)), 
                                  gyro[:, 1].reshape((len(times_Imu), 1)), 
                                  gyro[:, 2].reshape((len(times_Imu), 1))], 
                                  axis = 1),
            columns = ['tsImu', 'ax_data', 'ay_data', 'az_data', 
                    'gyrox_data', 'gyroy_data', 'gyroz_data']
            )

#%% Heading of motion from gps
indexNOTFixed = np.where(gpsdata['gpsFixOK'] == 0)[0]
indexFixed = np.where(gpsdata['gpsFixOK'] == 1)[0]
if (indexNOTFixed[0]).size > 0:
    index_start = np.max([indexNOTFixed[0]-30, 0])
    index_stop = np.min([indexNOTFixed[-1] + 1 + 30, len(gpsdata['gpsFixOK'])])
    index_buono_start = np.max([indexFixed[0], 0])
    index_buono_stop = np.min([indexFixed[-1] + 1, len(gpsdata['gpsFixOK'])]) 
da_non_cambiare = np.concatenate(
                [np.arange(index_buono_start, index_start), 
                    np.arange(index_stop, index_buono_stop)],
                    axis=0
                    )

gpsdata['headingOfMotion_deg'].loc[gpsdata.index[index_start]:gpsdata.index[index_stop]] = np.interp(
                            range(gpsdata.index[index_start], gpsdata.index[index_stop+1]),
                            gpsdata.index[da_non_cambiare], 
                            gpsdata['headingOfMotion_deg'].loc[gpsdata.index[da_non_cambiare]])

idx_heading = np.argmin(np.deg2rad(gpsdata['headingOfMotion_deg']))
gps_heading_fixed = np.deg2rad(gpsdata['headingOfMotion_deg'])
gps_heading_fixed[idx_heading:] = gps_heading_fixed[idx_heading:] + 2 * np.pi 

#%% Interpolation of gps data to have the same sampling frequency
gps_longitude = np.interp(tsImu, tsGps, gpsdata['longitude']).reshape((len(tsImu), 1))
gps_latitude = np.interp(tsImu, tsGps, gpsdata['latitude']).reshape((len(tsImu), 1))
gps_altitude = np.interp(tsImu, tsGps, gpsdata['heightMSL_mm'] * 1e-3).reshape((len(tsImu), 1))
gpsFixOK = np.interp(tsImu, tsGps, gpsdata["gpsFixOK"]).reshape((len(tsImu), 1))
gps_north_vel = np.interp(tsImu, tsGps, gpsdata['northVelocity_mm_s'] * 1e-3).reshape((len(tsImu), 1))
gps_east_vel = np.interp(tsImu, tsGps, gpsdata['eastVelocity_mm_s'] * 1e-3).reshape((len(tsImu), 1))
gps_up_vel = np.interp(tsImu, tsGps, -gpsdata['downVelocity_mm_s'] * 1e-3).reshape((len(tsImu), 1))       
gps_heading = np.interp(tsImu, tsGps, gps_heading_fixed).reshape((len(tsImu), 1))

#%% Create dataframe of gps data
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

gps_df, _ = fix.fixGPS(gps_df)

# #%% Plot X,Y,Z
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_trisurf(gps_df['gps_x_pos'], gps_df['gps_y_pos'], gps_df['gps_z_pos'], color='white',
#                  edgecolors='grey', alpha=0.5)
# ax.scatter(gps_df['gps_x_pos'], gps_df['gps_y_pos'], gps_df['gps_z_pos'], c='red')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# plt.show()

#%% Dataframe containing imu_df and gps_df
inputs_df = pd.merge(imu_df, gps_df, on = 'tsImu', how = 'outer')

#%% Normalization of the inputs dataframe except for the timestamp
scaler = MinMaxScaler()
inputs_df_norm = pd.DataFrame(
                data = np.concatenate([inputs_df['tsImu'].values.reshape((len(inputs_df), 1)),
                                        scaler.fit_transform(inputs_df.drop(columns = ['tsImu'])).astype(np.float32)], 
                                        axis = 1),
                columns = inputs_df.columns
                )

#%%Load the model
if cf.load_model:
    model = tf.keras.models.load_model(f'Models/{cf.config_name}/model', compile = False)
else:
    # Create the model
    model = model_GPS(cf.ker_reg)
    # Training
    history, model = train(tf.constant(inputs_df_norm, dtype = DTYPE), model, cf)

    #Save the model
    if cf.save:
        model.save(f'Models/{cf.config_name}/model')

#%%Create the folder for the images
os.makedirs(f'Models/{cf.config_name}/Immagini/', exist_ok = True)

#%% Print the predictions
with tf.device('/cpu:0'):
    preds = model.predict(np.array(inputs_df_norm))

#%% Rotate preds

veloc = tf.expand_dims(tf.stack([preds[...,3], preds[...,4], preds[...,5]], axis = 1), axis = -1)
posit = tf.expand_dims(tf.stack([preds[...,0], preds[...,1], preds[...,2]], axis = 1), axis = -1)
phi = preds[...,6]
theta = preds[...,7]
psi = preds[...,8]

rotation_matrix_2 = tf.transpose(tf.stack(
                    [[tf.cos(theta) * tf.cos(psi),
                      tf.cos(theta) * tf.sin(psi),
                      -tf.sin(theta)],
                     [tf.cos(psi) * tf.sin(phi) *\
                      tf.sin(theta) - tf.cos(phi) *\
                      tf.sin(psi),
                      tf.cos(phi) * tf.cos(psi) +\
                      tf.sin(phi) * tf.sin(psi) *\
                      tf.sin(theta),
                      tf.sin(phi) * tf.cos(theta)],
                      [tf.sin(phi) * tf.sin(psi) +\
                       tf.cos(phi) * tf.cos(psi) *\
                       tf.sin(theta),
                       tf.cos(phi) * tf.sin(psi) *\
                       tf.sin(theta) - tf.cos(psi) *\
                       tf.sin(phi),
                       tf.cos(phi) * tf.cos(theta)]],
                    axis = 1))

veloc_rotated = tf.squeeze(tf.matmul(rotation_matrix_2, veloc), axis = 2)
posit_rotated = tf.squeeze(tf.matmul(rotation_matrix_2, posit), axis = 2)


#%% 
preds_df = inputs_df_norm
# preds_df = inputs_df
preds_df['gps_x_vel'] = -veloc_rotated[...,0]
preds_df['gps_y_vel'] = -veloc_rotated[...,1]
preds_df['gps_z_vel'] = veloc_rotated[...,2]
preds_df['gps_x_pos'] = -posit_rotated[...,0]
preds_df['gps_y_pos'] = -posit_rotated[...,1]
preds_df['gps_z_pos'] = posit_rotated[...,2]

#%% Inverse normalization of the predictions dataframe except for the timestamp
preds_df_inv = pd.DataFrame(
                data = np.concatenate([preds_df['tsImu'].values.reshape((len(preds_df), 1)),
                                        scaler.inverse_transform(preds_df.drop(columns = ['tsImu'])).astype(np.float32)],
                                        axis = 1),    
                columns = inputs_df.columns
                )
preds_df_inv = pd.DataFrame.drop(
                preds_df_inv, 
                columns = ['tsImu', 'ax_data', 'ay_data', 'az_data', 
                            'gyrox_data', 'gyroy_data', 'gyroz_data', 
                            'gpsFixOK', 'gps_longitude', 'gps_latitude',
                            'gps_altitude', 'gps_heading']
                )

#Create a dataframe containing the predictions and the gps_df['tsImu']
df_ekf = pd.concat([imu_df, 
                          gps_df['gps_longitude'], 
                          gps_df['gps_latitude'], 
                          preds_df_inv], axis = 1)

# Save the dataframe as a pickle file
df_ekf.to_pickle(
    '~/Projects/Pia/verticaldisplacement/src/PINN/pos_vel.pkl', 
    compression='infer', protocol=5, storage_options=None)
#%% Predictions
px = preds_df_inv['gps_x_pos'].values
py = preds_df_inv['gps_y_pos'].values
pz = preds_df_inv['gps_z_pos'].values
vx = preds_df_inv['gps_x_vel'].values
vy = preds_df_inv['gps_y_vel'].values
vz = preds_df_inv['gps_z_vel'].values

#%% Plot
plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title('Position x ')
plt.plot(px, label = 'px')
plt.subplot(1, 3, 2) 
plt.title('Position y ')
plt.plot(py, label = 'py')
plt.subplot(1, 3, 3)
plt.title('Position z ')
plt.plot(pz, label = 'pz')
plt.tight_layout()  
plt.savefig(f'Models/{cf.config_name}/Immagini/Position.png')
plt.show()

plt.figure(figsize=(10, 6))
plt.subplot(1, 3, 1)
plt.title('Speed x ')
plt.plot(vx, label = 'vx')
plt.subplot(1, 3, 2) 
plt.title('Speed y ')
plt.plot(vy, label = 'vy')
plt.subplot(1, 3, 3)
plt.title('Speed z ')
plt.plot(vz, label = 'vz')
plt.tight_layout()  
plt.savefig(f'Models/{cf.config_name}/Immagini/Speed.png')
plt.show()

#%% Plot xy
plt.figure()
plt.title('Route 2D')
plt.plot(px, py)
plt.savefig(f'Models/{cf.config_name}/Immagini/Route2D.png')
plt.show()

# Plot X,Y,Z
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(px, py, pz, color='white', edgecolors='grey', alpha=0.5)
ax.scatter(px, py, pz, c='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.title('Route 3D')
plt.savefig(f'Models/{cf.config_name}/Immagini/Route3D.png')
plt.show()
# %%

'''
Preprocessing functions for the entire project
'''
#%% Import packages
import numpy as np
import pandas as pd
import pymap3d as pm
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, sosfilt

#%% Filtering class
class filtering():

    def __init__(self) -> None:
        pass

    def butter_highpass(self, highcut, fs, order=5):
        return butter(order, highcut, btype='high', fs=fs)
    
    def butter_lowpass(self, lowcut, fs, order=5):
        return butter(order, lowcut, btype='low', fs=fs)
    
    #Applico il filtro 2 volte
    def twice_filter_highpass(self, data, highcut, fs, order=5):
        b, a = self.butter_highpass(highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    def twice_filter_lowpass(self, data, lowcut, fs, order=5):
        b, a = self.butter_lowpass(lowcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    #Applico il filtro una volta (prima della doppia integrazione)
    def filter_highpass(self, data, highcut, fs, order = 2):
        sos_filt = butter(order, highcut, btype = 'high', output = 'sos', fs = fs)
        y = sosfilt(sos_filt, data)
        return y
    

class fixGPS_class():

    def __init__(self) -> None:
        pass

    def fixGPS(self, gps_df) -> pd.DataFrame:

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

        #Compute geodetic coordinates
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

        # Gira pos, vel di heading
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
                   
        return gps_df_final, gps_df_utilities
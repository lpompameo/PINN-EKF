'''
PINN functions
'''
#%% Import libraries
import os
import numpy as np
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
from keras.layers import Input, Dense , BatchNormalization, Activation

#%% Global variables
DTYPE = tf.float32

# #%% Model
def model_GPS(ker_reg=None):
    if ker_reg is not None:
        ker_reg = tf.keras.regularizers.L2(ker_reg)

    # Input layer 18-dimensional
    inputs = Input(shape=(18,))

    # Hidden layers

    x = Dense(20,activation='tanh', name='hidden1', 
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(inputs)
    x = Dense(20,activation='tanh', name='hidden2', 
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(x)  
    x = Dense(20,activation='tanh', name='hidden3',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(x)
    x = Dense(20,activation='linear', name='hidden4', 
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(x)
    x = Dense(20,activation='tanh', name='hidden5',
                kernel_initializer='glorot_uniform',
                kernel_regularizer=ker_reg)(x)
    
    # Output layer 9-dimensional

    outputs = Dense(9, activation='tanh', 
                    name='output', 
                    kernel_regularizer=ker_reg)(x)


    # Create the model

    model = Model(inputs=inputs, outputs=outputs)
    model.summary()

    return model

#%% 
# Loss function
def loss_fun(inputs, model):

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch(inputs)

            preds = model(inputs)

            px = preds[...,0]
            py = preds[...,1]
            pz = preds[...,2]
            phi = preds[...,6]
            theta = preds[...,7]
            psi = preds[...,8]

        grads_x = tape2.gradient(px, inputs)[...,0]
        grads_y = tape2.gradient(py, inputs)[...,0]
        grads_z = tape2.gradient(pz, inputs)[...,0]
        grads_phi = tape2.gradient(phi, inputs)[...,0]
        grads_theta = tape2.gradient(theta, inputs)[...,0]
        grads_psi = tape2.gradient(psi, inputs)[...,0]

        del tape2

    grads2_x = tape.gradient(grads_x, inputs)[...,0]
    grads2_y = tape.gradient(grads_y, inputs)[...,0]
    grads2_z = tape.gradient(grads_z, inputs)[...,0]

    del tape

    # Rotate acceleration, velocity and position for the inertial frame
    rotation_matrix = tf.transpose(tf.stack(
                            [[tf.math.cos(preds[...,7]) * tf.math.cos(preds[...,8]), 
                            tf.math.cos(preds[...,8]) * tf.math.sin(preds[...,6]) *\
                            tf.math.sin(preds[...,7]) - tf.math.cos(preds[...,6]) *\
                            tf.math.sin(preds[...,8]),
                            tf.math.sin(preds[...,6]) * tf.math.sin(preds[...,8]) +\
                            tf.math.cos(preds[...,6]) * tf.math.cos(preds[...,8]) *\
                            tf.math.sin(preds[...,7])],
                            [tf.math.cos(preds[...,7]) * tf.math.sin(preds[...,8]), 
                            tf.math.cos(preds[...,6]) * tf.math.cos(preds[...,8]) +\
                            tf.math.sin(preds[...,6]) * tf.math.sin(preds[...,8]) *\
                            tf.math.sin(preds[...,7]),
                            tf.math.cos(preds[...,6]) * tf.math.sin(preds[...,8]) *\
                            tf.math.sin(preds[...,7]) - tf.math.cos(preds[...,8]) *\
                            tf.math.sin(preds[...,6])],
                            [-tf.math.sin(preds[...,7]), 
                            tf.math.sin(preds[...,6]) * tf.math.cos(preds[...,7]),
                            tf.math.cos(preds[...,6]) * tf.math.cos(preds[...,7])]])
                            )                       

    inputs_acc = tf.stack([inputs[...,1], inputs[...,2], inputs[...,3]])
    acc_rotated = tf.matmul(rotation_matrix, inputs_acc)[0,...]
    ax = acc_rotated[0,...]
    ay = acc_rotated[1,...]
    az = acc_rotated[2,...]

    inputs_vel = tf.stack([inputs[...,11], inputs[...,12], inputs[...,13]])
    vel_rotated = tf.matmul(rotation_matrix, inputs_vel)[0,...]
    vx = vel_rotated[0,...]
    vy = vel_rotated[1,...]
    vz = vel_rotated[2,...]

    inputs_pos = tf.stack([inputs[...,15], inputs[...,16], inputs[...,17]])
    pos_rotated = tf.matmul(rotation_matrix, inputs_pos)[0,...]
    px = pos_rotated[0,...]
    py = pos_rotated[1,...]
    pz = pos_rotated[2,...]

    phi_dot = inputs[...,4] +\
                inputs[...,5] * tf.math.sin(preds[...,6]) *\
                tf.math.tan(preds[...,7]) + inputs[...,6] *\
                tf.math.cos(preds[...,6]) * tf.math.tan(preds[...,7])
    theta_dot = inputs[...,5] * tf.math.cos(preds[...,6]) -\
                inputs[...,6] * tf.math.sin(preds[...,6])
    psi_dot = inputs[...,5] * tf.math.sin(preds[...,6]) /\
                tf.math.cos(preds[...,7]) + inputs[...,6] *\
                tf.math.cos(preds[...,6]) / tf.math.cos(preds[...,7])

    # Initial conditions
    cond = tf.equal(inputs[...,0], 0.)
    phi_ic  = tf.where(cond, tf.math.atan(
                        tf.reduce_mean(inputs[...,2]) /\
                        tf.reduce_mean(inputs[...,3])), preds[...,6])
    theta_ic  = tf.where(cond, tf.math.atan(
                        tf.reduce_mean(inputs[...,1]) /\
                        tf.math.sqrt(tf.square(tf.reduce_mean(inputs[...,2])) +\
                                     tf.square(tf.reduce_mean(inputs[...,3])))), preds[...,7])
    psi_ic  = tf.where(cond, 0., preds[...,8])

    spost_x = preds[...,0] - px[0] - \
        vx[0]*inputs[...,0] - 0.5*grads2_x*tf.square(inputs[...,0])
    spost_y = preds[...,1] - py[0] - \
        vy[0]*inputs[...,0] - 0.5*grads2_y*tf.square(inputs[...,0])
    spost_z = preds[...,2] - pz[0] - \
        vz[0]*inputs[...,0] - 0.5*grads2_z*tf.square(inputs[...,0])
    
    veloc_x = preds[...,3] - vx[0] - grads_x*inputs[...,0]
    veloc_y = preds[...,4] - vy[0] - grads_y*inputs[...,0]
    veloc_z = preds[...,5] - vz[0] - grads_z*inputs[...,0]

    roll = preds[...,6] - phi_ic - grads_phi*inputs[...,0]
    pitch = preds[...,7] - theta_ic - grads_theta*inputs[...,0]
    yaw = preds[...,8] - psi_ic - grads_psi*inputs[...,0]

    # Loss function
    reg = tf.reduce_sum(model.losses)

    loss = tf.reduce_mean(tf.square(spost_x)) + \
            tf.reduce_mean(tf.square(spost_y)) + \
            tf.reduce_mean(tf.square(spost_z)) + \
            tf.reduce_mean(tf.square(veloc_x)) + \
            tf.reduce_mean(tf.square(veloc_y)) + \
            tf.reduce_mean(tf.square(veloc_z)) + \
            tf.reduce_mean(tf.square(roll)) + \
            tf.reduce_mean(tf.square(pitch)) + \
            tf.reduce_mean(tf.square(yaw)) + reg + \
            tf.reduce_mean(tf.square(grads2_x - ax)) + \
            tf.reduce_mean(tf.square(grads2_y - ay)) + \
            tf.reduce_mean(tf.square(grads2_z - az)) + \
            tf.reduce_mean(tf.square(grads_phi - phi_dot)) + \
            tf.reduce_mean(tf.square(grads_theta - theta_dot)) + \
            tf.reduce_mean(tf.square(grads_psi - psi_dot)) 


    return {'loss': loss,
            'spost_x': tf.reduce_mean(tf.square(spost_x)),
            'spost_y': tf.reduce_mean(tf.square(spost_y)),
            'spost_z': tf.reduce_mean(tf.square(spost_z)),
            'veloc_x': tf.reduce_mean(tf.square(veloc_x)),
            'veloc_y': tf.reduce_mean(tf.square(veloc_y)),
            'veloc_z': tf.reduce_mean(tf.square(veloc_z)),
            'roll': tf.reduce_mean(tf.square(roll)), 
            'pitch': tf.reduce_mean(tf.square(pitch)),
            'yaw': tf.reduce_mean(tf.square(yaw)),
            'acc_x': tf.reduce_mean(tf.square(grads2_x - ax)),
            'acc_y': tf.reduce_mean(tf.square(grads2_y - ay)),
            'acc_z': tf.reduce_mean(tf.square(grads2_z - az)),
            'phi_dot': tf.reduce_mean(tf.square(grads_phi - phi_dot)),
            'theta_dot': tf.reduce_mean(tf.square(grads_theta - theta_dot)),
            'psi_dot': tf.reduce_mean(tf.square(grads_psi - psi_dot)),
            'reg': reg
            }

 # %% Optimizator
def select_optimizator(cf):
    if cf.opt == 'Adam':
        opt = tf.keras.optimizers.Adam(learning_rate=cf.lr)
    elif cf.opt == 'RMSprop':
        opt = tf.keras.optimizers.RMSprop(learning_rate=cf.lr)
    elif cf.opt == 'Adadelta':
        opt = tf.keras.optimizers.Adadelta(learning_rate=cf.lr)
    return opt

#%% Dictionary for the loss
def obtain_loss_dict():
        return {'epoch':[],
                'loss': [],
                'spost_x': [],
                'spost_y': [],
                'spost_z': [],
                'veloc_x': [],
                'veloc_y': [],
                'veloc_z': [],
                'roll': [],
                'pitch': [],
                'yaw': [],
                'acc_x': [],
                'acc_y': [],
                'acc_z': [],
                'phi_dot': [],
                'theta_dot': [],
                'psi_dot': [],
                'reg': [],
                }

#%% Training step
def train_step(inputs, model, opt):

    with tf.GradientTape(persistent=True) as loss_tape:        
        loss_dict = loss_fun(inputs, model)
    gradients_of_model = loss_tape.gradient(loss_dict['loss'],  
                                            model.trainable_variables)    
    opt.apply_gradients(zip(gradients_of_model, 
                            model.trainable_variables))
    del loss_tape
    return loss_dict

#%% Training
def train(inputs, model, cf):
    import time
    opt = select_optimizator(cf)    
    t0 = time.time()
    best_loss = tf.constant(np.inf, dtype=DTYPE)
    patience = cf.patience
    wait = 0

    history = obtain_loss_dict()
    for epoch in range(1, cf.epochs + 1):     
        start_epoch = time.time()  
        
        loss_dict = train_step(inputs, model, opt)
        loss_value = loss_dict['loss']
        history['epoch'].append(epoch)
                
        for key, elem in loss_dict.items():
            history[key].append(elem)
        
        if loss_value<best_loss:
            mega_str = f'Best {epoch}: '             
            
            for key, elem in loss_dict.items():
                mega_str += f'{key} = {elem.numpy():10.6e}  '            
            
            mega_str += f'Time = {time.time()-start_epoch:.2f} sec'
            print('\r'+mega_str, end= '')           
                              
            best_loss = loss_value
            wait = 0
            best_weights= model.get_weights() 
            best_dict = loss_dict.copy()    

        elif wait>=patience:
            print('Stop the train phase')
            break
        else:
            wait +=1
            if epoch % cf.print_epoch == 0:   
                if wait > cf.print_epoch: epoch_string = f'Epoch {epoch} '
                else: epoch_string = f'\nEpoch {epoch} '

                for key, elem in loss_dict.items():
                    if key == 'epoch': continue
                    epoch_string += f'{key} = {elem.numpy():10.6e}  '               

                epoch_string += f'Time = {time.time()-start_epoch:.2f} sec'
                print('\r'+epoch_string)

        if epoch % cf.plot_epoch == 0: 
            plt.figure(figsize=(10,10))
            plt.plot(history['loss'][-cf.plot_epoch:], label='loss')
            plt.show()
            if cf.save:
                os.makedirs(f'Models/{cf.config_name}/', exist_ok=True)
                model.save(f'Models/{cf.config_name}/')

    if wait !=0:
        model.set_weights(best_weights)
    
    if cf.save:
        os.makedirs(f'Models/{cf.config_name}/', exist_ok=True)
        model.save(f'Models/{cf.config_name}/')

    print('\nComputation time: {} seconds'.format(time.time()-t0))


    return history, model            
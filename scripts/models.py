import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization,  Conv2D, MaxPooling2D
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import layers
import numpy as np
import os
from . import preprocessing, settings


def RIG(path_weights=settings.path_weights):
    """
    Transformer encoder model for lens detection.

    Parameters:
        path_weight (str): Path to the pre-trained weight file.

    Returns:
        tf.keras.models.Model: Lens detection model.
    """
    d_model = 128
    maximum_position_encoding = 10000
    scaling_factor = tf.keras.backend.constant(np.sqrt(d_model), shape = (1,1,1))

    # Define the model architecture
    # Encoder ##################################
    input_shape = (101,101,3)
    inputs=Input(shape=(input_shape),name='input_layer')

    x = Conv2D(filters=16, kernel_size=(4, 4), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01), name='Conv1' )(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01), name='Conv2' )(x)
    x = MaxPooling2D((2, 2),name='max1')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv4')(x)
    x = MaxPooling2D((2, 2),name='max2')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv6')(x)
    x = MaxPooling2D((2, 2),name='max3')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv7')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv8')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x= tf.reshape(tensor=x, shape=(tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2],d_model))

    ## positional encoding
    x = tf.keras.layers.Multiply()([x,scaling_factor])
    pos = preprocessing.positional_encoding(maximum_position_encoding, d_model)
    x = tf.keras.layers.Add()([x, pos[: , :tf.shape(x)[1], :]] )

    ## Multihead Attention 1

    x1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)
    x2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)

    x = tf.keras.layers.Add()([x1, x2])

    x3 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)
    x4 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)

    x = tf.keras.layers.Add()([x3, x4])

    ## Feed Forward 1
    dense =  Dense(2*d_model, activation = 'elu',kernel_initializer="glorot_uniform",name='dense1')(x)
    dense = layers.Dropout(0.5)(dense)
    dense =  Dense(d_model, activation = 'elu',kernel_initializer="glorot_uniform",name='dense2')(dense)
    dense = layers.Dropout(0.5)(dense)
    x = tf.keras.layers.Add()([x , dense])                                          # residual connection
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ######################################################

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    ######################################################


    x =  Dense(1024, activation = 'elu',kernel_initializer="glorot_uniform",name='dense4')(x)
    x = layers.Dropout(0.5)(x)

    x =  Dense(512, activation = 'elu',kernel_initializer="glorot_uniform",name='dense5')(x)
    x = layers.Dropout(0.5)(x)

    x =  Dense(128, activation = 'elu',kernel_initializer="glorot_uniform",name='dense6')(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    L2 = tf.keras.models.Model(inputs=inputs, outputs=output)

    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,decay_steps=10000,decay_rate=0.9)
    opt1=tf.keras.optimizers.Adam(learning_rate=lr_schedule1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    L2.compile(optimizer=opt1, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])
    L2.load_weights(os.path.join(path_weights, "Lens_Detector_RIG1.h5"))           ## Load the model by substituting the
                                                                           ## directory of Lens Detector 15
    return  L2



def lens15(path_weights=settings.path_weights):
    """
    Lens detection model with a transformer-like encoder architecture.

    Parameters:
        path_weights (str): Path to the pre-trained weight file.

    Returns:
        tf.keras.models.Model: Lens detection model.
    """
    d_model = 128
    maximum_position_encoding = 10000

    scaling_factor = tf.keras.backend.constant(np.sqrt(d_model), shape=(1, 1, 1))

    # Encoder
    input_shape = (101, 101, 4)
    inputs = Input(shape=(input_shape), name='input_layer')

    # Convolutional layers
    x = Conv2D(filters=16, kernel_size=(4, 4), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv1')(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv2')(x)
    x = MaxPooling2D((2, 2), name='max1')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv4')(x)
    x = MaxPooling2D((2, 2), name='max2')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv6')(x)
    x = MaxPooling2D((2, 2), name='max3')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv7')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu',
               kernel_initializer=glorot_uniform(seed=0), kernel_regularizer=tf.keras.regularizers.l2(0.01),
               name='Conv8')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x = tf.reshape(tensor=x, shape=(tf.shape(x)[0], tf.shape(x)[1] * tf.shape(x)[2], d_model))

    # Positional encoding
    x = tf.keras.layers.Multiply()([x, scaling_factor])
    pos = preprocessing.positional_encoding(maximum_position_encoding, d_model)
    x = tf.keras.layers.Add()([x, pos[:, :tf.shape(x)[1], :]])

    # Multihead Attention layers
    x1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x, x)
    x2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x, x)
    x = tf.keras.layers.Add()([x1, x2])

    x3 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01))(x, x)
    x4 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True,
                                            kernel_initializer='glorot_uniform',
                                            kernel_regularizer=tf.keras.regularizers.l2(0.01),
                                            name='Att')(x, x)
    x = tf.keras.layers.Add()([x3, x4])

    # Feed Forward layer
    dense = Dense(2 * d_model, activation='elu', kernel_initializer="glorot_uniform", name='dense1')(x)
    dense = layers.Dropout(0.5)(dense)
    dense = Dense(d_model, activation='elu', kernel_initializer="glorot_uniform", name='dense2')(dense)
    dense = layers.Dropout(0.5)(dense)
    x = tf.keras.layers.Add()([x, dense])  # Residual connection
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    # Global Average Pooling
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    # Dense layers
    x = Dense(1024, activation='elu', kernel_initializer="glorot_uniform", name='dense4')(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(512, activation='elu', kernel_initializer="glorot_uniform", name='dense5')(x)
    x = layers.Dropout(0.5)(x)
    x = Dense(128, activation='elu', kernel_initializer="glorot_uniform", name='dense6')(x)
    x = layers.Dropout(0.5)(x)

    # Output layer
    output = layers.Dense(1, activation="sigmoid")(x)

    # Model
    Lens_Detector_15 = tf.keras.models.Model(inputs=inputs, outputs=output)

    # Optimizer and compilation
    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4, decay_steps=10000,
                                                                  decay_rate=0.9)
    opt1 = tf.keras.optimizers.Adam(learning_rate=lr_schedule1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)
    Lens_Detector_15.compile(optimizer=opt1, loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC()])

    # Load pre-trained weights
    Lens_Detector_15.load_weights(os.path.join(path_weights, "Lens Detector 15"))

    return Lens_Detector_15


def lens16(path_weights= settings.path_weights):

    """
    Lens detection model with a transformer-like encoder architecture.

    Parameters:
        path_weights (str): Path to the pre-trained weight file.

    Returns:
        tf.keras.models.Model: Lens detection model.
    """

    d_model = 128
    maximum_position_encoding = 10000

    scaling_factor = tf.keras.backend.constant(np.sqrt(d_model), shape = (1,1,1))

    # Encoder ##################################
    input_shape = (101,101,4)

    inputs=Input(shape=(input_shape),name='input_layer')

    x = Conv2D(filters=16, kernel_size=(4, 4), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01), name='Conv1' )(inputs)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01), name='Conv2' )(x)
    x = MaxPooling2D((2, 2),name='max1')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv3')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv4')(x)
    x = MaxPooling2D((2, 2),name='max2')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv5')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv6')(x)
    x = MaxPooling2D((2, 2),name='max3')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv7')(x)
    x = layers.Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv8')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x= tf.reshape(tensor=x, shape=(tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2],d_model))

    ## positional encoding
    x = tf.keras.layers.Multiply()([x,scaling_factor])
    pos = preprocessing.positional_encoding(maximum_position_encoding, d_model)
    x = tf.keras.layers.Add()([x, pos[: , :tf.shape(x)[1], :]] )

    ## Multihead Attention 1

    x1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)
    x2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)
    x3 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)
    x4 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)

    x = tf.keras.layers.Add()([x1, x2, x3, x4])

    ## Feed Forward 1
    dense =  Dense(2*d_model, activation = 'elu',kernel_initializer="glorot_uniform",name='dense1')(x)
    dense = layers.Dropout(0.5)(dense)
    dense =  Dense(d_model, activation = 'elu',kernel_initializer="glorot_uniform",name='dense2')(dense)
    dense = layers.Dropout(0.5)(dense)
    x = tf.keras.layers.Add()([x , dense])                                          # residual connection
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ######################################################

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    ######################################################

    x =  Dense(512, activation = 'elu',kernel_initializer="glorot_uniform",name='dense4')(x)
    x = layers.Dropout(0.5)(x)

    x =  Dense(256, activation = 'elu',kernel_initializer="glorot_uniform",name='dense5')(x)
    x = layers.Dropout(0.5)(x)

    x =  Dense(64, activation = 'elu',kernel_initializer="glorot_uniform",name='dense6')(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Dense(1, activation="sigmoid")(x)

    x1 = Conv2D(filters=16, kernel_size=(4, 4), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    x1 = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = BatchNormalization(axis=3)(x1)

    x1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = BatchNormalization(axis=3)(x1)

    x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = layers.Dropout(0.5)(x1)

    x1 = Conv2D(filters=d_model, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = Conv2D(filters=d_model, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1)
    x1 = BatchNormalization(axis=3)(x1)
    x1 = layers.Dropout(0.50)(x1)

    x1 = tf.reshape(tensor=x1, shape=(tf.shape(x1)[0], tf.shape(x1)[1]*tf.shape(x1)[2],d_model))

    ## positional encoding
    x1 = tf.keras.layers.Multiply()([x1,scaling_factor])
    pos1 = preprocessing.positional_encoding(maximum_position_encoding, d_model)
    x1 = tf.keras.layers.Add()([x1, pos1[: , :tf.shape(x1)[1], :]] )

    ## Multihead Attention 2

    x1_1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1,x1)
    x1_2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1_1,x1_1)
    x1_3 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1_2,x1_2)
    x1_4 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x1_3,x1_3)

    ## Feed Forward 2
    dense1 =  Dense(2*d_model, activation = 'elu',kernel_initializer="glorot_uniform")(x1_4)
    dense1 = layers.Dropout(0.5)(dense1)
    dense1 =  Dense(d_model, activation = 'elu',kernel_initializer="glorot_uniform")(dense1)
    dense1 = layers.Dropout(0.5)(dense1)
    x1 = tf.keras.layers.Add()([x1 , dense1])                                          # residual connection
    x1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x1)

    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)

    x1 =  Dense(512, activation = 'elu',kernel_initializer="glorot_uniform",name='dense4_1')(x1)
    x1 = layers.Dropout(0.5)(x1)

    x1 =  Dense(256, activation = 'elu',kernel_initializer="glorot_uniform",name='dense5_1')(x1)
    x1 = layers.Dropout(0.5)(x1)

    x1 =  Dense(64, activation = 'elu',kernel_initializer="glorot_uniform",name='dense6_1')(x1)
    x1 = layers.Dropout(0.5)(x1)

    x1 = layers.Dense(1, activation="sigmoid")(x1)

    output = tf.keras.layers.Concatenate()([x1, x])
    output = layers.Dense(1, activation="sigmoid")(output)

    Lens_Detector_16 = tf.keras.models.Model(inputs=inputs, outputs=output)

    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,decay_steps=10000,decay_rate=0.9)
    opt1=tf.keras.optimizers.Adam(learning_rate=lr_schedule1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    Lens_Detector_16.compile(optimizer=opt1, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])
    Lens_Detector_16.load_weights(os.path.join(path_weights, "Lens Detector 16"))         ## Load the model by substituting the
                                                                               ## directory of Lens Detector 16
    return Lens_Detector_16





def lens21(path_weights=settings.path_weights):

    """
    Lens detection model with a transformer-like encoder architecture.

    Parameters:
        path_weights (str): Path to the pre-trained weight file.

    Returns:
        tf.keras.models.Model: Lens detection model.
    """
        
    d_model = 128
    maximum_position_encoding = 10000
    scaling_factor = tf.keras.backend.constant(np.sqrt(d_model), shape = (1,1,1))

    # Define the model architecture
    # Encoder ##################################
    input_shape = (101,101,4)
    inputs=Input(shape=(input_shape),name='input_layer')

    x = Conv2D(filters=16, kernel_size=(4, 4), strides=(1, 1), padding='valid',activation='elu', 
               kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01), name='Conv1' )(inputs)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01), name='Conv2' )(x)
    x = MaxPooling2D((2, 2),name='max1')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv3')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv4')(x)
    x = MaxPooling2D((2, 2),name='max2')(x)
    x = BatchNormalization(axis=3)(x)

    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv5')(x)
    x = BatchNormalization(axis=3)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv6')(x)
    x = MaxPooling2D((2, 2),name='max3')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv7')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='valid',activation='elu', kernel_initializer=glorot_uniform(seed=0),kernel_regularizer=tf.keras.regularizers.l2(0.01),name='Conv8')(x)
    x = BatchNormalization(axis=3)(x)
    x = layers.Dropout(0.5)(x)

    x= tf.reshape(tensor=x, shape=(tf.shape(x)[0], tf.shape(x)[1]*tf.shape(x)[2],d_model))

    ## positional encoding
    x = tf.keras.layers.Multiply()([x,scaling_factor])
    pos = preprocessing.positional_encoding(maximum_position_encoding, d_model)
    x = tf.keras.layers.Add()([x, pos[: , :tf.shape(x)[1], :]] )

    ## Multihead Attention 1

    x1 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)
    x2 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)

    x = tf.keras.layers.Add()([x1, x2])

    x3 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)
    x4 = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=d_model, value_dim=d_model, dropout=0.20, use_bias=True, kernel_initializer='glorot_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x,x)

    x = tf.keras.layers.Add()([x3, x4])

    ## Feed Forward 1
    dense =  Dense(2*d_model, activation = 'elu',kernel_initializer="glorot_uniform",name='dense1')(x)
    dense = layers.Dropout(0.5)(dense)
    dense =  Dense(d_model, activation = 'elu',kernel_initializer="glorot_uniform",name='dense2')(dense)
    dense = layers.Dropout(0.5)(dense)
    x = tf.keras.layers.Add()([x , dense])                                          # residual connection
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)

    ######################################################

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    ######################################################


    x =  Dense(1024, activation = 'elu',kernel_initializer="glorot_uniform",name='dense4')(x)
    x = layers.Dropout(0.5)(x)

    x =  Dense(512, activation = 'elu',kernel_initializer="glorot_uniform",name='dense5')(x)
    x = layers.Dropout(0.5)(x)

    x =  Dense(128, activation = 'elu',kernel_initializer="glorot_uniform",name='dense6')(x)
    x = layers.Dropout(0.5)(x)

    output = layers.Dense(1, activation="sigmoid")(x)

    Lens_Detector_21 = tf.keras.models.Model(inputs=inputs, outputs=output)

    lr_schedule1 = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-4,decay_steps=10000,decay_rate=0.9)
    opt1=tf.keras.optimizers.Adam(learning_rate=lr_schedule1, beta_1=0.9, beta_2=0.999, epsilon=1e-07)

    Lens_Detector_21.compile(optimizer=opt1, loss='binary_crossentropy', metrics=['accuracy',tf.keras.metrics.AUC()])
    Lens_Detector_21.load_weights(os.path.join(path_weights, "Lens Detector_21"))       
                                                                           
    return Lens_Detector_21

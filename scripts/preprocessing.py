import numpy as np
import tensorflow as tf

def scaling_clipping(X):
    """
    Applies clipping and scaling to the input data.

    Parameters:
    - X (numpy.ndarray): Input data.

    Returns:
    - numpy.ndarray: Processed data after clipping and scaling.
    """

    # Clipping and scaling parameters applied to the data as preprocessing
    vmin = -1e-9
    vmax = 1e-9
    scale = 100

    # Identify elements in the array with the value 100 and set them to 0
    mask = np.where(X == 100)
    X[mask] = 0

    # Simple clipping and rescaling the images
    X = np.clip(X, vmin, vmax) / vmax * scale

    # Set elements with the value 100 to 0 again
    X[mask] = 0

    return X

def get_angles(pos, i, d_model):
    """
    Calculate angle rates for positional encoding.

    Parameters:
    - pos (numpy.ndarray): Positional indices.
    - i (int): Index.
    - d_model (int): Dimension of the model.

    Returns:
    - numpy.ndarray: Angle rates.
    """

    angle_rates = 1 / np.power(12800, (2 * (i//2)) / np.float64(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    """
    Generate positional encodings for input sequences.

    Parameters:
    - position (int): Maximum position for the encoding.
    - d_model (int): Dimension of the model.

    Returns:
    - tensorflow.Tensor: Positional encodings.
    """

    # Calculate angle rates for each position and dimension
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # Apply sin to even indices in the array; 2i
    angle_rads[0::2, :] = np.sin(angle_rads[0::2, :])

    # Apply cos to odd indices in the array; 2i+1
    angle_rads[1::2, :] = np.cos(angle_rads[1::2, :])

    # Add an extra dimension to the array
    pos_encoding = angle_rads[np.newaxis]

    # Convert to TensorFlow float32
    return tf.cast(pos_encoding, dtype=tf.float32)

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import gym
import scipy.signal
import time
from numba import jit

@jit
def dcs(x:np.array, d):
    """Cumulative sum of rewards

    Args:
        x (np.array): _description_
        d (float): _description_

    Returns:
        _type_: _description_
    """
    disc_sum = np.zeros(len(x))
    for i in range(len(x)):
        tmp = 0
        for j in range(len(x[i:])):
            tmp += x[i+j]*pow(d, j)
        disc_sum[i] = tmp
        
    return disc_sum


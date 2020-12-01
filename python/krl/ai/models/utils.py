'''Models utilities'''

import tensorflow as tf

from tensorflow.keras import layers


__all__ = ['Sampling']


class Sampling(layers.Layer):
    '''Uses (z_mean, z_log_var) to sample z, the vector encoding a digit.'''

    def call(self, inputs):
        z_mean, z_log_var = inputs
        epsilon = tf.keras.backend.random_normal(shape=tf.shape(z_mean))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

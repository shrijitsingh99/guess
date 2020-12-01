'''Variational autoencoder class implementation'''

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from typing import Any, Dict, Optional, Tuple

from .utils import Sampling


__all__ = ['VariationalAutoEncoder']


class VariationalAutoEncoder(keras.Model):
    _input_shape: Tuple[int, ...]
    _latent_space_size: int
    _encoder: keras.Model
    _decoder: keras.Model
    _reconstrunction_loss_weight: float
    _kl_divergence_loss_weight: float

    def __init__(self, latent_space_size: int,
                 input_shape: Tuple[int, ...],
                 initial_compression_filters_number: int,
                 compression_layers_number: int,
                 initial_expansion_filters_number: int,
                 expansion_layers_number: int,
                 filter_layers_multiplier: int,
                 kernel_size: int,
                 dropout: float = 0.,
                 verbose: bool = False,
                 reconstrunction_loss_weight: float = 1.,
                 kl_divergence_loss_weight: float = 1.,
                 **kwargs: Any) -> None:
        super(VariationalAutoEncoder, self).__init__(**kwargs)

        self._reconstrunction_loss_weight = reconstrunction_loss_weight
        self._kl_divergence_loss_weight = kl_divergence_loss_weight
        self._input_shape = input_shape
        self._latent_space_size = latent_space_size

        self.build_encoder(initial_filters_number=initial_compression_filters_number,
                           filter_layers_multiplier=filter_layers_multiplier,
                           compression_layers_number=compression_layers_number,
                           kernel_size=kernel_size, dropout=dropout, verbose=verbose)

        self.build_decoder(initial_filters_number=initial_expansion_filters_number,
                           filter_layers_multiplier=filter_layers_multiplier,
                           expansion_layers_number=expansion_layers_number,
                           kernel_size=kernel_size, dropout=dropout, verbose=verbose)

    def build_encoder(self, initial_filters_number: int, filter_layers_multiplier: int,
                      compression_layers_number: int, kernel_size: int, dropout: float = 0.0, stride: int = 2,
                      activation: str = "relu", padding: str = "same", verbose: bool = False) -> None:
        encoder_inputs = keras.Input(shape=self._input_shape)
        x = encoder_inputs

        layer_filters = initial_filters_number
        for l in range(compression_layers_number):
            x = layers.Conv2D(layer_filters, kernel_size, activation=activation, strides=stride, padding=padding)(x)
            if dropout > 0.:
                x = layers.Dropout(dropout)(x)
            layer_filters = initial_filters_number * filter_layers_multiplier

        x = layers.Flatten()(x)
        x = layers.Dense(self._latent_space_size * 8, activation=activation)(x)
        z = layers.Dense(self._latent_space_size, activation=activation)(x)

        x0, x1 = layers.Lambda(lambda l: tf.split(l, num_or_size_splits=2, axis=1))(x)
        z_mean = layers.Dense(self._latent_space_size, name="z_mean")(x0)
        z_log_var = layers.Dense(self._latent_space_size, name="z_log_var")(x1)
        z = Sampling()([z_mean, z_log_var])

        self._encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        if verbose:
            self._encoder.summary()

    def build_decoder(self, initial_filters_number: int, filter_layers_multiplier: int, expansion_layers_number: int,
                      kernel_size: int, stride: int = 2, activation: str = 'relu', padding: str = 'same',
                      dropout: float = 0., verbose: bool = False) -> None:
        decoder_inputs = keras.Input(shape=(self._latent_space_size,))

        initial_filters_multiplier = int(self._input_shape[0] / (expansion_layers_number*stride))
        x = layers.Dense(initial_filters_multiplier * initial_filters_multiplier * initial_filters_number,
                         activation=activation)(decoder_inputs)
        x = layers.Reshape((initial_filters_multiplier, initial_filters_multiplier, initial_filters_number))(x)

        layer_filters = initial_filters_number
        for l in range(expansion_layers_number):
            x = layers.Conv2DTranspose(layer_filters, kernel_size,
                                       activation=activation,
                                       strides=stride,
                                       padding=padding)(x)
            layer_filters = int(initial_filters_number /
                                filter_layers_multiplier)

        x = layers.Conv2DTranspose(1, kernel_size, activation="sigmoid", padding=padding)(x)
        self._decoder = keras.Model(decoder_inputs, x, name="decoder")

        if verbose:
            self._decoder.summary()

    def define_reconstruction_loss(self, predict_value: tf.Tensor,
                                   true_value: tf.Tensor) -> tf.Tensor:
        loss = tf.losses.mean_squared_error(predict_value, true_value)
        loss *= np.prod(list(self._input_shape))
        return loss

    def define_kl_loss(self, mean: tf.Tensor, log_var: tf.Tensor) -> tf.Tensor:
        loss = 1 + log_var - tf.square(mean) - tf.exp(log_var)
        loss = -0.5 * tf.reduce_mean(loss)
        return loss

    @tf.function
    def call(self, input_tensor: tf.Tensor) -> tf.Tensor:
        z = self._encoder(input_tensor)
        return self._decoder(z)

    @tf.function
    def encode(self, input_tensor: tf.Tensor) -> tf.Tensor:
        return self._encoder(input_tensor)

    @tf.function
    def train_step(self, data: tf.Tensor) -> Dict[str, tf.Tensor]:
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self._encoder(data)
            reconstruction = self._decoder(z)
            rc_loss = self.define_reconstruction_loss(
                predict_value=reconstruction, true_value=data)
            kl_loss = self.define_kl_loss(mean=z_mean, log_var=z_log_var)
            loss = self._reconstrunction_loss_weight * rc_loss + self._kl_divergence_loss_weight * kl_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": loss,
            "rc_loss": rc_loss,
            "kl_loss": kl_loss,
        }

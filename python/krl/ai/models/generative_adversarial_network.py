'''Generative Adversarial Network (CGAN) class implementation'''

import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from typing import Any, Dict, Optional, Tuple


__all__ = ['CGAN']


class CGAN(keras.Model):
    _latent_input_shape: Tuple[int, ...]
    _condition_input_size: Tuple[int, ...]
    _batch_size: int
    _noise_size: int
    _label_smoothing_factor: float
    _discriminator: keras.Model
    _generator: keras.Model
    _discriminator_optimizer: tf.keras.optimizers.Adam
    _generator_optimizer: tf.keras.optimizers.Adam

    def __init__(self, latent_input_shape: Tuple[int, ...], condition_input_size: int,
                 batch_size: int, noise_size: int, label_smoothing_factor: float,
                 verbose: bool = False, **kwargs: Any) -> None:
        super(CGAN, self).__init__(**kwargs)
        assert label_smoothing_factor >= 0., f'label_smoothing_factor cannot be < 0. Provided {label_smoothing_factor}.'
        assert noise_size >= 0., f'noise_size cannot be < 0. Provided {noise_size}.'
        assert batch_size >= 0., f'batch_size cannot be < 0. Provided {batch_size}.'
        assert condition_input_size > 0., f'condition_input_size cannot be <= 0. Provided {condition_input_size}.'

        self._discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
        self._generator_optimizer = tf.keras.optimizers.Adam(1e-4)

        self._latent_input_shape = latent_input_shape
        self._condition_input_size = condition_input_size
        self._batch_size = batch_size
        self._noise_size = noise_size
        self._label_smoothing_factor = label_smoothing_factor

        discriminator_initial_filters_number = 32
        generator_initial_filters_number = 128
        discriminator_layers_number = 2
        generator_layers_number = 2
        filter_layers_multiplier = 2
        kernel_size = 5

        self._discriminator = self.build_discriminator(input_shape=self._latent_input_shape,
                                                       condition_size=self._condition_input_size,
                                                       initial_filters_number=discriminator_initial_filters_number,
                                                       discriminator_layers_number=discriminator_layers_number,
                                                       filter_layers_multiplier=filter_layers_multiplier,
                                                       kernel_size=kernel_size, verbose=verbose)

        self._generator = self.build_generator(input_shape=self._latent_input_shape,
                                               condition_size=self._condition_input_size,
                                               noise_shape=(self._noise_size,),
                                               initial_filters_number=generator_initial_filters_number,
                                               generator_layers_number=generator_layers_number,
                                               filter_layers_multiplier=filter_layers_multiplier,
                                               kernel_size=kernel_size, verbose=verbose)
        self.compile()

    def build_discriminator(self, input_shape: Tuple[int, ...], condition_size: Tuple[int, ...], initial_filters_number: int,
                            discriminator_layers_number: int, filter_layers_multiplier: int, kernel_size: int,
                            dropout: float = 0., stride: int = 2, activation: str = 'lrelu', padding: str = 'same',
                            verbose: bool = False) -> keras.Model:
        model_input = keras.Input(shape=input_shape)
        cond_input = keras.Input(shape=condition_size)

        cond_embedding = layers.Embedding(input_dim=condition_size, output_dim=np.sum(list(input_shape)))(cond_input)
        cond_embedding = layers.Dense(units=np.prod(list(input_shape)))(cond_embedding)
        cond_embedding = layers.Reshape(target_shape=input_shape)(cond_embedding)
        x = layers.Concatenate(axis=-1)([model_input, cond_embedding])

        layer_filters = initial_filters_number
        for l in range(discriminator_layers_number):
            x = layers.Conv2D(layer_filters, (kernel_size, kernel_size), strides=(stride, stride),
                              padding=padding, activation=activation if activation != 'lrelu' else None)(x)
            if activation == 'lrelu':
                x = layers.LeakyReLU()(x)
            if dropout > 0.:
                x = layers.Dropout(dropout)(x)

            layer_filters = layer_filters * filter_layers_multiplier

        x = layers.Flatten()(x)
        x = layers.Dense(1)(x)

        model = keras.Model([model_input, cond_input], [x], name="discriminator")

        if verbose:
            model.summary()

        return model

    def build_generator(self, input_shape: Tuple[int, ...], condition_size: Tuple[int, ...], noise_shape: Tuple[int, ...],
                        initial_filters_number: int, generator_layers_number: int, filter_layers_multiplier: int,
                        kernel_size: int, dropout: float = 0., stride: int = 2, activation: str = 'lrelu',
                        padding: str = 'same', verbose: bool = False) -> keras.Model:
        latent_input = keras.Input(shape=noise_shape)
        cond_input = keras.Input(shape=condition_size)

        cond_embedding = layers.Embedding(input_dim=condition_size, output_dim=np.sum(list(noise_shape)))(cond_input)
        cond_embedding = layers.Dense(units=np.prod(list(noise_shape)))(cond_embedding)
        cond_embedding = layers.Reshape(target_shape=noise_shape)(cond_embedding)
        x = layers.Concatenate(axis=-1)([latent_input, cond_embedding])

        initial_filters_multiplier = int(input_shape[0] / (generator_layers_number*stride))
        x = layers.Dense(initial_filters_multiplier*initial_filters_multiplier*initial_filters_number, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x)
        x = layers.Reshape((initial_filters_multiplier, initial_filters_multiplier, initial_filters_number))(x)

        layer_filters = initial_filters_number
        for l in range(generator_layers_number):
            layer_filters = layer_filters / filter_layers_multiplier
            x = layers.Conv2DTranspose(layer_filters, (kernel_size, kernel_size),
                                       strides=(1, 1) if l == 0 else (stride, stride), padding=padding, use_bias=False,
                                       activation=activation if activation != 'lrelu' else None)(x)
            x = layers.BatchNormalization()(x)
            if activation == 'lrelu':
                x = layers.LeakyReLU()(x)
            if dropout > 0.:
                x = layers.Dropout(dropout)(x)

        x = layers.Conv2DTranspose(1, (kernel_size, kernel_size), strides=(stride, stride),
                                   padding=padding, use_bias=False, activation='tanh')(x)

        model = keras.Model([latent_input, cond_input], [x], name="generator")

        if verbose:
            model.summary()

        return model

    def cross_entropy(self) -> Any:
        return tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def define_discriminator_loss(self, true_value: tf.Tensor, predict_value: tf.Tensor) -> tf.Tensor:
        real_loss = self.cross_entropy()(tf.ones_like(true_value) * (1. - self._label_smoothing_factor), true_value)
        fake_loss = self.cross_entropy()(tf.zeros_like(predict_value) * self._label_smoothing_factor, predict_value)
        lss = real_loss + fake_loss
        return lss

    def define_generator_loss(self, predict_value: tf.Tensor) -> tf.Tensor:
        return self.cross_entropy()(tf.ones_like(predict_value), predict_value)

    @tf.function
    def train_step(self, data: Tuple[tf.Tensor, tf.Tensor]) -> Dict[str, tf.Tensor]:
        x_data, y_data = data
        noise = tf.random.normal([self._batch_size, self._noise_size])

        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            x_generated = self._generator([noise, y_data], training=True)

            real_output = self._discriminator([x_data, y_data], training=True)
            fake_output = self._discriminator([x_generated, y_data], training=True)

            d_loss = self.define_discriminator_loss(real_output, fake_output)
            g_loss = self.define_generator_loss(fake_output)

        d_grads = d_tape.gradient(d_loss, self._discriminator.trainable_variables)
        g_grads = g_tape.gradient(g_loss, self._generator.trainable_variables)

        self._discriminator_optimizer.apply_gradients(zip(d_grads, self._discriminator.trainable_variables))
        self._generator_optimizer.apply_gradients(zip(g_grads, self._generator.trainable_variables))

        return {
            "D-loss": d_loss,
            "G-loss": g_loss,
        }

    def generate(self, y: tf.Tensor) -> tf.Tensor:
        y_noise = tf.random.normal([tf.shape(y)[0], self._noise_size])
        return self._generator([y_noise, y], training=False)

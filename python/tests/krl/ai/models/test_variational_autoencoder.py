from guess.models import VariationalAutoEncoder
from tensorflow import keras
import tensorflow as tf
import numpy as np
import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def test_vae_convergence_on_mnist():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    _ = [tf.config.experimental.set_memory_growth(d, True)
         for d in physical_devices]
    tf.config.experimental_run_functions_eagerly(False)

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data(path='/tmp/mnist.npz')
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_digits = np.expand_dims(mnist_digits, -1).astype(np.float32) / 255
    mnist_digits = mnist_digits[:256*3]

    vae = VariationalAutoEncoder(latent_space_size=6, input_shape=(28, 28, 1),
                                 initial_compression_filters_number=32,
                                 compression_layers_number=2,
                                 initial_expansion_filters_number=64,
                                 expansion_layers_number=2,
                                 filter_layers_multiplier=2,
                                 kernel_size=3, dropout=0.,
                                 verbose=False)

    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))
    history = vae.fit(mnist_digits, epochs=100, batch_size=256)

    # lss = history.history['loss']
    predictions = vae.predict(mnist_digits[110:111])
    # np.save('/tmp/pred0.npy', (predictions[0] * 255).astype(np.uint8))
    # np.save('/tmp/true0.npy', (mnist_digits[110] * 255).astype(np.uint8))

    np.testing.assert_array_less(np.abs(np.mean(predictions - mnist_digits)), 0.1)

import pytest
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import tensorflow as tf

from tensorflow import keras

from guess.models import CGAN

def test_gan_convergence_on_mnist():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    _ = [tf.config.experimental.set_memory_growth(d, True)
         for d in physical_devices]
    tf.config.experimental_run_functions_eagerly(True)

    batch_size = 32
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path='/tmp/mnist.npz')
    mnist_digits = np.concatenate([x_train, x_test], axis=0)
    mnist_labels = np.expand_dims(np.concatenate([y_train, y_test], axis=0), -1).astype("float32")
    mnist_digits = (np.expand_dims(mnist_digits, -1).astype("float32") - 127.5) / 127.5
    mnist_digits = mnist_digits[:256*27]
    mnist_labels = mnist_labels[:256*27]

    gan = CGAN(latent_input_shape=(28, 28, 1), condition_input_size=10,
               batch_size=batch_size, noise_size=10, label_smoothing_factor=0., verbose=True)

    history = gan.fit(mnist_digits, mnist_labels, epochs=5, batch_size=batch_size)
    pred0 = gan.generate(tf.constant(np.array([[3]], dtype=np.float32))).numpy()
    pred1 = gan.generate(tf.constant(np.array([[5]], dtype=np.float32))).numpy()

    # np.save('/tmp/true0.npy', ((mnist_digits[0] * 127.5) + 127.5).astype(np.uint8))
    # np.save('/tmp/pred0.npy', ((pred0[0] * 127.5) + 127.5).astype(np.uint8))
    # np.save('/tmp/pred1.npy', ((pred1[0] * 127.5) + 127.5).astype(np.uint8))
    np.testing.assert_array_less(np.abs(np.mean(pred1 - mnist_digits[0])), 0.1)

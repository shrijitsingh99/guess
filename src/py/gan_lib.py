#!/usr/bin/env python
# coding: utf-8

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, Flatten, Reshape, multiply
from keras.layers import Conv1D, Conv2DTranspose, UpSampling1D, UpSampling2D, LSTM
from keras.layers import LeakyReLU, Dropout, Lambda
from keras.layers import BatchNormalization, Dense
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, RMSprop, SGD
from keras.utils import plot_model
from keras import backend as K

from utils import LaserScans
from autoencoder_lib import AutoEncoder

class GAN:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.discriminator_net = None
        self.generator_net = None
        self.discriminator = None
        self.adversarial = None

    def discriminator_network(self):
        if self.discriminator_net is None:
            depth = 32
            dropout = 0.4

            self.discriminator_net = Sequential()
            self.discriminator_net.add(Dense(depth, input_shape=self.discriminator_input_shape))
            self.discriminator_net.add(LeakyReLU(alpha=0.2))
            # self.discriminator_net.add(Dropout(dropout))
            # self.discriminator_net.add(Dense(int(depth/2)))
            # self.discriminator_net.add(LeakyReLU(alpha=0.2))
            # self.discriminator_net.add(Dense(int(depth/4)))
            # self.discriminator_net.add(LeakyReLU(alpha=0.2))
            # self.discriminator_net.add(Dropout(dropout))
            self.discriminator_net.add(Dense(1))
            self.discriminator_net.add(Activation('sigmoid'))

            if self.verbose:
                self.discriminator_net.summary()

        return self.discriminator_net

    def generator_network(self):
        if self.generator_net is None:
            dropout = 0.4
            depth = 64+64
            dim = 16

            self.generator_net = Sequential()
            if self.model_id == "conv":
                self.generator_net.add(Dense(depth, input_shape=self.genenerator_input_shape))
                self.generator_net.add(BatchNormalization(momentum=0.9))
                self.generator_net.add(LeakyReLU(alpha=0.2))
                self.generator_net.add(Reshape((8, dim, int(depth/dim))))
                self.generator_net.add(Dropout(dropout))
                self.generator_net.add(UpSampling2D(size=(2, 1)))
                self.generator_net.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
                self.generator_net.add(LeakyReLU(alpha=0.2))
                self.generator_net.add(UpSampling2D(size=(2, 1)))
                self.generator_net.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
                self.generator_net.add(LeakyReLU(alpha=0.2))
                self.generator_net.add(UpSampling2D(size=(2, 1)))
                self.generator_net.add(Conv2DTranspose(1, 5, padding='same'))
                self.generator_net.add(Flatten())
                self.generator_net.add(Dense(self.discriminator_input_shape[0]))
                self.generator_net.add(Activation('tanh'))

            elif self.model_id == "lstm":
                self.generator_net.add(LSTM(depth, input_shape=self.genenerator_input_shape,
                                            return_sequences=True, activation='tanh',
                                            recurrent_activation='hard_sigmoid'))
                self.generator_net.add(LSTM(int(depth/2), return_sequences=True,
                                            activation='tanh', recurrent_activation='hard_sigmoid'))
                self.generator_net.add(Dense(depth))
                self.generator_net.add(BatchNormalization(momentum=0.9))
                self.generator_net.add(LeakyReLU(alpha=0.2))
                self.generator_net.add(Dropout(dropout))
                self.generator_net.add(UpSampling1D(4))
                self.generator_net.add(Conv1D(int(depth/4), 5, padding='same'))
                self.generator_net.add(BatchNormalization(momentum=0.9))
                self.generator_net.add(LeakyReLU(alpha=0.2))
                self.generator_net.add(Flatten())
                self.generator_net.add(Dense(self.discriminator_input_shape[0]))
                self.generator_net.add(Activation('tanh'))

            else:
                depth = 128
                self.generator_net.add(Flatten(input_shape=self.genenerator_input_shape))
                self.generator_net.add(Dense(int(depth/2)))
                self.generator_net.add(LeakyReLU(alpha=0.2))
                # self.generator_net.add(Dense(int(depth/4)))
                # self.generator_net.add(LeakyReLU(alpha=0.2))
                self.generator_net.add(Dense(self.discriminator_input_shape[0]))
                self.generator_net.add(Activation('tanh'))

            if self.verbose:
                self.generator_net.summary()

        return self.generator_net

    def discriminator_model(self, lr=0.1):
        if self.discriminator is None:
            self.discriminator = Sequential()
            self.discriminator.add(self.discriminator_network())

            optimizer = Adam(lr=lr, beta_1=0.5, decay=3e-8)
            self.discriminator.compile(loss='binary_crossentropy',
                                       optimizer=optimizer, metrics=['accuracy'])
        return self.discriminator

    def adversarial_model(self, lr=0.1):
        if self.adversarial is None:
            self.adversarial = Sequential()
            self.adversarial.add(self.generator_network())
            d = self.discriminator_network()
            d.trainable = False
            self.adversarial.add(d)

            optimizer = Adam(lr=lr, beta_1=0.5, decay=3e-8)
            self.adversarial.compile(loss='binary_crossentropy',
                                     optimizer=optimizer, metrics=['accuracy'])
        return self.adversarial

    def set_trainable(self, net, tr=False):
        net.trainable = tr
        for l in net.layers:
            l.trainable = tr
        return net

    def build_model(self, discriminator_input_shape, genenerator_input_shape,
                    discriminator_lr=0.0002, generator_lr=0.002,
                    smoothing_factor=0.05, noise_dim=5, noise_magnitude=2., model_id="afmk"):
        assert noise_dim > 0, 'Noise dimension must be graeater than 1.'

        self.model_id = model_id
        self.noise_dim = noise_dim
        self.noise_magnitude = noise_magnitude
        self.smoothing_factor = smoothing_factor
        self.discriminator_input_shape = discriminator_input_shape
        self.genenerator_input_shape = (genenerator_input_shape[0], genenerator_input_shape[1] + noise_dim)

        self.discriminator = self.discriminator_model(lr=discriminator_lr)
        self.generator_net = self.generator_network()
        self.adversarial = self.adversarial_model(lr=generator_lr)


    def train(self, x, y, batch_sz=32, train_steps=10, verbose=None):
        assert x.shape[0] == y.shape[0], "Wrong input size."
        verbose = self.verbose if verbose is None else verbose

        dataset_n_samples = int(x.shape[0]/batch_sz)*batch_sz
        x = x[:dataset_n_samples]
        y = y[:dataset_n_samples]
        noise = (np.random.rand(dataset_n_samples, self.genenerator_input_shape[0],
                                self.noise_dim)*self.noise_magnitude) - 0.5*self.noise_magnitude
        xn = np.concatenate([x, noise], axis=-1)

        target_label = np.ones((batch_sz, 1))
        real_label = 1. - np.random.uniform(0.0, self.smoothing_factor, size=((dataset_n_samples, 1)))
        fake_label = np.random.uniform(0.0, self.smoothing_factor, size=((dataset_n_samples, 1)))

        metrics = []
        for t in range(train_steps):
            d_loss, d_acc, a_loss, a_acc = 0., 0., 0., 0.

            for b in range(0, dataset_n_samples, batch_sz):
                batch_slice = slice(b, b + batch_sz, None)
                real = y[batch_slice]
                fake = self.generator_net.predict(xn[batch_slice])

                # fit discriminator
                for i in range(1):
                    d_loss_real = self.discriminator.train_on_batch(real, real_label)
                    d_loss_fake = self.discriminator.train_on_batch(fake, fake_label)
                    d_loss, d_acc = (0.5*np.add(d_loss_real, d_loss_fake)).tolist()

                # fit generator
                a_loss, a_acc = self.adversarial.train_on_batch(xn[batch_slice], target_label)

            metrics.append([d_loss, d_acc, a_loss, a_acc])
            if verbose:
                log_msg = "-- %d/%d: [D loss: %f, acc: %f] - [A loss: %f, acc: %f]" % (b + batch_sz,
                                                                                       dataset_n_samples,
                                                                                       d_loss, d_acc, a_loss, a_acc)
                print(log_msg)
                sys.stdout.write("\033[F\033[K")

        if verbose:
            print(log_msg)

        return np.array(metrics)

    def generate(self, x):
        noise = (np.random.rand(self.genenerator_input_shape[0],
                                self.noise_dim)*self.noise_magnitude) - 0.5*self.noise_magnitude
        xn = np.expand_dims(np.concatenate([x, noise], axis=-1), axis=0)
        return self.generator_net.predict(xn)[0]


if __name__ == "__main__":
    # params
    scan_n = 34*8
    scan_to_predict_idx = 16
    scan_beam_num = 512
    latent_dim = 32
    correlated_sequence_step = 8
    prediction_step = 8
    ae_batch_sz = 64
    gan_batch_sz = 32

    # diag_first_floor.txt ; diag_underground.txt ; diag_labrococo.txt
    cwd = os.path.dirname(os.path.abspath(__file__))
    dataset_file = os.path.join(os.path.join(cwd, "../../dataset/"), "diag_underground.txt")

    # ---- laser-scans
    ls = LaserScans(verbose=True)
    ls.load(dataset_file, scan_res=0.00653590704, scan_fov=(3/2)*np.pi, scan_beam_num=scan_beam_num,
            clip_scans_at=8, scan_offset=8)
    scans = ls.getScans()[:scan_n]
    cmdv = ls.cmdVel()[:scan_n, ::5]
    cmdv_dim = cmdv.shape[-1]

    correlated_steps = slice((scan_to_predict_idx - (prediction_step + correlated_sequence_step)),
                             (scan_to_predict_idx - prediction_step), None)
    scan_to_predict = scans[scan_to_predict_idx]
    correlated_scan_sequence = scans[correlated_steps]
    correlated_cmdv_sequence = cmdv[correlated_steps]

    rnd_indices = np.arange(scans.shape[0])
    np.random.shuffle(rnd_indices)

    # ---- autoencoder
    ae = AutoEncoder(ls.originalScansDim(), variational=True, convolutional=False,
                     batch_size=ae_batch_sz, latent_dim=latent_dim, verbose=False)
    ae.build_model()
    print('-- Fitting VAE model done.')

    ae.train(scans[rnd_indices], x_test=None, epochs=10, verbose=0)
    encoded_scans = ae.encode(scans)
    decoded_scans = ae.decode(encoded_scans)

    rnd_idx = int(np.random.rand() * scan_n)
    # ls.plotScan(scans[rnd_idx], decoded_scans[rnd_idx])
    # ls.plotScan(scan_to_predict, decoded_scans[scan_to_predict_idx])
    # plt.show()

    # ---- gan - generating scans
    correlated_latent = np.concatenate((ae.encode(correlated_scan_sequence),
                                        correlated_cmdv_sequence), axis=-1)

    # gan = GAN(verbose=True)
    # gan.build_model(discriminator_input_shape=(ls.originalScansDim(),),
    #                genenerator_input_shape=(correlated_sequence_step, latent_dim + cmdv_dim),
    #                smoothing_factor=0.03, noise_dim=1, model_id="afmk")

    # latent = np.concatenate([encoded_scans, cmdv], axis=-1)
    # gan_x = latent.reshape((-1, correlated_sequence_step, latent_dim + cmdv_dim))
    # gan_y = scans[(correlated_sequence_step + prediction_step)::correlated_sequence_step]
    # dataset_dim = min(gan_x.shape[0], gan_y.shape[0])
    # gan_x = gan_x[:dataset_dim]
    # gan_y = 2*gan_y[:dataset_dim] - 1.0

    # rnd_indices = np.arange(dataset_dim)
    # np.random.shuffle(rnd_indices)
    # gan.train(gan_x[rnd_indices], gan_y[rnd_indices], train_steps=20, batch_sz=gan_batch_sz, verbose=True)

    # # for i in range(1, 8):
    # #     ls.plotScan(correlated_scan_sequence[0], correlated_scan_sequence[i])
    # #     plt.show()

    # ls.plotScan(correlated_scan_sequence[0], scan_to_predict)
    # ls.plotScan(0.5*(gan.generate(correlated_latent) + 1.0), scan_to_predict)

    # del gan

    # ---- gan - generating latent spaces
    gan = GAN(verbose=True)
    gan.build_model(discriminator_input_shape=(latent_dim,),
                    genenerator_input_shape=(correlated_sequence_step, latent_dim + cmdv_dim),
                    discriminator_lr=0.002, generator_lr=0.2, smoothing_factor=0.0,
                    noise_dim=1, noise_magnitude=0.1, model_id="afmk")

    latent = np.concatenate([encoded_scans, cmdv], axis=-1)
    gan_x = latent.reshape((-1, correlated_sequence_step, latent_dim + cmdv_dim))
    gan_y = encoded_scans[(correlated_sequence_step + prediction_step)::correlated_sequence_step]
    # [plt.plot(gan_y[int(np.random.rand() * gan_y.shape[0])]) for _ in range(10)]

    dataset_dim = min(gan_x.shape[0], gan_y.shape[0])
    gan_x = gan_x[:dataset_dim]
    gan_y = gan_y[:dataset_dim]

    rnd_indices = np.arange(dataset_dim)
    np.random.shuffle(rnd_indices)
    metrics = gan.train(gan_x[rnd_indices], gan_y[rnd_indices], train_steps=1000, batch_sz=gan_batch_sz, verbose=True)

    gen_latent = gan.generate(correlated_latent)
    gen_scan = 0.5*(ae.decode(gen_latent.reshape((1, latent_dim)))[0] + 1.0)

    plt.plot(metrics[:, 0])
    plt.plot(metrics[:, 2])

    plt.figure()
    plt.plot(gen_latent)
    plt.plot(encoded_scans[scan_to_predict_idx])

    ls.plotScan(gen_scan, decoded_scans[scan_to_predict_idx])
    # ls.plotScan(gen_scan, scan_to_predict)
    plt.show()

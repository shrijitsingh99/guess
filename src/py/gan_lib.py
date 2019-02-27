#!/usr/bin/env python
# coding: utf-8

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, Flatten, Reshape
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
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model

    def discriminator(self):
        if self.D: return self.D
        depth = 64
        dropout = 0.4

        self.D = Sequential()
        self.D.add(Dense(depth, input_shape=self.dis_input_shape))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Dense(int(depth/2)))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dense(int(depth/4)))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        if self.verbose: self.D.summary()
        return self.D

    def generator(self):
        if self.G: return self.G
        dropout = 0.4
        depth = 64+64
        dim = 16

        self.G = Sequential()
        if self.model_id == "conv":
            self.G.add(Dense(depth, input_shape=self.gen_input_shape))
            self.G.add(BatchNormalization(momentum=0.9))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Reshape((8, dim, int(depth/dim))))
            self.G.add(Dropout(dropout))
            self.G.add(UpSampling2D(size=(2, 1)))
            self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(UpSampling2D(size=(2, 1)))
            self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(UpSampling2D(size=(2, 1)))
            self.G.add(Conv2DTranspose(1, 5, padding='same'))
            self.G.add(Flatten())
            self.G.add(Dense(self.dis_input_shape[0]))
            self.G.add(Activation('tanh'))

        elif self.model_id == "lstm":
            self.G.add(LSTM(depth, input_shape=self.gen_input_shape,
                            return_sequences=True, activation='tanh',
                            recurrent_activation='hard_sigmoid'))
            self.G.add(LSTM(int(depth/2), return_sequences=True,
                            activation='tanh', recurrent_activation='hard_sigmoid'))
            self.G.add(Dense(depth))
            self.G.add(BatchNormalization(momentum=0.9))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Dropout(dropout))
            self.G.add(UpSampling1D(4))
            self.G.add(Conv1D(int(depth/4), 5, padding='same'))
            self.G.add(BatchNormalization(momentum=0.9))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Flatten())
            self.G.add(Dense(self.dis_input_shape[0]))
            self.G.add(Activation('tanh'))

        else:
            self.G.add(Dense(int(depth/8), input_shape=self.gen_input_shape))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Dense(int(depth/4)))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Dense(int(depth/2)))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Flatten())
            self.G.add(Dense(self.dis_input_shape[0]))
            self.G.add(Activation('tanh'))

        if self.verbose: self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM: return self.DM
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        optimizer = Adam(lr=0.002, beta_1=0.5, decay=3e-8)
        self.DM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM: return self.AM
        self.AM = Sequential()
        self.AM.add(self.generator())
        disc = self.discriminator()
        disc.trainable = False
        self.AM.add(disc)
        optimizer = Adam(0.0002, beta_1=0.5, decay=3e-8)
        self.AM.compile(loss='mse',
                        optimizer=optimizer, metrics=['accuracy'])
        if self.verbose: self.AM.summary()
        return self.AM

    def setTrainable(self, net, tr=False):
        net.trainable = tr
        for l in net.layers: l.trainable = tr
        return net

    def buildModel(self, dis_input_shape, gen_input_shape,
                   model_id="afmk", noise_dim=4, smoothing_factor=0.0):
        self.dis_input_shape = dis_input_shape
        self.model_id = model_id
        self.noise_dim = noise_dim
        self.gen_input_shape = (gen_input_shape[0], gen_input_shape[1] + noise_dim)
        self.smoothing_factor = smoothing_factor
        self.DIS = self.discriminator_model()
        self.GEN = self.generator()
        self.ADV = self.adversarial_model()

    def fitModel(self, x, x_label, train_steps=10, batch_sz=32, verbose=None):
        assert x.shape[0] == x_label.shape[0], "wrong input size"
        if verbose is None: verbose = self.verbose
        ret = []
        in_shape = (batch_sz, self.gen_input_shape[0], self.gen_input_shape[1])
        for b in range(0, x.shape[0], batch_sz):
            if b + batch_sz > x.shape[0]: continue
            for t in range(train_steps):
                xn = np.empty(in_shape)
                if self.noise_dim > 0:
                    noise = np.random.normal(0.0, 1.0,
                                             size=(in_shape[0], in_shape[1], self.noise_dim))
                    xn[:, :, :-self.noise_dim] = x[b:b + batch_sz]
                    xn[:, :, -self.noise_dim:] = noise
                else:
                    xn = x[b:b + batch_sz]

                real = x_label[b:b + batch_sz]
                fake = self.GEN.predict(xn)
                if self.smoothing_factor != 0.0:
                    real_label = 1.0 - np.random.uniform(0.0, self.smoothing_factor, size=((batch_sz, 1)))
                    fake_label = np.random.uniform(0.0, self.smoothing_factor, size=((batch_sz, 1)))
                else:
                    real_label = np.ones((batch_sz, 1))
                    fake_label = np.zeros((batch_sz, 1))

                # fit discriminator
                for i in range(2):
                    d_real_l = self.DIS.train_on_batch(real, real_label)
                    d_fake_l = self.DIS.train_on_batch(fake, fake_label)
                    d_loss = 0.5*np.add(d_real_l, d_fake_l)

                # fit generator
                a_loss = self.ADV.train_on_batch(xn, np.ones((batch_sz, 1)))

                if t == train_steps - 1:
                    ret.append([d_loss[0], d_loss[1], a_loss[0], a_loss[1]])
                    if verbose:
                        log_mesg = "-- %d/%d: [D loss: %f, acc: %f]" % \
                                   ((b + batch_sz)/batch_sz, x.shape[0]/batch_sz, d_loss[0], d_loss[1])
                        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                        print(log_mesg)
                        sys.stdout.write("\033[F\033[K")
        if verbose: print(log_mesg)
        return np.array(ret)

    def generate(self, x):
        xn = np.empty((1, self.gen_input_shape[0], self.gen_input_shape[1]))
        if self.noise_dim > 0:
            noise = np.random.normal(0.0, 1.0, size=(self.gen_input_shape[0], self.noise_dim))
            xn[0, :, :-self.noise_dim] = x
            xn[0, :, -self.noise_dim:] = noise
            return self.GEN.predict(xn)[0]
        else:
            xn[0] = x
            return self.GEN.predict(xn)[0]

if __name__ == "__main__":
    # params
    scan_idx = 1000
    to_show_idx = 10
    ae_latent_dim = 10
    gan_sequence = 8
    gan_batch_sz = 32
    gan_pred_step = 8
    n_steps = 2

    # diag_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    ls = LaserScans(verbose=True)
    ls.load("../../dataset/diag_underground.txt",
            scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
            scan_beam_num=512, clip_scans_at=8, scan_offset=8)
    scans = ls.getScans()

    ae = AutoEncoder(ls.originalScansDim(),
                     variational=True, convolutional=False,
                     batch_size=128, latent_dim=ae_latent_dim, verbose=False)
    ae.buildModel()

    ae.fitModel(scans[:2000], x_test=None, epochs=40, verbose=0)
    print('-- Fitting VAE model done.')

    scans_ae = scans[scan_idx:(scan_idx + gan_sequence*gan_batch_sz)]
    latent = ae.encode(scans_ae)
    dscan = ae.decode(latent)
    scan_to_predict = scans_ae[gan_sequence + gan_pred_step]
    ls.plotScan(scan_to_predict, dscan[gan_sequence + gan_pred_step])

    gan = GAN(verbose=True)
    gan.buildModel((ls.originalScansDim(),), (gan_sequence, ae_latent_dim + 6),
                   smoothing_factor=0.1, noise_dim=1, model_id="afmk")

    scan_num = 13000
    scans_gan = scans[:scan_num]
    cmdv = ls.cmdVel()[:scan_num]
    ae_encoding = ae.encode(scans_gan)
    latent = np.concatenate((ae_encoding, cmdv), axis=1)
    in_latent = latent.reshape((int(latent.shape[0]/gan_sequence), gan_sequence, latent.shape[1]))

    in_label = scans_gan[(gan_sequence + gan_pred_step)::gan_sequence]
    in_label = 2*in_label - 1.0
    in_latent = in_latent[:in_label.shape[0]]

    for i in range(n_steps):
        print('-- step %d: Fitting GAN...' % i)
        gan.fitModel(in_latent, in_label, train_steps=10, batch_sz=gan_batch_sz, verbose=True)
        if i % 3 == 0:
            gs = gan.generate(latent[:gan_sequence]) + 1.0
            ls.plotScan(0.5*gs)
    plt.show()

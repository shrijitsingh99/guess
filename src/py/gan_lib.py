#!/usr/bin/env python
# coding: utf-8

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
from keras.optimizers import Adam, RMSprop
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
            self.G.add(Conv1D(depth, 5,
                              input_shape=self.gen_input_shape, strides=2, padding='same'))
            self.G.add(BatchNormalization(momentum=0.9))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Dropout(dropout))
            self.G.add(Conv1D(depth*2, 5, strides=1, padding='same'))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Dropout(dropout))
            self.G.add(Flatten())
            self.G.add(Dense(int(self.dis_input_shape[0]/2)))
            self.G.add(LeakyReLU(alpha=0.2))
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
            self.G.add(Dense(int(depth)))
            self.G.add(LeakyReLU(alpha=0.2))
            self.G.add(Dense(self.dis_input_shape[0]))
            self.G.add(Activation('tanh'))

        if self.verbose: self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM: return self.DM
        optimizer = Adam(lr=0.001, decay=3e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM: return self.AM
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        optimizer = Adam(lr=0.001, decay=3e-8)
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        if self.verbose: self.AM.summary()
        return self.AM

    def setTrainable(self, net, tr=False):
        net.trainable = tr
        for l in net.layers: l.trainable = tr
        return net

    def buildModel(self, dis_input_shape, gen_input_shape,
                   model_id="lstm", smoothing_factor=0.0, noise_dim=4):
        self.noise_dim = noise_dim
        self.dis_input_shape = dis_input_shape
        self.gen_input_shape = (gen_input_shape[0], gen_input_shape[1] + noise_dim)
        self.smoothing_factor = smoothing_factor
        self.model_id = model_id
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
            for t in range(2*train_steps):
                noise = np.random.normal(0.0, 1.0, size=(in_shape[0], in_shape[1], self.noise_dim))
                xn = np.empty(in_shape)
                xn[:, :, :-self.noise_dim] = x[b:b + batch_sz]
                xn[:, :, -self.noise_dim:] = noise

                real = x_label[b:b + batch_sz]
                fake = self.GEN.predict(xn)
                x_train = np.vstack((real, fake))

                # label smoothing [todo]
                y = np.zeros((2*batch_sz, 1))
                y[:batch_sz] += 1.0

                if (t % 2) == 0:
                    # self.setTrainable(self.DIS, True)
                    for i in range(3): d_loss = self.DIS.train_on_batch(x_train, y)
                    # self.setTrainable(self.DIS, False)
                else:
                    y = np.ones([batch_sz, 1])
                    a_loss = self.ADV.train_on_batch(xn, y)

                if t == train_steps - 1:
                    ret.append([d_loss[0], d_loss[1], a_loss[0], a_loss[1]])
                    if verbose:
                        log_mesg = "-- %d/%d: [D loss: %f, acc: %f]" % \
                                   ((b + batch_sz)/batch_sz, x.shape[0]/batch_sz, d_loss[0], d_loss[1])
                        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                        print(log_mesg)
        return np.array(ret)

    def generate(self, x):
        noise = np.random.normal(0.0, 1.0, size=(self.gen_input_shape[0], self.noise_dim))
        xn = np.empty((1, self.gen_input_shape[0], self.gen_input_shape[1]))
        xn[0, :, :-self.noise_dim] = x
        xn[0, :, -self.noise_dim:] = noise
        return self.GEN.predict(xn)[0]

if __name__ == "__main__":
    # params
    scan_idx = 1000
    to_show_idx = 10
    ae_latent_dim = 20
    gan_sequence = 8
    gan_batch_sz = 32
    gan_pred_step = 8

    # diag_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    ls = LaserScans(verbose=True)
    ls.load("../../dataset/diag_underground.txt",
            scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
            scan_beam_num=512, clip_scans_at=8, scan_offset=8)
    x, x_test = ls.getScans(0.9)

    ae = AutoEncoder(ls.originalScansDim(),
                     variational=True, convolutional=False,
                     batch_size=128, latent_dim=ae_latent_dim, verbose=False)
    ae.buildModel()

    ae.fitModel(x[:2000], x_test=None, epochs=30, verbose=0)
    print('-- Fitting VAE model done.')

    scan = x[scan_idx:(scan_idx + gan_sequence*gan_batch_sz)]
    latent = ae.encode(scan)
    dscan = ae.decode(latent)
    scan_to_predict = scan[gan_sequence + gan_pred_step]
    ls.plotScan(scan_to_predict)

    gan = GAN(verbose=True)
    gan.buildModel((ls.originalScansDim(),), (gan_sequence, ae_latent_dim + 6),
                   smoothing_factor=0.1, model_id="afmk")

    scan_num = 10000
    scans = x[:scan_num]
    cmdv = ls.cmdVel()[:scan_num]
    ae_encoding = ae.encode(scans)
    latent = np.concatenate((ae_encoding, cmdv), axis=1)
    in_latent = latent.reshape((int(latent.shape[0]/gan_sequence), gan_sequence, latent.shape[1]))

    in_label = scans[(gan_sequence + gan_pred_step)::gan_sequence]
    in_label = 2*in_label - 1.0
    in_latent = in_latent[:in_label.shape[0]]

    gan.fitModel(in_latent, in_label, train_steps=30, batch_sz=gan_batch_sz, verbose=True)
    print('-- step 0: Fitting GAN done.\n')
    gs = gan.generate(latent[:gan_sequence]) + 1.0
    ls.plotScan(0.5*gs)

    gan.fitModel(in_latent, in_label, train_steps=30, batch_sz=gan_batch_sz, verbose=True)
    print('-- step 1: Fitting GAN done.\n')
    gs = gan.generate(latent[:gan_sequence]) + 1.0
    ls.plotScan(0.5*gs)

    # gan.fitModel(in_latent, in_label, train_steps=30, batch_sz=gan_batch_sz, verbose=True)
    # print('-- step 2: Fitting GAN done.\n')
    # gs = gan.generate(latent[:gan_sequence]) + 1.0
    # ls.plotScan(0.5*gs)
    plt.show()

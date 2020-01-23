#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, Flatten, Reshape
from keras.layers import Conv2D, Conv2DTranspose, UpSampling2D, LSTM, TimeDistributed
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers import Lambda, Input, Dense
from keras.models import Model, Sequential, clone_model
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras import backend as K

from utils import LaserScans

class AutoEncoder:
    def __init__(self, original_dim,
                 variational=True, convolutional=True,
                 batch_size=128, latent_dim=10, intermediate_dim=128, verbose=False):
        self.original_dim = original_dim
        self.variational = variational
        self.convolutional = convolutional
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        self.intermediate_dim = intermediate_dim
        self.verbose = verbose
        self.reshape_rows = 32
        self.encoder = None
        self.decoder = None
        self.pencoder = None
        self.pdecoder = None
        self.ae = None

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def __createEncoder(self, e_in, depth=32, dropout=0.4, tr=True):
        if self.convolutional:
            enc = Reshape((self.reshape_rows, int(self.original_dim/self.reshape_rows), 1,))(e_in)
            enc = Conv2D(depth, 5, activation='relu', strides=2, padding='same', trainable=tr)(enc)
            enc = Conv2D(depth*4, 5, strides=2, padding='same', trainable=tr)(enc)
            enc = LeakyReLU(alpha=0.2)(enc)
            enc = Dropout(dropout)(enc)
            # shape info needed to build decoder model
            self.enc_shape = K.int_shape(enc)
            # Out: 1-dim probability
            enc = Flatten()(enc)
        else:
            enc = e_in
        enc = Dense(self.intermediate_dim, activation='relu', trainable=tr)(enc)

        if self.variational:
            self.z_mean = Dense(self.latent_dim, trainable=tr, name='z_mean')(enc)
            self.z_log_var = Dense(self.latent_dim, trainable=tr, name='z_log_var')(enc)
            z = Lambda(self.sampling, output_shape=(self.latent_dim,),
                       name='z')([self.z_mean, self.z_log_var])
            encoder = Model(e_in, [self.z_mean, self.z_log_var, z], name='encoder')
        else:
            e_out = Dense(self.latent_dim, activation='sigmoid', trainable=tr)(enc)
            encoder = Model(e_in, e_out, name='encoder')
        return encoder

    def __createDecoder(self, d_in, depth=32, dropout=0.4, tr=True):
        if self.convolutional:
            dec = Dense(self.enc_shape[1]*self.enc_shape[2]*self.enc_shape[3],
                        activation='relu', trainable=tr)(d_in)
            dec = Reshape((self.enc_shape[1], self.enc_shape[2], self.enc_shape[3]))(dec)

            dec = Conv2DTranspose(filters=int(depth/2), kernel_size=5,
                              activation='relu', strides=2, padding='same', trainable=tr)(dec)
            dec = Conv2DTranspose(filters=int(depth/4), kernel_size=5,
                              activation='relu', strides=2, padding='same', trainable=tr)(dec)
            dec = Dropout(dropout)(dec)
            dec = Dense(int(depth/16), activation='relu', trainable=tr)(dec)
            dec = Flatten()(dec)
        else:
            dec = Dense(self.intermediate_dim, activation='relu', trainable=tr)(d_in)

        d_out = Dense(self.original_dim, activation='sigmoid', trainable=tr)(dec)
        decoder = Model(d_in, d_out, name='decoder')
        return decoder

    def buildModel(self):
        if not self.ae is None:
            return self.ae
        input_shape = (self.original_dim,)
        depth = 32
        dropout = 0.2

        ## ENCODER
        e_in = Input(shape=input_shape, name='encoder_input')
        self.encoder = self.__createEncoder(e_in, depth=depth, dropout=dropout, tr=True)
        self.pencoder = self.__createEncoder(e_in, depth=depth, dropout=dropout, tr=False)
        if self.verbose: self.encoder.summary()

        ## DECODER
        d_in = Input(shape=(self.latent_dim,), name='decoder_input')
        self.decoder = self.__createDecoder(d_in, depth=depth, dropout=dropout, tr=True)
        self.pdecoder = self.__createDecoder(d_in, depth=depth, dropout=dropout, tr=False)
        if self.verbose: self.decoder.summary()

        ## AUTOENCODER
        if self.variational:
            vae_out = self.decoder(self.encoder(e_in)[2])
            self.ae = Model(e_in, vae_out, name='vae_mlp')
            reconstruction_loss = mse(e_in, vae_out) # binary_crossentropy

            kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.square(K.exp(self.z_log_var))
            kl_loss = -0.5*K.sum(kl_loss, axis=-1)
            vae_loss = K.mean(reconstruction_loss + kl_loss)

            self.ae.add_loss(vae_loss)
            self.ae.compile(optimizer=Adam(lr=0.01), metrics=['accuracy'])
        else:
            vae_out = self.decoder(self.encoder(e_in))
            self.ae = Model(e_in, vae_out, name='autoencoder')
            self.ae.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
        if self.verbose: self.ae.summary()
        return self.ae

    def fitModel(self, x, x_test=None, epochs=10, verbose=None):
        if not x_test is None:
            x_test = (x_test, None)
        ret = []

        if self.variational:
            met = self.ae.fit(x, epochs=epochs, batch_size=self.batch_size,
                              shuffle=True, validation_data=x_test, verbose=verbose)
            ret = [[l, -1] for l in met.history['loss']]
        else:
            for e in range(epochs):
                for i in range(0, x.shape[0] - self.batch_size, self.batch_size):
                    met = self.ae.train_on_batch(x[i:i + self.batch_size], x[i:i + self.batch_size])
                    ret.append(met)

        self.pencoder.set_weights(self.encoder.get_weights())
        self.pdecoder.set_weights(self.decoder.get_weights())
        ret_avgs = np.mean(ret, axis=0)
        return np.array(ret_avgs)

    def encode(self, x, batch_size=None):
        if len(x.shape) == 1: x = np.array([x])
        if self.variational:
            z_mean, _, _ = self.encoder.predict(x, batch_size=batch_size)
            return z_mean
        else:
            return self.encoder.predict(x, batch_size=batch_size)

    def decode(self, z_mean):
        return self.decoder.predict(z_mean)

if __name__ == "__main__":
    batch_sz = 8
    scan_idx = 8000
    to_show_idx = 10
    gan_batch_sz = 32

    # diag_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    ls = LaserScans(verbose=True)
    ls.load("../../dataset/diag_first_floor.txt",
            scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
            scan_beam_num=512, clip_scans_at=8, scan_offset=8)
    x, x_test = ls.getScans(0.9)

    ae = AutoEncoder(ls.originalScansDim(), variational=True, convolutional=False,
                     batch_size=128, latent_dim=20, verbose=True)
    ae.buildModel()

    ae.fitModel(x[:scan_idx], x_test=None, epochs=3, verbose=0)
    print('-- step 0: Fitting VAE model done.')

    scan = x[scan_idx:(scan_idx + batch_sz*gan_batch_sz)]
    dscan = ae.decode(ae.encode(scan))
    # ls.plotScan(scan[to_show_idx], dscan[to_show_idx])

    ae.fitModel(x[scan_idx:scan_idx + 6000], x_test=None, epochs=30, verbose=0)
    print('-- step 1: Fitting VAE model done.')

    dscan = ae.decode(ae.encode(scan))
    ls.plotScan(scan[to_show_idx], dscan[to_show_idx])
    plt.show()

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
from keras.models import Model, Sequential
from keras.losses import mse, binary_crossentropy
from keras.optimizers import Adam, RMSprop
from keras.utils import plot_model
from keras import backend as K

class ElapsedTimer:
    def __init__(self):
        self.start_time = time.time()
    def __elapsed(self,sec):
        if sec < 60:
            return str(round(sec, 1)) + " sec"
        elif sec < (60 * 60):
            return str(int(sec / 60)) + " min"
        else:
            return str(int(sec / (60 * 60))) + " hr"
    def elapsed_time(self):
        return self.__elapsed(time.time() - self.start_time)

class LaserScans:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.ts = None
        self.cmd_vel = None
        self.scans = None
        self.scan_bound = 0
        self.scan_fov = (3/2)*np.pi  # [270 deg]
        self.scan_res = 0.001389*np.pi # [0.25 deg]
        self.scan_offset = 0

    def load(self, datafile, scan_res, scan_fov,
             scan_beam_num=None, clip_scans_at=None, scan_offset=0):
        self.data = np.loadtxt(datafile).astype('float32')
        self.scan_res = scan_res
        self.scan_fov = scan_fov
        self.scan_beam_num = scan_beam_num
        self.clip_scans_at = clip_scans_at
        self.scan_offset = scan_offset

        self.ts = self.data[:, :1]
        self.cmd_vel = self.data[:, 1:7]
        self.scans = self.data[:, 7:]
        if self.verbose:
            print("timesteps --", self.ts.shape)
            print("cmd_vel --", self.cmd_vel.shape)
            print("scans --", self.scans.shape,
                  "range [", np.min(self.scans), "-", np.max(self.scans), "]")

        irange = 0
        beam_num = int(self.scan_fov/self.scan_res)
        assert beam_num == self.scans.shape[1], \
            "Wrong number of scan beams " + str(beam_num) + " != " + str(self.scans.shape[1])
        if not self.scan_beam_num is None:
            if self.scan_beam_num + self.scan_offset < beam_num:
                irange = int(0.5*(beam_num - self.scan_beam_num)) + self.scan_offset
            elif self.scan_beam_num < beam_num:
                irange = int(0.5*(beam_num - self.scan_beam_num))
            self.scan_bound = (irange*self.scan_res)
        else:
            self.scan_bound = 0
            self.scan_beam_num = beam_num
        self.scans = self.scans[:, irange:irange + self.scan_beam_num]
        if self.verbose:
            r_msg = "[" + str(irange) + "-" + str(irange + self.scan_beam_num) + "]"
            print("resized scans --", self.scans.shape, r_msg)

        if not self.clip_scans_at is None:
            np.clip(self.scans, a_min=0, a_max=self.clip_scans_at, out=self.scans)
            self.scans = self.scans / self.clip_scans_at    # normalization makes the vae work

    def initRand(self, rand_scans_num, scan_dim, scan_res, scan_fov, clip_scans_at=5.0):
        self.scan_beam_num = scan_dim
        self.scan_res = scan_res
        self.scan_fov = scan_fov
        self.clip_scans_at = clip_scans_at
        self.scans = np.random.uniform(0, 1.0, size=[rand_scans_num, scan_dim])
        self.cmd_vel = np.zeros((rand_scans_num, 6))
        self.ts = np.zeros((rand_scans_num, 1))

    def computeTransform(self, cmdv, ts):
        cb, tb = cmdv, np.zeros((cmdv.shape[0],))
        for t in range(1, ts.shape[0]): tb[t] = ts[t] - ts[t - 1]
        tstep = np.mean(tb)
        x, y, th = 0.0, 0.0, 0.0
        for n in range(cmdv.shape[0]):
            rk_th = th + 0.5*cb[n, 5]*tstep  # runge-kutta integration
            x = x + cb[n, 0]*np.cos(rk_th)*tstep
            y = y + cb[n, 0]*np.sin(rk_th)*tstep
            th = th + cb[n, 5]*tstep
        cth, sth = np.cos(th), np.sin(th)
        return np.array(((cth, -sth, x), (sth, cth, y), (0, 0, 1))), x, y, th

    def computeTransforms(self, cmdv, ts):
        hm = np.empty((cmdv.shape[0], 9))
        hp = np.empty((cmdv.shape[0], 3))
        for i in range(cmdv.shape[0]):
            h, x, y, t = self.computeTransform(cmdv[i], ts[i])
            hm[i, :] = h.reshape((9,))
            hp[i, :] = np.array([x, y, t])
        return hm, hp

    def projectScan(self, scan, cmdv, ts):
        hm, _, _, _ = computeTransform(cmdv, ts)

        assert scan.shape[0] == self.scan_beam_num, "Wrong scan size"
        theta = self.scan_res*np.arange(-0.5*self.scan_beam_num, 0.5*self.scan_beam_num)
        pts = np.ones((3, self.scan_beam_num))
        pts[0] = scan*np.cos(theta)
        pts[1] = scan*np.sin(theta)

        pts = np.matmul(hm, pts)

        x2 = pts[0]*pts[0]
        y2 = pts[1]*pts[1]
        return np.sqrt(x2 + y2)

    def projectScans(self, scans, cmdv, ts):
        pscans = np.empty(scans.shape)
        for i in range(scans.shape[0]):
            pscans[i] = self.projectScan(scans[i], cmdv[i], ts[i])
        return pscans

    def originalScansDim(self):
        if self.scans is None: return -1
        return self.scans.shape[1]

    def timesteps(self):
        if self.ts is None: return np.zeros((1, 1))
        return self.ts

    def cmdVel(self):
        if self.cmd_vel is None: return np.zeros((1, 1))
        return self.cmd_vel

    def getScans(self, split_at=0):
        if self.scans is None: return np.zeros((1, 1))
        if split_at == 0: return self.scans

        x_train = self.scans[:int(self.scans.shape[0]*split_at), :]
        x_test = self.scans[int(self.scans.shape[0]*split_at):, :]

        if self.verbose:
            print("scans train:", x_train.shape)
            print("scans test:", x_test.shape)

        return x_train, x_test

    def getScanSegments(self, scan, threshold):
        segments = []
        iseg = 0
        useg = bool(scan[0] > threshold)
        for d in range(scan.shape[0]):
            if useg and scan[d] < threshold:
                segments.append([iseg, d, useg])
                iseg = d
                useg = False
            if not useg and scan[d] > threshold:
                segments.append([iseg, d, useg])
                iseg = d
                useg = True
            if d == scan.shape[0] - 1: segments.append([iseg, d, useg])
        return segments

    def plotScan(self, scan, y=None):
        assert scan.shape[0] == self.scan_beam_num, "Wrong scan size"
        theta = self.scan_res*np.arange(-0.5*self.scan_beam_num, 0.5*self.scan_beam_num)
        theta = theta[::-1]

        x_axis = np.arange(self.scan_beam_num)
        segments = self.getScanSegments(scan, 0.99)
        if self.verbose: print("Segments -- ", np.array(segments).shape, "--", segments)

        plt.figure(figsize=(15, 5))
        plt.subplot(121)
        y_axis = scan
        if not y is None:
            y_axis = y
            plt.plot(x_axis, y_axis, color='lightgray')

        plt.plot(x_axis, scan, color='lightgray')
        for s in segments:
            if s[2]:
                col = '#ff7f0e'
                plt.plot(x_axis[s[0]:s[1]], y_axis[s[0]:s[1]], 'o', markersize=0.5, color=col)
            else:
                col = '#1f77b4'
                plt.plot(x_axis[s[0]:s[1]], scan[s[0]:s[1]], 'o', markersize=0.5, color=col)

        ax = plt.subplot(122, projection='polar')
        ax.set_theta_offset(0.5*np.pi)
        plt.plot(theta, scan, color='lightgray')
        for s in segments:
            if s[2]:
                col = '#ff7f0e'
                plt.plot(theta[s[0]:s[1]], y_axis[s[0]:s[1]], 'o', markersize=0.5, color=col)
            else:
                col = '#1f77b4'
                plt.plot(theta[s[0]:s[1]], scan[s[0]:s[1]], 'o', markersize=0.5, color=col)

    def plotProjection(self, scan, params0=None, params1=None):
        assert scan.shape[0] == self.scan_beam_num, "Wrong scan size"
        theta = self.scan_res*np.arange(-0.5*self.scan_beam_num, 0.5*self.scan_beam_num)
        pts = np.ones((3, self.scan_beam_num))
        pts[0] = scan*np.cos(theta)
        pts[1] = scan*np.sin(theta)

        plt.figure()
        plt.axis('equal')
        plt.plot(pts[1], pts[0], label='ref')

        if not params0 is None:
            x, y, th = params0[0], params0[1], params0[2]
            cth, sth = np.cos(th), np.sin(th)
            hm = np.array(((cth, -sth, x), (sth, cth, y), (0, 0, 1)))
            pts0 = np.matmul(hm, pts)
            plt.plot(pts0[1], pts0[0], label='proj')

        if not params1 is None:
            x, y, th = params1[0], params1[1], params1[2]
            cth, sth = np.cos(th), np.sin(th)
            hm = np.array(((cth, -sth, x), (sth, cth, y), (0, 0, 1)))
            pts1 = np.matmul(hm, pts)
            plt.plot(pts1[1], pts1[0], label='pred')

        plt.legend()

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
        self.ae = None

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        # by default, random_normal has mean=0 and std=1.0
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    def buildModel(self):
        if not self.ae is None: return self.ae
        input_shape = (self.original_dim,)
        depth = 32
        dropout = 0.4

        ## ENCODER
        e_in = Input(shape=input_shape, name='encoder_input')
        if self.convolutional:
            enc = Reshape((self.reshape_rows, int(self.original_dim/self.reshape_rows), 1,))(e_in)
            enc = Conv2D(depth, 5, activation='relu', strides=2, padding='same')(enc)
            enc = Conv2D(depth*4, 5, strides=2, padding='same')(enc)
            enc = LeakyReLU(alpha=0.2)(enc)
            enc = Dropout(dropout)(enc)
            # shape info needed to build decoder model
            self.enc_shape = K.int_shape(enc)
            # Out: 1-dim probability
            enc = Flatten()(enc)
        else: enc = e_in
        enc = Dense(self.intermediate_dim, activation='relu')(enc)

        if self.variational:
            self.z_mean = Dense(self.latent_dim, name='z_mean')(enc)
            self.z_log_var = Dense(self.latent_dim, name='z_log_var')(enc)
            z = Lambda(self.sampling, output_shape=(self.latent_dim,), name='z')([self.z_mean, self.z_log_var])
            self.encoder = Model(e_in, [self.z_mean, self.z_log_var, z], name='encoder')
        else:
            e_out = Dense(self.latent_dim, activation='sigmoid')(enc)
            self.encoder = Model(e_in, e_out, name='encoder')
        if self.verbose: self.encoder.summary()

        ## DECODER
        d_in = Input(shape=(self.latent_dim,), name='z_sampling')
        if self.convolutional:
            dec = Dense(self.enc_shape[1]*self.enc_shape[2]*self.enc_shape[3], activation='relu')(d_in)
            dec = Reshape((self.enc_shape[1], self.enc_shape[2], self.enc_shape[3]))(dec)

            dec = Conv2DTranspose(filters=int(depth/2), kernel_size=5,
                              activation='relu', strides=2, padding='same')(dec)
            dec = Conv2DTranspose(filters=int(depth/4), kernel_size=5,
                              activation='relu', strides=2, padding='same')(dec)
            dec = Dropout(dropout)(dec)
            dec = Dense(int(depth/16), activation='relu')(dec)
            dec = Flatten()(dec)
        else:
            dec = Dense(self.intermediate_dim, activation='relu')(d_in)
        d_out = Dense(self.original_dim, activation='sigmoid')(dec)
        self.decoder = Model(d_in, d_out, name='decoder')
        if self.verbose: self.decoder.summary()

        ## AUTOENCODER
        if self.variational:
            vae_out = self.decoder(self.encoder(e_in)[2])
            self.ae = Model(e_in, vae_out, name='vae_mlp')

            reconstruction_loss = binary_crossentropy(e_in, vae_out)
            reconstruction_loss *= self.original_dim

            kl_loss = 1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var)
            kl_loss = -0.5*K.sum(kl_loss, axis=-1)
            vae_loss = K.mean(reconstruction_loss + kl_loss)

            self.ae.add_loss(vae_loss)
            self.ae.compile(optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        else:
            vae_out = self.decoder(self.encoder(e_in))
            self.ae = Model(e_in, vae_out, name='autoencoder')
            self.ae.compile(optimizer='adadelta',
                            loss='binary_crossentropy', metrics=['accuracy'])
        if self.verbose: self.ae.summary()
        return self.ae

    def fitModel(self, x, x_test=None, epochs=10, verbose=None):
        if not x_test is None: x_test = (x_test, None)
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

class GAN:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model

    def discriminatorThin(self):
        if self.D: return self.D
        self.D = Sequential()
        depth = 16
        dropout = 0.4

        self.D.add(Conv2D(depth, 5, strides=2,
                          input_shape=self.input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        if self.verbose: self.D.summary()
        return self.D

    def discriminator(self):
        if self.D: return self.D
        self.D = Sequential()
        depth = 64
        dropout = 0.4

        self.D.add(Conv2D(depth, 5, strides=2,
                          input_shape=self.input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*2, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        # Out: 1-dim probability
        self.D.add(Flatten())
        self.D.add(Dense(1))
        self.D.add(Activation('sigmoid'))
        if self.verbose: self.D.summary()
        return self.D

    def generatorThin(self):
        if self.G: return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64
        dim = 16

        self.G.add(Dense(dim*depth, input_dim=2*self.latent_input_dim))  # input + noise
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, 1, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        if self.verbose: self.G.summary()
        return self.G

    def generator(self):
        if self.G: return self.G
        self.G = Sequential()
        dropout = 0.4
        depth = 64+64+64+64
        dim = 16

        self.G.add(Dense(dim*depth, input_dim=2*self.latent_input_dim))  # input + noise
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, 1, depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(int(depth/2), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))
        if self.verbose: self.G.summary()
        return self.G

    def discriminator_model(self, model_id="default"):
        if self.DM: return self.DM
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        self.DM = Sequential()
        if model_id == "thin": self.DM.add(self.discriminatorThin())
        else: self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self, model_id="default"):
        if self.AM: return self.AM
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        self.AM = Sequential()
        if model_id == "thin":
            self.AM.add(self.generatorThin())
            self.AM.add(self.discriminatorThin())
        else:
            self.AM.add(self.generator())
            self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.AM

    def buildModel(self, input_shape, latent_input_dim, model_id="default"):
        self.input_shape = input_shape
        self.latent_input_dim = latent_input_dim

        self.DIS = self.discriminator_model(model_id)
        if model_id == "thin": self.GEN = self.generatorThin()
        else: self.GEN = self.generator()
        self.ADV = self.adversarial_model(model_id)

    def fitModel(self, x, x_label, train_steps=10, batch_sz=32, verbose=None):
        if verbose is None: verbose = self.verbose
        ret = []
        for b in range(0, x.shape[0], batch_sz):
            if b + batch_sz > x.shape[0]: continue
            for t in range(train_steps):
                noise = np.random.uniform(-1.0, 1.0, size=[batch_sz, self.latent_input_dim])
                gen_in = np.empty((batch_sz, 2*self.latent_input_dim))
                gen_in[:, ::2] = x[b:b + batch_sz]
                gen_in[:, 1::2] = noise

                real = np.zeros((batch_sz, self.input_shape[0], 1, 1))
                real[:, :, 0, 0] = x_label[b:b + batch_sz]
                fake = self.GEN.predict(gen_in)

                x_train = np.concatenate((real, fake))
                y = np.ones([x_train.shape[0], 1])
                y[batch_sz:, :] = 0

                d_loss = self.DIS.train_on_batch(x_train, y)

                y = np.ones([batch_sz, 1])
                a_loss = self.ADV.train_on_batch(gen_in, y)

                if t == train_steps - 1:
                    ret.append([d_loss[0], d_loss[1], a_loss[0], a_loss[1]])
                    if verbose:
                        log_mesg = "%d/%d: [D loss: %f, acc: %f]" % \
                                   (b + batch_sz, x.shape[0], d_loss[0], d_loss[1])
                        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                        print(log_mesg)
        return np.array(ret).reshape(4,)

    def latentInputDim(self):
        return self.latent_input_dim

    def inputDim(self):
        return self.original_dim

    def generate(self, x):
        noise = np.random.uniform(-1.0, 1.0, size=[x.shape[0], self.latent_input_dim])
        gen_in = np.empty((x.shape[0], 2*self.latent_input_dim))
        gen_in[:, ::2] = x
        gen_in[:, 1::2] = noise
        return self.GEN.predict(gen_in)[:, :, 0, 0]

class RGAN:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.D = None   # discriminator
        self.G = None   # generator
        self.DM = None  # discriminator model
        self.AM = None  # adversarial model

    def discriminator(self):
        if self.D: return self.D
        self.D = Sequential()
        depth = 16
        dropout = 0.4

        self.D.add(Conv2D(depth, 5, strides=2,
                          input_shape=self.input_shape, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*4, 5, strides=2, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Conv2D(depth*8, 5, strides=1, padding='same'))
        self.D.add(LeakyReLU(alpha=0.2))
        self.D.add(Dropout(dropout))

        self.D.add(Flatten())
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
        self.G.add(LSTM(depth, input_shape=(self.input_length_dim, 2*self.latent_input_dim),
                        return_sequences=True,
                        activation='tanh', recurrent_activation='hard_sigmoid'))
        self.G.add(LSTM(depth, return_sequences=True,
                        activation='tanh', recurrent_activation='hard_sigmoid'))
        self.G.add(Dense(dim*depth))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))
        self.G.add(Reshape((dim, 1, self.input_length_dim*depth)))
        self.G.add(Dropout(dropout))

        self.G.add(UpSampling2D(size=(2, 1)))
        self.G.add(Conv2DTranspose(int(depth/4), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(int(depth/8), 5, padding='same'))
        self.G.add(BatchNormalization(momentum=0.9))
        self.G.add(Activation('relu'))

        self.G.add(UpSampling2D(size=(4, 1)))
        self.G.add(Conv2DTranspose(1, 5, padding='same'))
        self.G.add(Activation('sigmoid'))

        if self.verbose: self.G.summary()
        return self.G

    def discriminator_model(self):
        if self.DM: return self.DM
        optimizer = RMSprop(lr=0.00002, decay=6e-8)
        self.DM = Sequential()
        self.DM.add(self.discriminator())
        self.DM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.DM

    def adversarial_model(self):
        if self.AM: return self.AM
        optimizer = RMSprop(lr=0.00001, decay=3e-8)
        self.AM = Sequential()
        self.AM.add(self.generator())
        self.AM.add(self.discriminator())
        self.AM.compile(loss='binary_crossentropy',
                        optimizer=optimizer, metrics=['accuracy'])
        return self.AM

    def buildModel(self, input_shape, latent_input_dim, input_length_dim):
        self.input_shape = input_shape
        self.input_length_dim = input_length_dim
        self.latent_input_dim = latent_input_dim

        self.DIS = self.discriminator_model()
        self.GEN = self.generator()
        self.ADV = self.adversarial_model()

    def fitModel(self, x, x_label, train_steps=10, batch_sz=32, verbose=None):
        if verbose is None: verbose = self.verbose
        ret = []
        for b in range(0, x.shape[0], batch_sz):
            if b + batch_sz > x.shape[0]: continue
            for t in range(train_steps):
                noise = np.random.uniform(-1.0, 1.0,
                                          size=[batch_sz, self.input_length_dim, self.latent_input_dim])
                gen_in = np.empty((batch_sz, self.input_length_dim, 2*self.latent_input_dim))
                gen_in[:, :, ::2] = x[b:b + batch_sz]
                gen_in[:, :, 1::2] = noise

                real = np.zeros((batch_sz, self.input_shape[0], 1, 1))
                real[:, :, 0, 0] = x_label[b:b + batch_sz]
                fake = self.GEN.predict(gen_in)

                x_train = np.concatenate((real, fake))
                y = np.ones([x_train.shape[0], 1])
                y[batch_sz:, :] = 0

                d_loss = self.DIS.train_on_batch(x_train, y)

                y = np.ones([batch_sz, 1])
                a_loss = self.ADV.train_on_batch(gen_in, y)

                if t == train_steps - 1:
                    ret.append([d_loss[0], d_loss[1], a_loss[0], a_loss[1]])
                    if verbose:
                        log_mesg = "%d/%d: [D loss: %f, acc: %f]" % \
                                   (b + batch_sz, x.shape[0], d_loss[0], d_loss[1])
                        log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])
                        print(log_mesg)
        return np.array(ret)

    def latentInputDim(self):
        return self.latent_input_dim

    def inputDim(self):
        return self.original_dim

    def generate(self, x):
        noise = np.random.uniform(-1.0, 1.0, size=[x.shape[0],
                                                   self.input_length_dim, self.latent_input_dim])
        gen_in = np.empty((x.shape[0], self.input_length_dim, 2*self.latent_input_dim))
        gen_in[:, :, ::2] = x
        gen_in[:, :, 1::2] = noise
        return self.GEN.predict(gen_in)[:, :, 0, 0]

class SimpleLSTM:
    def __init__(self, batch_seq_num, input_dim, output_dim, batch_size=32, verbose=False):
        self.verbose = verbose
        self.batch_seq_num = batch_seq_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.net = None
        self.net_model = None

    def lstm(self):
        if self.net: return self.net
        dropout = 0.4
        depth = 64+64

        self.net = Sequential()
        self.net.add(LSTM(depth, input_shape=(self.batch_seq_num, self.input_dim),
                          return_sequences=True, activation='tanh',
                          recurrent_activation='hard_sigmoid'))
        self.net.add(LSTM(int(0.5*depth), return_sequences=True,
                          activation='tanh', recurrent_activation='hard_sigmoid'))

        self.net.add(Dense(depth))
        self.net.add(BatchNormalization(momentum=0.9))
        self.net.add(Activation('relu'))
        self.net.add(Reshape((self.batch_seq_num, 32, int(depth/32))))

        self.net.add(Dense(self.output_dim*4))

        # self.net.add(Conv2D(int(0.25*depth), 5, strides=2, padding='same'))
        # self.net.add(Activation('relu'))
        # self.net.add(Conv2D(int(0.125*depth), 5, strides=2, padding='same'))
        # self.net.add(Activation('relu'))
        # self.net.add(Dropout(dropout))

        self.net.add(Flatten())
        self.net.add(Dense(self.output_dim))
        self.net.add(Activation('sigmoid'))
        self.net.add(Activation('linear'))

        if self.verbose: self.net.summary()
        return self.net

    def buildModel(self):
        if self.net_model: return self.net_model
        optimizer = RMSprop(lr=0.00002, decay=6e-8)
        self.net_model = Sequential()
        self.net_model.add(self.lstm())
        self.net_model.compile(loss='binary_crossentropy',
                               optimizer=optimizer, metrics=['accuracy'])
        return self.net_model

    def fitModel(self, x, y, epochs=10, x_test=None, y_test=None):
        v = 1 if self.verbose else 0
        ret = []
        for e in range(epochs):
            for i in range(0, x.shape[0], self.batch_size):
                met = self.net_model.train_on_batch(
                    x[i:i + self.batch_size], y[i:i + self.batch_size])
                ret.append(met)
        if len(ret) == 0: ret = np.zeros((2,))
        else: ret = np.array(ret)
        return np.mean(ret, axis=0)

    def predict(self, x):
        return self.net_model.predict(x)

if __name__ == "__main__":
    # params
    batch_sz = 8
    gan_batch_sz = 32
    scan_idx = 1000 # 1000
    to_show_idx = 10
    scan_ahead_step = 8

    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    ls = LaserScans(verbose=True)
    ls.load("../../dataset/diag_labrococo.txt",
            scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
            scan_beam_num=512, clip_scans_at=8, scan_offset=8)

    # p_scan_num = 1500
    # # ls.plotScan(ls.getScans()[p_scan_num])
    # p_scans = ls.getScans()[scan_idx:scan_idx + p_scan_num]
    # p_cmds = ls.cmdVel()[scan_idx:scan_idx + p_scan_num]
    # p_ts = ls.timesteps()[scan_idx:scan_idx + p_scan_num]
    # p_scans = np.array([p for p in p_scans[batch_sz + scan_ahead_step::batch_sz]])
    # p_cmds = np.array([p_cmds[ns:ns + scan_ahead_step] \
    #                    for ns in range(batch_sz, p_cmds.shape[0] - scan_ahead_step, batch_sz)])
    # p_ts = np.array([p_ts[ns:ns + scan_ahead_step] \
    #                  for ns in range(batch_sz, p_ts.shape[0] - scan_ahead_step, batch_sz)])
    # # pscan = ls.projectScan(p_scans[0], p_cmds[0], p_ts[0])

    # lstm = SimpleLSTM(batch_sz, 7, 3, batch_size=32, verbose=False)
    # lstm.buildModel()

    # idx = 0
    # lstm_x = np.concatenate((p_cmds, p_ts), axis=2)
    # hms, lstm_y = ls.computeTransforms(p_cmds, p_ts)

    # metrics = lstm.fitModel(lstm_x, lstm_y, epochs=40)
    # print("metrics lstm: [loss acc]", metrics)
    # y = lstm.predict(lstm_x)
    # ls.plotProjection(p_scans[100], lstm_y[100], y[100])

    # metrics = lstm.fitModel(lstm_x, lstm_y, epochs=40)
    # metrics = lstm.fitModel(lstm_x, lstm_y, epochs=40)
    # metrics = lstm.fitModel(lstm_x, lstm_y, epochs=40)
    # print("metrics lstm: [loss acc]", metrics)
    # y = lstm.predict(lstm_x)
    # # ls.plotProjection(p_scans[100], lstm_y[100], y[100])
    # plt.show()

    ae = AutoEncoder(ls.originalScansDim(),
                     variational=True, convolutional=False,
                     batch_size=128, latent_dim=10, verbose=False)
    ae.buildModel()

    x, x_test = ls.getScans(0.9)

    ae.fitModel(x[:1000], x_test=None, epochs=30, verbose=0)
    print('Fitting VAE model done.')

    scan = x[scan_idx:(scan_idx + batch_sz*gan_batch_sz)]
    latent = ae.encode(scan)
    # plt.plot(latent[to_show_idx])
    dscan = ae.decode(latent)
    # ls.plotScan(scan[to_show_idx], dscan[to_show_idx])

    ae.fitModel(x[900:5000], x_test=None, epochs=30, verbose=0)

    scan = x[scan_idx:(scan_idx + batch_sz*gan_batch_sz)]
    latent = ae.encode(scan)
    # plt.plot(latent[to_show_idx])
    dscan = ae.decode(latent)
    ls.plotScan(scan[to_show_idx], dscan[to_show_idx])

    scan_ahead = x[batch_sz + scan_ahead_step]

    gan = RGAN()
    gan.buildModel((ls.originalScansDim(), 1, 1,), 16, batch_sz)

    conc = int(8192/32)
    latent = ae.encode(x[:conc])
    in_latent = np.concatenate((latent, ls.cmdVel()[:conc]), axis=1)
    in_latent = np.reshape(in_latent,
                           (int(in_latent.shape[0]/batch_sz), batch_sz, in_latent.shape[1]))
    in_label = x[(batch_sz + scan_ahead_step):(conc + scan_ahead_step + 1):batch_sz, :]
    gan.fitModel(in_latent, in_label, train_steps=15, batch_sz=gan_batch_sz, verbose=True)

    gscan = gan.generate(np.concatenate((latent, ls.cmdVel()[:conc]), axis=1)[:batch_sz])
    ls.plotScan(scan_ahead)
    ls.plotScan(gscan[0])
    plt.show()

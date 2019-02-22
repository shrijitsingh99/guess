#!/usr/bin/env python
# coding: utf-8

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Dense, Embedding, Activation, Flatten, Reshape
from keras.layers import Conv1D, Conv2D, Conv2DTranspose, UpSampling2D, LSTM, TimeDistributed
from keras.layers import LeakyReLU, Dropout
from keras.layers import BatchNormalization
from keras.layers import Lambda, Input, Dense
from keras.models import Model, Sequential, clone_model
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
            print("-- [LasersScans] timesteps:", self.ts.shape)
            print("-- [LasersScans] cmd_vel:", self.cmd_vel.shape)
            print("-- [LasersScans] scans:", self.scans.shape,
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
            print("-- [LasersScans] resized scans:", self.scans.shape, r_msg)

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

    def reshapeInSequences(self, scans, cmdv, ts, seq_length, seq_step, normalize=None):
        next_scan, pparams, hp = None, None, None
        if cmdv.shape[0] < seq_length + seq_step \
           or cmdv.shape[0] != ts.shape[0]: return next_scan, pparams, hp
        e_iter = ts.shape[0] - seq_length - seq_step
        n_rows = int(e_iter/seq_length) + 1
        prev_ts = np.empty((n_rows, seq_length, 1))
        prev_cmdv = np.empty((n_rows, seq_length, cmdv.shape[1]))
        next_ts = np.empty((n_rows, seq_step, 1))
        next_cmdv = np.empty((n_rows, seq_step, cmdv.shape[1]))
        next_scan = np.empty((n_rows, scans.shape[1]))

        for n in range(0, e_iter, seq_length):
            row = int(n/seq_length)
            prev_ts[row] = ts[n:n + seq_length]
            next_ts[row] = ts[n + seq_length:n + seq_length + seq_step]
            prev_cmdv[row] = cmdv[n:n + seq_length]
            next_cmdv[row] = cmdv[n + seq_length:n + seq_length + seq_step]
            if not scans is None: next_scan[row] = scans[n + seq_length + seq_step]

        # next_ts[-1] = ts[-self.gen_step_ahead:]
        # next_cmdv[-1] = cmd_vel[-self.gen_step_ahead:]
        pparams = np.concatenate((prev_cmdv, prev_ts), axis=2)
        _, hp = self.computeTransforms(next_cmdv, next_ts)
        if not normalize is None:
            translation = hp[:, :2]
            np.clip(translation, a_min=-normalize, a_max=normalize, out=translation)
            # translation normalization [-max_dist, max_dist] -> [0, 1]
            hp[:, :2] = translation/normalize
            # theta normalization [-pi, pi] -> [0, 1]
            hp[:, 2] = hp[:, 2]/(2*np.pi)
        return next_scan, pparams, hp

    def computeTransform(self, cmdv, ts):
        cb, tb = cmdv, np.zeros((cmdv.shape[0],))
        for t in range(1, ts.shape[0]): tb[t] = ts[t] - ts[t - 1]
        tstep = max(0.03, np.mean(tb))
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
        hm, _, _, _ = self.computeTransform(cmdv, ts)
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
            print("-- [LasersScans] scans train:", x_train.shape)
            print("-- [LasersScans] scans test:", x_test.shape)
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
        # if self.verbose: print("Segments -- ", np.array(segments).shape, "--", segments)

        plt.figure(figsize=(7, 3))
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

class TfPredictor:
    def __init__(self, batch_seq_num, input_dim, output_dim,
                 model_id="conv", batch_size=32, verbose=False):
        self.verbose = verbose
        self.batch_seq_num = batch_seq_num
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.net = None
        self.net_model = None
        self.model_id = model_id

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
        self.net.add(Flatten())
        self.net.add(Dense(self.output_dim, use_bias=True))
        self.net.add(Activation('tanh'))

        if self.verbose: self.net.summary()
        return self.net

    def conv(self):
        if self.net: return self.net
        dropout = 0.4
        depth = 64+64

        self.net = Sequential()
        self.net.add(Conv1D(depth, 5, strides=2,
                            input_shape=(self.batch_seq_num, self.input_dim), padding='same'))
        # self.net.add(Dense(depth, input_shape=(self.batch_seq_num, self.input_dim)))
        self.net.add(BatchNormalization(momentum=0.9))
        self.net.add(Dropout(dropout))
        self.net.add(LeakyReLU(alpha=0.2))
        # self.net.add(Dense(int(0.5*depth)))
        self.net.add(Conv1D(depth*2, 5, strides=2, padding='same'))
        self.net.add(LeakyReLU(alpha=0.2))

        self.net.add(Dense(int(0.25*depth)))
        self.net.add(LeakyReLU(alpha=0.2))
        self.net.add(Flatten())
        self.net.add(Dense(4*self.output_dim))
        self.net.add(LeakyReLU(alpha=0.2))
        self.net.add(Dense(self.output_dim, use_bias=True))
        self.net.add(Activation('tanh'))

        if self.verbose: self.net.summary()
        return self.net

    def buildModel(self):
        if self.net_model: return self.net_model
        optimizer = RMSprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.0) #6e-8)
        self.net_model = Sequential()
        if self.model_id == "lstm": self.net_model.add(self.lstm())
        else: self.net_model.add(self.conv())
        self.net_model.compile(
            optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
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

    def predict(self, x, denormalize=None):
        if denormalize is None: self.net_model.predict(x)
        tf = self.net_model.predict(x)
        # denormalize
        tf[:, :2] *= denormalize
        tf[:, 2] *= 2*np.pi
        return tf

if __name__ == "__main__":
    batch_sz = 8
    scan_idx = 1000
    to_show_idx = 100
    scan_ahead_step = 10
    max_vel = 0.45
    max_dist = 0.33*scan_ahead_step*max_vel

    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    ls = LaserScans(verbose=True)
    ls.load("../../dataset/diag_underground.txt",
            scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
            scan_beam_num=512, clip_scans_at=8, scan_offset=8)

    p_scan_num = 1500
    p_scans = ls.getScans()[scan_idx:scan_idx + p_scan_num]
    p_cmds = ls.cmdVel()[scan_idx:scan_idx + p_scan_num]
    p_ts = ls.timesteps()[scan_idx:scan_idx + p_scan_num]

    n_scan, tf_x, tf_y = ls.reshapeInSequences(p_scans, p_cmds, p_ts,
                                               batch_sz, scan_ahead_step, normalize=max_dist)

    tfp = TfPredictor(batch_sz, 7, 3, batch_size=32, verbose=True)
    tfp.buildModel()

    metrics = tfp.fitModel(tf_x, tf_y, epochs=40)
    print("-- step 0: simple tfp: [loss acc]", metrics)
    y = tfp.predict(tf_x, denormalize=max_dist)
    ls.plotProjection(p_scans[to_show_idx], tf_y[to_show_idx], y[to_show_idx])

    metrics = tfp.fitModel(tf_x, tf_y, epochs=40)
    print("-- step 1: simple tfp: [loss acc]", metrics)
    y = tfp.predict(tf_x, denormalize=max_dist)
    ls.plotProjection(p_scans[to_show_idx], tf_y[to_show_idx], y[to_show_idx])

    metrics = tfp.fitModel(tf_x, tf_y, epochs=40)
    print("-- step 2: simple tfp: [loss acc]", metrics)
    y = tfp.predict(tf_x, denormalize=max_dist)
    ls.plotProjection(p_scans[to_show_idx], tf_y[to_show_idx], y[to_show_idx])
    plt.show()

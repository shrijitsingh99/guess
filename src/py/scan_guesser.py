#!/usr/bin/env python
# coding: utf-8

import numpy as np
from threading import Thread, Lock
from utility_guess import LaserScans, AutoEncoder, GAN, RGAN, ElapsedTimer

class ScanGuesser:
    def __init__(self,
                 original_scan_dim, net_model="default", scan_batch_sz=8, clip_scans_at=8, gen_scan_ahead_step=1,
                 gan_batch_sz=32, gan_train_steps=5,
                 ae_batch_sz=128, ae_latent_dim=10, ae_intermediate_dim=128, ae_epochs=20,
                 start_update_thr=False, verbose=False):
        self.start_update_thr = start_update_thr
        self.verbose = verbose
        self.original_scan_dim = original_scan_dim
        self.net_model = net_model
        self.scan_batch_sz = scan_batch_sz
        self.clip_scans_at = clip_scans_at
        self.gen_scan_ahead_step = gen_scan_ahead_step
        self.ae_epochs = ae_epochs
        self.ae_latent_dim = ae_latent_dim
        self.gan_batch_sz = gan_batch_sz
        self.gan_train_steps = gan_train_steps
        self.online_scans = None
        self.online_cmd_vel = None
        self.sim_step = 0

        self.ls = LaserScans(verbose=verbose)
        self.ae = AutoEncoder(self.original_scan_dim, variational=True, convolutional=True,
                               latent_dim=ae_latent_dim, intermediate_dim=ae_intermediate_dim,
                               batch_size=ae_batch_sz, verbose=verbose)
        self.__initModels()

        self.update_mtx = Lock()
        self.updating_model = False
        self.thr_scans = None
        self.thr_cmd_vel = None
        if self.start_update_thr: self.thr = Thread(target=self.__updateThr)

    def __updateThr(self):
        while True:
            scans = None
            cmd_vel = None
            self.update_mtx.acquire()
            if not self.updating_model:
                scans = self.thr_scans
                cmd_vel = self.thr_cmd_vel
                self.updating_model = True
            self.update_mtx.release()

            if not scans is None and not cmd_vel is None:
                timer = ElapsedTimer()
                print("-- Model updated in", end='')
                self.__updateAE(scans)
                self.__updateGan(scans, cmd_vel, False) # False : verbose
                print(" ", timer.elapsed_time())

            self.update_mtx.acquire()
            self.updating_model = False
            self.update_mtx.release()
        print("-- Terminating update thread.")

    def __initModels(self):
        self.ae.buildModel()
        if self.net_model == "lstm":
            self.gan_latent_dim = (6 + self.ae_latent_dim)
            self.gan = RGAN(verbose=self.verbose)
            self.gan.buildModel((self.original_scan_dim, 1, 1,), self.gan_latent_dim, self.scan_batch_sz)
        else:
            self.gan_latent_dim = (6 + self.ae_latent_dim)*scan_batch_sz
            self.gan = GAN(verbose=self.verbose)
            self.gan.buildModel((self.original_scan_dim, 1, 1,), self.gan_latent_dim, model_id=self.net_model)

    def __updateAE(self, scans=None):
        v = 0
        if self.verbose: v = 1
        if scans is None: scans = self.ls.getScans()
        # if scans == np.zeros((1, 1)): return
        self.ae.fitModel(scans, epochs=self.ae_epochs, verbose=v)

    def __updateGan(self, scans=None, cmd_vel=None, verbose=False):
        if scans is None: scans = self.ls.getScans()
        if cmd_vel is None: cmd_vel = self.ls.cmdVel()

        z_latent = self.encodeScan(scans)
        if self.net_model == "lstm":
            x_latent, next_scan = self.__reshapeRGanInput(scans, cmd_vel, z_latent)
        else:
            x_latent, next_scan = self.__reshapeGanInput(scans, cmd_vel, z_latent)
        self.gan.fitModel(x_latent, next_scan,
                          train_steps=self.gan_train_steps,
                          batch_sz=self.gan_batch_sz, verbose=verbose)

    def __reshapeRGanInput(self, scans, cmd_vel, ae_latent):
        x_latent = np.concatenate((ae_latent, cmd_vel), axis=1)
        reshaped = np.reshape(x_latent[:self.scan_batch_sz, :], (1, self.scan_batch_sz, self.gan_latent_dim))
        next_scan = None
        if not scans is None and scans.shape[0] > self.scan_batch_sz + self.gen_scan_ahead_step:
            next_scan = np.reshape(scans[self.scan_batch_sz + self.gen_scan_ahead_step - 1, :], (1, scans.shape[1]))

        reshape_step = self.scan_batch_sz
        next_scan_step = self.gen_scan_ahead_step
        if next_scan is None: next_scan_step = 0
        for i in range(self.scan_batch_sz, ae_latent.shape[0] - next_scan_step, reshape_step):
            res = x_latent[i:i + self.scan_batch_sz, :]
            res = np.reshape(res, (1, self.scan_batch_sz, self.gan_latent_dim))
            reshaped = np.concatenate((reshaped, res))
            if not next_scan is None:
                nscan = np.reshape(scans[i + self.scan_batch_sz + next_scan_step - 1, :], (1, scans.shape[1]))
                next_scan = np.concatenate((next_scan, nscan))
        return reshaped, next_scan

    def __reshapeGanInput(self, scans, cmd_vel, ae_latent):
        x_latent = np.concatenate((ae_latent, cmd_vel), axis=1)
        reshaped = np.reshape(x_latent[:self.scan_batch_sz, :], (1, self.gan_latent_dim))
        next_scan = None
        if not scans is None and scans.shape[0] > self.scan_batch_sz + self.gen_scan_ahead_step:
            next_scan = np.reshape(scans[self.scan_batch_sz + self.gen_scan_ahead_step, :], (1, scans.shape[1]))

        reshape_step = self.scan_batch_sz
        # reshape_step = 1
        for i in range(1, ae_latent.shape[0] - self.scan_batch_sz - self.gen_scan_ahead_step, reshape_step):
            res = x_latent[i:i + self.scan_batch_sz, :]
            res = np.reshape(res, (1, self.gan_latent_dim))
            reshaped = np.concatenate((reshaped, res))
            if not next_scan is None:
                nscan = np.reshape(scans[i + self.scan_batch_sz + self.gen_scan_ahead_step, :], (1, scans.shape[1]))
                next_scan = np.concatenate((next_scan, nscan))
        return reshaped, next_scan

    def setInitDataset(self, raw_scans_file, init_models=False, init_scan_batch_num=None):
        print("- Initialize:")
        if raw_scans_file is None:
            if init_scan_batch_num is None: init_scan_batch_num = 1
            init_scan_num = self.gan_batch_sz*self.scan_batch_sz*init_scan_batch_num + self.gen_scan_ahead_step
            print("-- Init random scans... ", end='')
            self.ls.initRand(init_scan_num, self.original_scan_dim)
        else:
            print("-- Init scans from dataset... ", end='')
            self.ls.load(raw_scans_file,
                         clip_scans_at=self.clip_scans_at, scan_center_range=self.original_scan_dim)
        print("done.")

        if init_models:
            if init_scan_batch_num is None:
                scans = self.ls.getScans()
                cmd_vel = self.ls.cmdVel()
            else:
                scans = self.ls.getScans()[:self.gan_batch_sz*self.scan_batch_sz*init_scan_batch_num + self.gen_scan_ahead_step]
                cmd_vel = self.ls.cmdVel()[:self.gan_batch_sz*self.scan_batch_sz*init_scan_batch_num + self.gen_scan_ahead_step]

            timer = ElapsedTimer()
            print("-- Init AutoEncoder and GAN... ", end='')
            self.__updateAE(scans)
            self.__updateGan(scans, cmd_vel, verbose=(not raw_scans_file is None))
            print("done (" + timer.elapsed_time() + ").")
            if self.start_update_thr:
                print("-- Init update thread... ", end='')
                self.thr.start()
                print("done.\n")

    def addScans(self, scans, cmd_vel):
        if scans.shape[0] < self.scan_batch_sz: return
        if self.online_scans is None:
            self.online_scans = scans
            self.online_cmd_vel = cmd_vel
        else:
            self.online_scans = np.concatenate((self.online_scans, scans))
            self.online_cmd_vel = np.concatenate((self.online_cmd_vel, cmd_vel))

        min_scan_num = self.gan_batch_sz*self.scan_batch_sz + self.gen_scan_ahead_step
        if self.online_scans.shape[0] < min_scan_num: return

        if self.online_scans.shape[0]%(min_scan_num - self.gen_scan_ahead_step) == 0:
            scan_idx = self.online_scans.shape[0] - min_scan_num
            self.update_mtx.acquire()
            if self.updating_model:
                self.update_mtx.release()
            else:
                self.updating_model = False
                self.thr_scans = self.online_scans[scan_idx:]
                self.thr_cmd_vel = self.online_cmd_vel[scan_idx:]
                self.update_mtx.release()

            if not self.start_update_thr:
                self.__updateAE(self.thr_scans)
                self.__updateGan(self.thr_scans, self.thr_cmd_vel, False) # False : verbose
                self.updating_model = False

    def addRawScans(self, raw_scans, cmd_vel):
        if raw_scans.shape[0] < self.scan_batch_sz: return
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans / self.clip_scans_at
        self.addScans(scans, cmd_vel)

    def simStep(self):
        print("self.sim_step --", self.sim_step)
        scans = self.ls.getScans()[self.sim_step:self.sim_step + self.scan_batch_sz*self.gan_batch_sz + self.gen_scan_ahead_step - 1]
        cmd_vel = self.ls.cmdVel()[self.sim_step:self.sim_step + self.scan_batch_sz*self.gan_batch_sz + self.gen_scan_ahead_step - 1]
        self.addScans(scans, cmd_vel)
        self.sim_step = self.sim_step + self.scan_batch_sz

    def encodeScan(self, scan):
        return self.ae.encode(scan)

    def decodeScan(self, scan_latent):
        return self.ae.decode(scan_latent)

    def generateScan(self, scans, cmd_vel):
        if self.verbose: timer = ElapsedTimer()
        if scans.shape[0] < self.scan_batch_sz: return np.zeros((1,))
        z_latent = self.encodeScan(scans)
        if self.net_model == "lstm":
            x_latent, _ = self.__reshapeRGanInput(None, cmd_vel, z_latent)
        else:
            x_latent, _ = self.__reshapeGanInput(None, cmd_vel, z_latent)

        gen = self.gan.generate(x_latent)
        if self.verbose: print("Prediction in", timer.elapsed_time())
        return gen, self.decodeScan(z_latent)

    def generateRawScan(self, raw_scans, cmd_vel):
        if self.verbose: timer = ElapsedTimer()
        if raw_scans.shape[0] < self.scan_batch_sz: return np.zeros((1,))
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans / self.clip_scans_at
        # self.plotScan(scans[0])
        gscan, vscan = self.generateScan(scans, cmd_vel)
        return gscan[0]*self.clip_scans_at, vscan[0]*self.clip_scans_at

    def getScans(self):
        if not self.online_scans is None: return self.online_scans
        else: return self.ls.getScans()

    def cmdVel(self):
        if not self.online_cmd_vel is None: return self.online_cmd_vel
        else: return self.ls.cmdVel()

    def plotScan(self, scan, decoded_scan=None):
        if decoded_scan is None: self.ls.plotScan(scan)
        else: self.ls.plotScan(scan, decoded_scan)

if __name__ == "__main__":
    print("ScanGuesser test-main")
    scan_ahead_step = 5
    scan_seq_batch = 8
    guesser = ScanGuesser(512, # original_scan_dim
                          net_model="lstm",  # default; thin; lstm
                          scan_batch_sz=scan_seq_batch,  # sequence of scans to concatenate to create one input
                          gen_scan_ahead_step=scan_ahead_step,  # numbr of 'scansteps' to look ahaed for generation
                          gan_batch_sz=32, gan_train_steps=10, start_update_thr=False)
    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    guesser.setInitDataset("../../dataset/diag_underground.txt", init_models=True, init_scan_batch_num=1)

    # z_latent = guesser.encodeScan(guesser.getScans())
    # dscans = guesser.decodeScan(z_latent)
    # guesser.plotScan(guesser.getScans()[0], dscans[0])

    scan_idx = 1000
    gscan = guesser.generateScan(guesser.getScans()[scan_idx:scan_idx + scan_seq_batch],
                                 guesser.cmdVel()[scan_idx:scan_idx + scan_seq_batch])

    guesser.plotScan(guesser.getScans()[scan_idx + scan_ahead_step])
    guesser.plotScan(gscan)

for i in range(2):
    guesser.simStep()
    gscan = guesser.generateScan(guesser.getScans()[scan_idx:scan_idx + scan_seq_batch],
                                 guesser.cmdVel()[scan_idx:scan_idx + scan_seq_batch])

    guesser.plotScan(guesser.getScans()[scan_idx + scan_ahead_step])
    guesser.plotScan(gscan)

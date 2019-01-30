#!/usr/bin/env python
# coding: utf-8

import numpy as np
from threading import Thread, Lock
from utility_guess import LaserScans, AutoEncoder, GAN, RGAN, ElapsedTimer

class ScanGuesser:
    def __init__(self,
                 original_scan_dim, net_model="default",
                 scan_batch_sz=8, clip_scans_at=8, gen_scan_ahead_step=1,
                 gan_batch_sz=32, gan_train_steps=5,
                 ae_variational=True, ae_convolutional=False,
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
        self.ae = AutoEncoder(self.original_scan_dim,
                              variational=ae_variational, convolutional=ae_convolutional,
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
            self.gan.buildModel((self.original_scan_dim, 1, 1,),
                                self.gan_latent_dim, self.scan_batch_sz)
        else:
            self.gan_latent_dim = (6 + self.ae_latent_dim)*scan_batch_sz
            self.gan = GAN(verbose=self.verbose)
            self.gan.buildModel((self.original_scan_dim, 1, 1,),
                                self.gan_latent_dim, model_id=self.net_model)

    def __reshapeGanInput(self, scans, cmd_vel, ae_latent):
        x_latent = np.concatenate((ae_latent, cmd_vel), axis=1)
        n_rows = int(ae_latent.shape[0]/self.scan_batch_sz)
        if self.net_model == "lstm":
            x_latent = np.reshape(x_latent[:n_rows*self.scan_batch_sz],
                                  (n_rows, self.scan_batch_sz, self.gan_latent_dim))
        else:
            x_latent = np.reshape(x_latent[:n_rows*self.scan_batch_sz],
                                  (n_rows, self.gan_latent_dim))
        next_scan = None
        if not scans is None and scans.shape[0] > self.scan_batch_sz + self.gen_scan_ahead_step:
            next_scan = scans[self.scan_batch_sz + self.gen_scan_ahead_step::self.scan_batch_sz]
        if not next_scan is None and x_latent.shape[0] != next_scan.shape[0]:
            next_scan = np.concatenate((next_scan, np.tile(scans[-1:],
                                                           (1, x_latent.shape[0] - next_scan.shape[0]))))
        return x_latent, next_scan

    def __updateAE(self, scans=None):
        v = 0
        if self.verbose: v = 1
        if scans is None: scans = self.ls.getScans()
        self.ae.fitModel(scans, epochs=self.ae_epochs, verbose=v)

    def __updateGan(self, scans=None, cmd_vel=None, verbose=False):
        if scans is None: scans = self.ls.getScans()
        if cmd_vel is None: cmd_vel = self.ls.cmdVel()

        latent = self.encodeScan(scans)
        x_latent, next_scan = self.__reshapeGanInput(scans, cmd_vel, latent)

        self.gan.fitModel(x_latent, next_scan,
                          train_steps=self.gan_train_steps,
                          batch_sz=self.gan_batch_sz, verbose=verbose)

    def setInitDataset(self, raw_scans_file, init_models=False, init_scan_batch_num=None):
        print("- Initialize:")
        init_scan_num = self.gan_batch_sz*self.scan_batch_sz*init_scan_batch_num + self.gen_scan_ahead_step
        if raw_scans_file is None:
            if init_scan_batch_num is None: init_scan_batch_num = 1

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
                scans = self.ls.getScans()[:init_scan_num]
                cmd_vel = self.ls.cmdVel()[:init_scan_num]

            timer = ElapsedTimer()
            print("-- Init AutoEncoder and GAN... ", end='')
            self.__updateAE(scans)
            self.__updateGan(scans, cmd_vel)
            print("done (" + timer.elapsed_time() + ").")
            if self.start_update_thr:
                print("-- Init update thread... ", end='')
                self.thr.start()
                print("done.\n")

    def addScans(self, scans, cmd_vel):
        if scans.shape[0] < self.scan_batch_sz: return False
        if self.online_scans is None:
            self.online_scans = scans
            self.online_cmd_vel = cmd_vel
        else:
            self.online_scans = np.concatenate((self.online_scans, scans))
            self.online_cmd_vel = np.concatenate((self.online_cmd_vel, cmd_vel))

        min_scan_num = self.gan_batch_sz*self.scan_batch_sz + self.gen_scan_ahead_step
        if self.online_scans.shape[0] < min_scan_num: return False

        if self.start_update_thr \
           and self.online_scans.shape[0]%(min_scan_num - self.gen_scan_ahead_step) == 0:
            self.update_mtx.acquire()
            if self.updating_model:
                self.update_mtx.release()
            else:
                self.updating_model = False
                self.thr_scans = self.online_scans[-min_scan_num:]
                self.thr_cmd_vel = self.online_cmd_vel[-min_scan_num:]
                self.update_mtx.release()
            return True

        if not self.start_update_thr:
            self.__updateAE(self.online_scans[-min_scan_num:])
            self.__updateGan(self.online_scans[-min_scan_num:], self.online_cmd_vel[-min_scan_num:], verbose=True)
            self.updating_model = False
            return True
        else: return False

    def addRawScans(self, raw_scans, cmd_vel):
        if raw_scans.shape[0] < self.scan_batch_sz: return False
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans/self.clip_scans_at
        return self.addScans(scans, cmd_vel)

    def encodeScan(self, scans):
        return self.ae.encode(scans)

    def decodeScan(self, scans_latent):
        return self.ae.decode(scans_latent)

    def generateScan(self, scans, cmd_vel):
        if self.verbose: timer = ElapsedTimer()
        if scans.shape[0] < self.scan_batch_sz: None

        latent = self.encodeScan(scans)
        x_latent, _ = self.__reshapeGanInput(None, cmd_vel, latent)

        gen = self.gan.generate(x_latent)
        if self.verbose: print("Prediction in", timer.elapsed_time())
        return gen[0], self.decodeScan(latent)

    def generateRawScan(self, raw_scans, cmd_vel):
        if self.verbose: timer = ElapsedTimer()
        if raw_scans.shape[0] < self.scan_batch_sz: return None
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans/self.clip_scans_at
        gscan, vscan = self.generateScan(scans, cmd_vel)
        return gscan*self.clip_scans_at, vscan*self.clip_scans_at

    def getScans(self):
        if not self.online_scans is None and self.online_scans.shape[0] > self.scan_batch_sz:
            return self.online_scans
        else: return self.ls.getScans()

    def cmdVel(self):
        if not self.online_cmd_vel is None and self.online_cmd_vel.shape[0] > self.scan_batch_sz:
            return self.online_cmd_vel
        else: return self.ls.cmdVel()

    def plotScan(self, scan, decoded_scan=None):
        if decoded_scan is None: self.ls.plotScan(scan)
        else: self.ls.plotScan(scan, decoded_scan)

    def simStep(self):
        print("--- SIMULATE step:",
              int(self.sim_step/self.scan_batch_sz*self.gan_batch_sz) + 1,
              "; #sample:", self.sim_step)
        scans_num = self.scan_batch_sz*self.gan_batch_sz + self.gen_scan_ahead_step
        scans = self.ls.getScans()[self.sim_step:self.sim_step + scans_num]
        cmd_vel = self.ls.cmdVel()[self.sim_step:self.sim_step + scans_num]

        self.sim_step = self.sim_step + self.scan_batch_sz*self.gan_batch_sz
        return self.addScans(scans, cmd_vel)

if __name__ == "__main__":
    print("ScanGuesser test-main")
    scan_ahead_step = 5
    scan_seq_batch = 8
    guesser = ScanGuesser(512, # number of scan beams considered
                          net_model="lstm",  # default; thin; lstm
                          scan_batch_sz=scan_seq_batch,  # sequence of scans to concatenate to create one input
                          ae_epochs=30,
                          ae_variational=True, ae_convolutional=False,
                          gen_scan_ahead_step=scan_ahead_step,  # number of 'scansteps' to look ahaed for generation
                          gan_batch_sz=32, gan_train_steps=40, start_update_thr=False)
    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    guesser.setInitDataset("../../dataset/diag_underground.txt",
                           init_models=True, init_scan_batch_num=1)

    scan_idx = 8
    scans = guesser.getScans()[scan_idx:scan_idx + scan_seq_batch]
    cmdvs = guesser.cmdVel()[scan_idx:scan_idx + scan_seq_batch]
    scan_guessed = guesser.getScans()[scan_idx + scan_ahead_step]

    gscan, _ = guesser.generateScan(scans, cmdvs)
    # guesser.plotScan(scan_guessed,
    #                  guesser.decodeScan(guesser.encodeScan(scan_guessed))[0])

    for i in range(40):
        if guesser.simStep():
            if i == -1:
                gscan, _ = guesser.generateScan(scans, cmdvs)
                guesser.plotScan(gscan)

    gscan, _ = guesser.generateScan(scans, cmdvs)
    guesser.plotScan(scan_guessed,
                     guesser.decodeScan(guesser.encodeScan(scan_guessed))[0])
    guesser.plotScan(gscan)
    import matplotlib.pyplot as plt
    plt.show()

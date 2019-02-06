#!/usr/bin/env python
# coding: utf-8

import numpy as np
from threading import Thread, Lock
from utility_guess import LaserScans, ElapsedTimer
from utility_guess import AutoEncoder, GAN, RGAN, SimpleLSTM
import matplotlib.pyplot as plt

class ScanGuesser:
    def __init__(self,
                 original_scan_dim, net_model="default",
                 scan_batch_sz=8, clip_scans_at=8,
                 scan_res=0.00653590704, scan_fov=(3/2)*np.pi, gen_scan_ahead_step=1,
                 gan_batch_sz=32, gan_train_steps=5,
                 ae_variational=True, ae_convolutional=False,
                 ae_batch_sz=128, ae_latent_dim=10, ae_intermediate_dim=128, ae_epochs=20,
                 start_update_thr=False, verbose=False):
        self.start_update_thr = start_update_thr
        self.verbose = verbose
        self.original_scan_dim = original_scan_dim
        self.scan_resolution = scan_res
        self.scan_fov = scan_fov
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
        self.online_ts = None
        self.sim_step = 0

        self.ls = LaserScans(verbose=verbose)
        self.projector = SimpleLSTM(scan_batch_sz, 7, 3,
                                    batch_size=self.gan_batch_sz, verbose=verbose)
        self.ae = AutoEncoder(self.original_scan_dim,
                              variational=ae_variational, convolutional=ae_convolutional,
                              latent_dim=ae_latent_dim, intermediate_dim=ae_intermediate_dim,
                              batch_size=ae_batch_sz, verbose=verbose)
        self.__initModels()

        self.update_mtx = Lock()
        self.updating_model = False
        self.thr_scans = None
        self.thr_cmd_vel = None
        self.thr_ts = None
        if self.start_update_thr: self.thr = Thread(target=self.__updateThr)

    def __updateThr(self):
        while True:
            scans = None
            cmd_vel = None
            ts = None
            self.update_mtx.acquire()
            scans = self.thr_scans
            cmd_vel = self.thr_cmd_vel
            ts = self.thr_ts
            self.thr_scans = None
            self.thr_cmd_vel = None
            self.thr_ts = None
            self.update_mtx.release()

            if not scans is None and not cmd_vel is None and not ts is None:
                self.update_mtx.acquire()
                self.updating_model = True
                self.update_mtx.release()

                self.__fitModel(scans, cmd_vel, ts, verbose=False)

                self.update_mtx.acquire()
                self.updating_model = False
                self.update_mtx.release()
        print("-- Terminating update thread.")

    def __initModels(self):
        self.ae.buildModel()
        self.projector.buildModel()
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

    def __reshapeGanInput(self, scans, cmd_vel, ts, ae_latent):
        x_latent = np.concatenate((ae_latent, cmd_vel), axis=1)
        n_rows = int(ae_latent.shape[0]/self.scan_batch_sz)
        if self.net_model == "lstm":
            x_latent = np.reshape(x_latent[:n_rows*self.scan_batch_sz],
                                  (n_rows, self.scan_batch_sz, self.gan_latent_dim))
        else:
            x_latent = np.reshape(x_latent[:n_rows*self.scan_batch_sz],
                                  (n_rows, self.gan_latent_dim))

        next_scan, pparams, hp = None, None, None
        if not scans is None and scans.shape[0] > self.scan_batch_sz + self.gen_scan_ahead_step:
            next_scan = scans[self.scan_batch_sz + self.gen_scan_ahead_step::self.scan_batch_sz]
            if x_latent.shape[0] != next_scan.shape[0]:
                tail = np.tile(scans[-1:], (x_latent.shape[0] - next_scan.shape[0], 1))
                next_scan = np.concatenate((next_scan, tail))

        if not cmd_vel is None and not ts is None \
           and ts.shape[0] > self.scan_batch_sz + self.gen_scan_ahead_step \
           and cmd_vel.shape[0] > self.scan_batch_sz + self.gen_scan_ahead_step:
            prev_cmdv = [cmd_vel[ns:ns + self.scan_batch_sz] \
                         for ns in range(0, cmd_vel.shape[0] - self.gen_scan_ahead_step,
                                         self.scan_batch_sz)]
            prev_cmdv = np.array(prev_cmdv)
            prev_ts = [ts[ns:ns + self.scan_batch_sz] \
                       for ns in range(0, ts.shape[0] - self.gen_scan_ahead_step,
                                       self.scan_batch_sz)]
            prev_ts = np.array(prev_ts)
            if len(prev_ts.shape) != 3:
                prev_ts = prev_ts.reshape((prev_ts.shape[0], prev_ts.shape[1], 1))
            pparams = np.concatenate((prev_cmdv, prev_ts), axis=2)

            next_cmdv = [cmd_vel[ns:ns + self.gen_scan_ahead_step] \
                         for ns in range(self.scan_batch_sz, cmd_vel.shape[0], self.scan_batch_sz)]
            next_cmdv = np.array(next_cmdv)
            next_ts = [ts[ns:ns + self.gen_scan_ahead_step] \
                       for ns in range(self.scan_batch_sz, ts.shape[0], self.scan_batch_sz)]
            next_ts = np.array(next_ts)
            _, hp = self.ls.computeTransforms(next_cmdv, next_ts)
        return x_latent, next_scan, pparams, hp

    def __updateAE(self, scans, verbose=None):
        if verbose is None: verbose = self.verbose
        v = 1 if verbose else 0
        return self.ae.fitModel(scans, epochs=self.ae_epochs, verbose=v)

    def __updateGan(self, scans, cmd_vel, ts, verbose=False):
        latent = self.encodeScan(scans)
        x_latent, next_scan, pp, hp = self.__reshapeGanInput(scans, cmd_vel, ts, latent)
        self.projector.fitModel(pp, hp, epochs=20)
        return self.gan.fitModel(x_latent, next_scan,
                                 train_steps=self.gan_train_steps,
                                 batch_sz=self.gan_batch_sz, verbose=verbose)

    def __fitModel(self, scans, cmd_vel, ts, verbose=False):
        timer = ElapsedTimer()
        print("-- Update model... ", end='')
        ae_metrics = self.__updateAE(scans)
        gan_metrics = self.__updateGan(scans, cmd_vel, ts)[-1]
        print("done (\033[1;32m" + timer.elapsed_time() + "\033[0m).")
        print("  -- AE loss:", ae_metrics[0], "- acc:", ae_metrics[1])
        print("  -- GAN d-loss:", gan_metrics[0], "- d-acc:", gan_metrics[1], end='')
        print(" - a-loss:", gan_metrics[2], "- a-acc:", gan_metrics[3])

    def init(self, raw_scans_file, init_models=False, init_scan_batch_num=None, scan_offset=0):
        print("- Initialize:")
        init_scan_num = self.gan_batch_sz*self.scan_batch_sz*init_scan_batch_num + self.gen_scan_ahead_step
        if raw_scans_file is None:
            if init_scan_batch_num is None: init_scan_batch_num = 1

            print("-- Init random scans... ", end='')
            self.ls.initRand(init_scan_num,
                             self.original_scan_dim, self.scan_resolution, self.scan_fov,
                             clip_scans_at=self.clip_scans_at)
        else:
            print("-- Init scans from dataset... ", end='')
            self.ls.load(raw_scans_file, scan_res=self.scan_resolution, scan_fov=self.scan_fov,
                         scan_beam_num=self.original_scan_dim,
                         clip_scans_at=self.clip_scans_at, scan_offset=scan_offset)
        print("done.")

        if init_models:
            if init_scan_batch_num is None:
                scans = self.ls.getScans()
                cmd_vel = self.ls.cmdVel()
                ts = self.ls.timesteps()
            else:
                scans = self.ls.getScans()[:init_scan_num]
                cmd_vel = self.ls.cmdVel()[:init_scan_num]
                ts = self.ls.timesteps()[:init_scan_num]

            self.__fitModel(scans, cmd_vel, ts, verbose=False)
            if self.start_update_thr:
                print("-- Init update thread... ", end='')
                self.thr.start()
                print("done.\n")

    def addScans(self, scans, cmd_vel, ts):
        if scans.shape[0] < self.scan_batch_sz: return False
        if self.online_scans is None:
            self.online_scans = scans
            self.online_cmd_vel = cmd_vel
            self.online_ts = ts
        else:
            self.online_scans = np.concatenate((self.online_scans, scans))
            self.online_cmd_vel = np.concatenate((self.online_cmd_vel, cmd_vel))
            self.online_ts = np.concatenate((self.online_ts, ts))

        min_scan_num = self.gan_batch_sz*self.scan_batch_sz + self.gen_scan_ahead_step
        # print(self.online_scans.shape[0], min_scan_num)
        if self.online_scans.shape[0] < min_scan_num: return False

        if self.start_update_thr \
           and self.online_scans.shape[0]%((min_scan_num - self.gen_scan_ahead_step))/4 == 0:
            self.update_mtx.acquire()
            if not self.updating_model:
                self.thr_scans = self.online_scans[-min_scan_num:]
                self.thr_cmd_vel = self.online_cmd_vel[-min_scan_num:]
                self.thr_ts = self.online_ts[-min_scan_num:]
            self.update_mtx.release()

        if not self.start_update_thr:
            self.__fitModel(self.online_scans[-min_scan_num:],
                            self.online_cmd_vel[-min_scan_num:],
                            self.online_ts[-min_scan_num:], verbose=False)
        return True

    def addRawScans(self, raw_scans, cmd_vel, ts):
        if raw_scans.shape[0] < self.scan_batch_sz: return False
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans/self.clip_scans_at
        return self.addScans(scans, cmd_vel, ts)

    def encodeScan(self, scans):
        return self.ae.encode(scans)

    def decodeScan(self, scans_latent):
        return self.ae.decode(scans_latent)

    def generateScan(self, scans, cmd_vel, ts):
        if self.verbose: timer = ElapsedTimer()
        if scans.shape[0] < self.scan_batch_sz: return None

        latent = self.encodeScan(scans)
        x_latent, _, _, _ = self.__reshapeGanInput(None, cmd_vel, ts, latent)
        gen = self.gan.generate(x_latent)

        ts = ts.reshape((1, ts.shape[0], 1,))
        cmd_vel = cmd_vel.reshape((1, cmd_vel.shape[0], cmd_vel.shape[1],))
        pparams = np.concatenate((cmd_vel, ts), axis=2)
        hp = self.projector.predict(pparams)

        if self.verbose: print("Prediction in", timer.elapsed_time())
        return gen[0], self.decodeScan(latent), hp[0]

    def generateRawScan(self, raw_scans, cmd_vel, ts):
        if raw_scans.shape[0] < self.scan_batch_sz: return None
        scans = raw_scans
        np.clip(scans, a_min=0, a_max=self.clip_scans_at, out=scans)
        scans = scans/self.clip_scans_at
        gscan, vscan, hp = self.generateScan(scans, cmd_vel, ts)
        return gscan*self.clip_scans_at, vscan*self.clip_scans_at, hp

    def getScans(self):
        if not self.online_scans is None and self.online_scans.shape[0] > self.scan_batch_sz:
            return self.online_scans
        else: return self.ls.getScans()

    def cmdVel(self):
        if not self.online_cmd_vel is None and self.online_cmd_vel.shape[0] > self.scan_batch_sz:
            return self.online_cmd_vel
        else: return self.ls.cmdVel()

    def timesteps(self):
        if not self.online_ts is None and self.online_ts.shape[0] > self.scan_batch_sz:
            return self.online_ts
        else: return self.ls.timesteps()

    def plotScan(self, scan, decoded_scan=None):
        if decoded_scan is None: self.ls.plotScan(scan)
        else: self.ls.plotScan(scan, decoded_scan)

    def plotProjection(self, scan, hm_params=None, gen_params=None):
        self.ls.plotProjection(scan, params0=hm_params, params1=gen_params)

    def simStep(self):
        print("--- SIMULATE step:",
              int(self.sim_step/self.scan_batch_sz*self.gan_batch_sz) + 1,
              "; #sample:", self.sim_step)
        scans_num = self.scan_batch_sz*self.gan_batch_sz + self.gen_scan_ahead_step
        scans = self.ls.getScans()[self.sim_step:self.sim_step + scans_num]
        cmd_vel = self.ls.cmdVel()[self.sim_step:self.sim_step + scans_num]
        ts = self.ls.timesteps()[self.sim_step:self.sim_step + scans_num]

        self.sim_step = self.sim_step + self.scan_batch_sz*self.gan_batch_sz
        return self.addScans(scans, cmd_vel, ts)

if __name__ == "__main__":
    print("ScanGuesser test-main")
    scan_ahead_step = 5
    scan_seq_batch = 8
    guesser = ScanGuesser(512, # number of scan beams considered
                          net_model="lstm",  # default; thin; lstm
                          scan_batch_sz=scan_seq_batch,  # sequence of scans as input
                          scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                          gen_scan_ahead_step=scan_ahead_step, # \# of 'scansteps' to look ahead
                          ae_epochs=30,
                          ae_variational=True, ae_convolutional=False,
                          gan_batch_sz=32, gan_train_steps=30, start_update_thr=False)
    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    guesser.init("../../dataset/diag_underground.txt",
                 init_models=True, init_scan_batch_num=1, scan_offset=6)

    scan_idx = 8
    scans = guesser.getScans()[scan_idx:scan_idx + scan_seq_batch]
    cmdvs = guesser.cmdVel()[scan_idx:scan_idx + scan_seq_batch]
    ts = guesser.timesteps()[scan_idx:scan_idx + scan_seq_batch]
    scan_guessed = guesser.getScans()[scan_idx + scan_seq_batch + scan_ahead_step]

    gscan, _, hp = guesser.generateScan(scans, cmdvs, ts)
    guesser.plotProjection(scan_guessed, gen_params=hp)

    for i in range(25):
        if guesser.simStep():
            if i == -1:
                gscan, _, _ = guesser.generateScan(scans, cmdvs, ts)
                guesser.plotScan(gscan)

    gscan, dscan, hp = guesser.generateScan(scans, cmdvs, ts)
    guesser.plotScan(scan_guessed, dscan[0])
    guesser.plotScan(gscan)
    guesser.plotProjection(scan_guessed, gen_params=hp)

    import matplotlib.pyplot as plt
    plt.show()

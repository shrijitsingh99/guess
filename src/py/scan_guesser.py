#!/usr/bin/env python
# coding: utf-8

import datetime
import os
import sys
import numpy as np
from threading import Thread, Lock
from utils import LaserScans, MetricsSaver, ElapsedTimer, TfPredictor
from autoencoder_lib import AutoEncoder
from gan_lib import GAN
import matplotlib.pyplot as plt

class ScanGuesser:
    def __init__(self, net_model="conv", scan_dim, clip_scans_at=8.0,
                 scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                 # guess configs
                 correlated_steps=8, generation_step=1, buffer_max_size=20,
                 fit_ae=True, fit_projector=True, fit_gan=True, minibuffer_batches_num=16,
                 # projector configs
                 projector_lr=0.002, projector_max_dist=1.0,
                 # ae configs
                 ae_variational=True, ae_convolutional=False,
                 ae_batch_sz=64, ae_lr=0.01, ae_latent_dim=10, ae_intermediate_dim=128, ae_epochs=20,
                 # gan configs
                 gan_batch_sz=32, gan_train_steps=5, gan_noise_dim=8, gan_noise_magnitude = 1.,
                 gan_smoothing_label_factor = 0.1, gan_discriminator_lr=1e-3, gan_generator_lr=1e-3,
                 start_update_thr=False, verbose=False, run_id=None, metrics_save_interleave=100):
        self.start_update_thr = start_update_thr
        self.verbose = verbose
        self.scan_dim = scan_dim
        self.scan_resolution = scan_res
        self.scan_fov = scan_fov
        self.net_model = net_model
        self.correlated_steps = correlated_steps
        self.clip_scans_at = clip_scans_at
        self.generation_step = generation_step
        self.projector_max_dist = projector_max_dist
        self.fit_ae = fit_ae
        self.fit_projector = fit_projector
        self.fit_gan = fit_gan
        self.interpolate_scans_pts = False
        self.ae_epochs = ae_epochs

        self.gan_batch_sz = gan_batch_sz
        self.gan_train_steps = gan_train_steps

        self.generator_input_shape = (self.correlated_steps, ae_latent_dim + 2)
        self.data_buffer = {}
        self.data_buffer_max_size = buffer_max_size*(self.gan_batch_sz*self.correlated_steps) + self.generation_step
        self.minibuffer_batches_num = minibuffer_batches_num
        self.sim_step = 0
        self.metrics_step = 0
        self.metrics_save_interleave = metrics_save_interleave  # save every steps

        if run_id is None:
            self.ms = None
        else:
            dtn = datetime.datetime.now()
            dt = str(dtn.month) + "-" + str(dtn.day) + "_" + str(dtn.hour) + "-" + str(dtn.minute)
            metrics_save_path = os.path.split(os.path.realpath(__file__))[0]
            metrics_save_path = os.path.join(metrics_save_path, "../../dataset/metrics/" + run_id + "_" + dt)
            if not os.path.exists(metrics_save_path):
                os.makedirs(metrics_save_path)
            print('-- Saving metrics @:{}'.format(metrics_save_path))
            self.ms = MetricsSaver(metrics_save_path)

        self.ls = LaserScans(verbose=verbose)
        self.projector = TfPredictor(correlated_steps=correlated_steps, input_dim=3, output_dim=3,
                                     batch_size=batch_sz, verbose=verbose)
        self.ae = AutoEncoder(self.ls.scans_dim(), variational=ae_variational, convolutional=ae_convolutional,
                              latent_dim=ae_latent_dim, intermediate_dim=ae_intermediate_dim,
                              batch_size=ae_batch_sz, verbose=verbose)
        self.gan = GAN(verbose=self.verbose)

        # building models
        self.projector.build_model(lr=projector_lr)
        self.ae.build_model(lr=ae_lr)
        self.gan.build_model(discriminator_input_shape=(latent_dim,), generator_input_shape=self.generator_input_shape,
                             discriminator_lr=gan_discriminator_lr, generator_lr=gan_generator_lr,
                             smoothing_factor=gan_smoothing_label_factor, noise_dim=gan_noise_dim,
                             noise_magnitude=gan_noise_magnitude, model_id=self.net_model)

        self.update_mtx = Lock()
        self.updating_model = False
        self.thr_data_buffer = {}

        if self.start_update_thr:
            self.thr = Thread(target=self._update_thr)

    def _update_thr(self):
        while True:
            scans, next_scans, cmdv, ts = None, None, None, None

            self.update_mtx.acquire()
            scans = self.thr_data_buffer['scans']
            next_scans = self.thr_data_buffer['next_scans']
            cmdv = self.thr_data_buffer['cmdv']
            ts = self.thr_data_buffer['ts']
            self.thr_data_buffer = {}
            self.update_mtx.release()

            if all(x is not None for x in [scans, cmdv, ts]):
                self.update_mtx.acquire()
                self.updating_model = True
                self.update_mtx.release()
                self._train(scans, cmdv, ts, verbose=False)
                self.update_mtx.acquire()
                self.updating_model = False
                self.update_mtx.release()

        print("-- Terminating update thread.")

    def compute_transforms(self, cmdv):
        return np.array(list(self.ls.compute_transforms(cmdv)[1:]))

    def _reshape_gan_input(self, encodings, cmd_vel, ts):
        latent = np.concatenate([encodings, cmd_vel], axis=-1)
        latent = latent.reshape((-1, self.correlated_steps, self.generator_input_shape[1]))
        correlated_cmdv, target_tf = self.ls.reshape_correlated_scans(cmd_vel, ts,
                                                                      self.correlated_steps, self.generation_step,
                                                                      normalize=self.projector_max_dist, theta_axis=5) #theta_axis=1 todo
        return latent, correlated_cmdv, target_tf

    def _train_autoencoder(self, x, verbose=None):
        verbose = int(self.verbose if verbose is None else verboser)
        # rnd_indices = np.arange(x.shape[0])
        # np.random.shuffle(rnd_indices)
        return self.ae.train(x, epochs=self.ae_epochs, verbose=verbose) if self.fit_ae else np.zeros((2,))

    def _train_gan(self, x, next_x, cmd_vel, ts, verbose=False):
        encodings = self.encode_scan(x)
        latent, correlated_cmdv, target_tf = self._reshape_gan_input(encodings, cmd_vel, ts)
        projector_metrics, gan_metrics = np.zeros((2,)), np.zeros((3,))
        assert False, 'input dimensions'

        if self.fit_projector:
            projector_metrics = self.projector.train(correlated_cmdv, target_tf, epochs=10)

        if self.fit_gan:
            gan_metrics = self.gan.train(latent, next_x, train_steps=self.gan_train_steps,
                                         batch_sz=self.gan_batch_sz, verbose=verbose)

        return np.concatenate([projector_metrics, gan_metrics], axis=-1)

    def _train(self, x, next_x, cmd_vel, ts, verbose=False):
        print("-- Update model... ", end='')
        timer = ElapsedTimer()
        ae_metrics = self._train_autoencoder(x)
        gan_metrics = self._train_gan(x, cmd_vel, ts)
        elapsed_secs = timer.secs()
        print("done (\033[1;32m" + str(elapsed_secs) + "s\033[0m).")
        print("  -- AE loss:", ae_metrics[0], "- acc:", ae_metrics[1])
        print("  -- Proj loss:", gan_metrics[0], "- acc:", gan_metrics[1])
        print("  -- GAN d-loss:", gan_metrics[2], "- d-acc:", gan_metrics[3], end='')
        print(" - a-loss:", gan_metrics[4])

        if self.ms is not None:
            if self.fit_ae:
                self.ms.add("ae_mets", ae_metrics)
            if self.fit_projector or self.fit_gan:
                self.ms.add("gan-tf_mets", gan_metrics)
            self.ms.add("update_time", np.array([elapsed_secs]))

        if (self.metrics_step + 1) % self.metrics_save_interleave == 0:
            if self.ms is not None:
                self.ms.save()
        else:
            [sys.stdout.write("\033[F\033[K") for _ in range(4) if not self.verbose and elapsed_secs < 5.0]
        self.metrics_step += 1

    def init(self, raw_scans_file, init_models=False, init_batch_num=0, scan_offset=0):
        print("- Initialize:")
        print("-- Init %d random scans... " % init_scan_num, end='')
        init_scan_num = self.gan_batch_sz*self.correlated_steps + self.generation_step
        self.ls.init_rand(init_scan_num, self.scan_dim, self.scan_resolution, self.scan_fov, clip_scans_at=self.clip_scans_at)
        scans = self.ls.get_scans()
        cmd_vel = self.ls.get_cmd_vel()[..., ::5]
        ts = self.ls.timesteps()
        print("done.")

        if raw_scans_file is not None:
            print("-- Append scans from dataset... ", end='')
            self.ls.load(raw_scans_file, scan_res=self.scan_resolution, scan_fov=self.scan_fov,
                         scan_beam_num=self.scan_dim, clip_scans_at=self.clip_scans_at,
                         scan_offset=scan_offset)
            print("done.")
            ls_load_slice = slice(None, None if init_batch_num < 0 else init_scan_num, None)
            scans = np.vstack((scans, self.ls.get_scans()[ls_load_slice]))
            cmd_vel = np.vstack((cmd_vel, self.ls.get_cmd_vel()[ls_load_slice]))
            ts = np.vstack((ts, self.ls.timesteps()[ls_load_slice]))

        if init_models:
            self._train(scans, cmd_vel, ts, verbose=False)

        if self.start_update_thr:
            print("-- Init update thread... ", end='')
            self.thr.start()
            print("done.\n")

    def add_scans(self, scans, cmd_vel, ts):
        if scans.shape[0] < self.correlated_steps:
            return False

        if len(self.data_buffer) == 0:
            self.data_buffer['scans'] = []
            self.data_buffer['cmdv'] = []
            self.data_buffer['ts'] = []

        self.data_buffer['scans'].append(scans)
        self.data_buffer['cmdv'].append(cmd_vel)
        self.data_buffer['ts'].append(ts)

        bn = self.minibuffer_batches_num
        min_correlated_data_num = self.gan_batch_sz*self.correlated_steps
        minibuffer_size = bn*self.gan_batch_sz*self.correlated_steps + self.generation_step

        if self.b_scans.shape[0] < minibuffer_size:
            return False

        rnd_idx = np.random.randint(len(self.data_buffer['scans']) - (self.correlated_steps + self.generation_step) + 1,
                                    size=(bn*self.gan_batch_sz - 1)).tolist()
        thr_scans = [self.data_buffer['scans'][i:i + self.correlated_steps] for i in rnd_idx]
        thr_next_scans = [self.data_buffer['scans'][i + self.correlated_steps + self.generation_step] for i in rnd_idx]
        thr_cmdv = [self.data_buffer['cmdv'][i:i + self.correlated_steps] for i in rnd_idx]
        thr_ts = [self.data_buffer['ts'][i:i + self.correlated_steps] for i in rnd_idx]

        if self.start_update_thr:
            self.update_mtx.acquire()
            if not self.updating_model:
                self.thr_data_buffer['scans'] = np.array(thr_scans, dtype=np.float32)
                self.thr_data_buffer['next_scans'] = np.array(thr_next_scans, dtype=np.float32)
                self.thr_data_buffer['cmdv'] = np.array(thr_cmdv, dtype=np.float32)
                self.thr_data_buffer['ts'] = np.array(thr_ts, dtype=np.float32)
            self.update_mtx.release()
        else:
            self._train(thr_scans, thr_cmdv, thr_ts, verbose=False)

        return True

    def add_raw_scans(self, raw_scans, cmd_vel, ts):
        if raw_scans.shape[0] < self.correlated_steps:
            return False

        scans = np.clip(raw_scans, a_min=0, a_max=self.clip_scans_at)/self.clip_scans_at
        return self.add_scans(scans, cmd_vel, ts)

    def encode_scan(self, scans):
        return self.ae.encode(scans)

    def decode_scan(self, scans_latent, clip_max=True, interpolate=False):
        decoded = self.ae.decode(scans_latent)

        if interpolate:
            decoded = np.vstack([self.ls.interpolateScanPoints(d) for d in decoded])
        if clip_max:
            decoded[decoded > 0.9] = 0.0

        return decoded

    def generate_scan(self, scans, cmd_vel, ts, clip_max=True):
        if self.verbose: timer = ElapsedTimer()
        if scans.shape[0] < self.correlated_steps:
            return None

        ae_encoded = self.encode_scan(scans)
        latent = np.concatenate((ae_encoded, cmd_vel), axis=1)
        n_rows = int(ae_encoded.shape[0]/self.correlated_steps)
        x_latent = latent[:ae_encoded.shape[0]].reshape((n_rows, self.correlated_steps, self.generator_input_shape[1]))
        import pdb; pdb.set_trace()

        gen = self.decode_scan(self.gan.generate(x_latent), clip_max=False, interpolate=False)
        gen = 0.5*(gen + 1.0)  # tanh denormalize

        if self.interpolate_scans_pts:
            gen = self.ls.interpolateScanPoints(gen)
        if clip_max:
            gen[gen > 0.9] = 0.0

        ts = ts.reshape((1, ts.shape[0], 1,))
        cmd_vel = cmd_vel.reshape((1, cmd_vel.shape[0], cmd_vel.shape[1],))
        pparams = np.concatenate((cmd_vel, ts), axis=2)
        hp = self.projector.predict(pparams, denormalize=self.projector_max_dist)[0]

        vscan = self.decode_scan(ae_encoded, clip_max=clip_max, interpolate=self.interpolate_scans_pts)

        if self.verbose:
            print("-- Prediction in", timer.secs())
        return gen, vscan, hp

    def generate_raw_scan(self, raw_scans, cmd_vel, ts):
        if raw_scans.shape[0] < self.correlated_steps:
            return None

        scans = np.clip(raw_scans, a_min=0, a_max=self.clip_scans_at)/self.clip_scans_at
        gscan, vscan, hp = self.generate_scan(scans, cmd_vel, ts)
        return gscan*self.clip_scans_at, vscan*self.clip_scans_at, hp

    def get_scans(self):
        return self.b_scans if self.b_scans is not None and self.b_scans.shape[0] > self.correlated_steps else self.ls.get_scans()

    def get_cmd_vel(self):
        return self.b_cmdv if self.b_cmdv is not None and self.b_cmdv.shape[0] > self.correlated_steps else self.ls.get_cmd_vel()

    def timesteps(self):
        return self.b_ts if self.b_ts is not None and self.b_ts.shape[0] > self.correlated_steps else self.ls.timesteps()

    def plot_scans(self, scan, decoded_scan=None, save_fig=""):
        self.ls.plot_scans(scan, y=decoded_scan, fig_path=save_fig)

    def plot_projections(self, scan, hm_params=None, gen_params=None, save_fig=""):
        self.ls.plot_projections(scan, params0=hm_params, params1=gen_params, fig_path=save_fig)

    def save_fig_prediction(self, file_path):
        scans_num = self.correlated_steps*self.gan_batch_sz + self.generation_step
        if self.b_scans.shape[0] < scans_num:
            return

        scans = self.ls.get_scans()[-scans_num:]
        cmdv = self.ls.get_cmd_vel()[-scans_num:]
        ts = self.ls.timesteps()[-scans_num:]

        ref_tf = self.compute_transforms(cmdv[-self.generation_step:])
        gscan, dscan, pred_p = self.generate_scan(scans[-(self.correlated_steps + self.generation_step):-self.generation_step],
                                                 cmdv[-(self.correlated_steps + self.generation_step):-self.generation_step],
                                                 ts[-(self.correlated_steps + self.generation_step):-self.generation_step], clip_max=False)

        self.plot_scans(scans[-1], dscan[-1], save_fig=file_path + "target_vae.pdf")
        self.plot_scans(gscan, save_fig=file_path + "gen.pdf")
        self.plot_projections(scans[-1], hm_params=ref_tf, gen_params=pred_p, save_fig=file_path + "tf.pdf")

    def simumlate_step(self):
        scans_num = self.correlated_steps*self.gan_batch_sz + self.generation_step
        scans = self.ls.get_scans()[self.sim_step:self.sim_step + scans_num]
        cmd_vel = self.ls.get_cmd_vel()[self.sim_step:self.sim_step + scans_num]
        ts = self.ls.timesteps()[self.sim_step:self.sim_step + scans_num]
        self.sim_step = self.sim_step + self.correlated_steps*self.gan_batch_sz
        return self.add_scans(scans, cmd_vel, ts)

if __name__ == "__main__":
    print("ScanGuesser test-main")
    scan_seq_size = 8
    scan_generation_step = 5
    guesser = ScanGuesser(512, # number of scan beams considered
                          net_model="afmk",  # conv; lstm
                          scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                          correlated_steps=scan_seq_size,  # sequence of scans as input
                          generation_step=scan_generation_step, # \# of 'scan steps' to look ahead
                          projector_max_dist=scan_generation_step*0.03*0.5,
                          fit_ae=True, fit_projector=True, fit_gan=True,
                          # autoencoder configs
                          ae_epochs=10, ae_variational=True, ae_convolutional=False,
                          ae_latent_dim=10,
                          # gan configs
                          gan_batch_sz=32, gan_train_steps=15, gan_noise_dim=1,
                          start_update_thr=False, # run_id="diag_underground",
                          metrics_save_interleave=20)

    base_path, _ = os.path.split(os.path.realpath(__file__))
    base_path = base_path + "/../../dataset/metrics/pplots/"

    # DIAG_first_floor.txt
    # diag_labrococo.txt
    # diag_underground.txt
    guesser.init("../../dataset/diag_underground.txt", init_models=True, init_batch_num=0, scan_offset=6)

    scan_idx = 1000
    scan_step = scan_idx + scan_seq_size
    scans = guesser.get_scans()[scan_idx:scan_step]
    cmdv = guesser.get_cmd_vel()[scan_idx:scan_step]
    ts = guesser.timesteps()[scan_idx:scan_step]
    scan_guessed = guesser.get_scans()[scan_step + scan_generation_step]
    cmdv_guessed = guesser.get_cmd_vel()[scan_step:scan_step + scan_generation_step]

    gscan, _, _ = guesser.generate_scan(scans, cmdv, ts, clip_max=False)
    guesser.plot_scans(gscan, save_fig=(base_path + "it0_gen.pdf"))

    nsteps = 50
    for i in range(nsteps):
        if guesser.simumlate_step():
            if i % int(0.45*nsteps) == 0:
                gscan, _, _ = guesser.generate_scan(scans, cmdv, ts, clip_max=False)
                guesser.plot_scans(gscan, save_fig=(base_path + "it%d_gen.pdf"%i))

    ref_tf = guesser.compute_transforms(cmdv_guessed)
    print("ref_tf", ref_tf)
    gscan, dscan, hp = guesser.generate_scan(scans, cmdv, ts, clip_max=False)
    print("gen_tf", hp)

    guesser.plot_scans(scan_guessed, dscan[0], save_fig=base_path + "t_ae.pdf")
    guesser.plot_scans(gscan, save_fig=base_path + "gen.pdf")
    guesser.plot_projections(scan_guessed, gen_params=hp, hm_params=ref_tf, save_fig=base_path + "tf.pdf")

    # import matplotlib.pyplot as plt
    # plt.show()

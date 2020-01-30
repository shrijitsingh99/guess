#!/usr/bin/env python
# coding: utf-8

import datetime
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from threading import Thread, Lock

from utils import LaserScans, MetricsSaver, ElapsedTimer, TfPredictor
from autoencoder_lib import AutoEncoder
from gan_lib import GAN


class ScanGuesser:
    def __init__(self, net_model="conv", scan_dim=-1, clip_scans_at=8.0,
                 scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                 # guess configs
                 correlated_steps=8, generation_step=1, buffer_max_size=20,
                 fit_ae=True, fit_projector=True, fit_gan=True, minibuffer_batches_num=16,
                 # projector configs
                 projector_lr=0.002, projector_max_dist=1.0,
                 # ae configs
                 ae_variational=True, ae_convolutional=False,
                 ae_batch_sz=128, ae_lr=0.01, ae_latent_dim=10, ae_intermediate_dim=128, ae_epochs=20,
                 # gan configs
                 gan_batch_sz=32, gan_train_steps=5, gan_noise_dim=8, gan_noise_magnitude = 1.,
                 gan_smoothing_label_factor = 0.1, gan_discriminator_lr=1e-3, gan_generator_lr=1e-3,
                 start_update_thr=False, verbose=False, metrics_save_path_dir='', metrics_save_interleave=10):
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
        self.thr_data_buffer = {}
        self.data_buffer_max_size = buffer_max_size*(self.gan_batch_sz*self.correlated_steps) + self.generation_step
        self.minibuffer_batches_num = minibuffer_batches_num
        self.sim_step = 0
        self.metrics_step = 0
        self.metrics_save_interleave = metrics_save_interleave

        if len(metrics_save_path_dir) == 0:
            self.ms = None
        else:
            print('-- Saving metrics @:{}'.format(metrics_save_path_dir))
            if not os.path.exists(metrics_save_path_dir):
                os.makedirs(metrics_save_path_dir)
            self.ms = MetricsSaver(metrics_save_path_dir)

        self.ls = LaserScans(verbose=verbose)
        self.projector = TfPredictor(correlated_steps=correlated_steps, input_dim=3, output_dim=3,
                                     batch_size=gan_batch_sz, verbose=verbose)
        self.ae = AutoEncoder(scan_dim, variational=ae_variational, convolutional=ae_convolutional,
                              latent_dim=ae_latent_dim, intermediate_dim=ae_intermediate_dim,
                              batch_size=ae_batch_sz, verbose=verbose)
        self.gan = GAN(verbose=self.verbose)

        # building models
        self.projector.build_model(lr=projector_lr)
        self.ae.build_model(lr=ae_lr)
        self.gan.build_model(discriminator_input_shape=(ae_latent_dim,), generator_input_shape=self.generator_input_shape,
                             discriminator_lr=gan_discriminator_lr, generator_lr=gan_generator_lr,
                             smoothing_factor=gan_smoothing_label_factor, noise_dim=gan_noise_dim,
                             noise_magnitude=gan_noise_magnitude, model_id=self.net_model)

        self.update_mtx = Lock()
        self.updating_model = False

        if self.start_update_thr:
            self.thr = Thread(target=self._update_thr)

    def init(self, raw_scans_file='', init_models=False, init_batch_num=0, scan_offset=0):
        print("- Initialize:")
        init_scan_num = self.gan_batch_sz*self.correlated_steps + self.generation_step
        print("-- Init %d random scans... " % init_scan_num, end='')
        self.ls.init_rand(init_scan_num, self.scan_dim, self.scan_resolution, self.scan_fov, clip_scans_at=self.clip_scans_at)
        n_factor = self.correlated_steps + self.generation_step

        scans = self.ls.get_scans()
        next_scans = scans[n_factor::n_factor]
        cmd_vel = self.ls.get_cmd_vel()[..., ::5]
        ts = self.ls.timesteps()
        print("done.")

        if len(raw_scans_file) != 0:
            print("-- Append scans from dataset... ", end='')
            self.ls.load(raw_scans_file, scan_res=self.scan_resolution, scan_fov=self.scan_fov,
                         scan_beam_num=self.scan_dim, clip_scans_at=self.clip_scans_at,
                         scan_offset=scan_offset)
            print("done.")
            ls_load_slice = slice(None, None if init_batch_num < 0 else init_scan_num, None)
            scans = np.vstack((scans, self.ls.get_scans()[ls_load_slice]))
            next_scans = scans[n_factor::n_factor]
            cmd_vel = np.vstack((cmd_vel, self.ls.get_cmd_vel()[ls_load_slice, ::5]))
            ts = np.vstack((ts, self.ls.timesteps()[ls_load_slice]))

        if init_models:
            n_rows = int(scans.shape[0]/n_factor)*n_factor
            scans = scans[:n_rows].reshape((-1, n_factor, self.scan_dim))[:, :self.correlated_steps]
            cmd_vel = cmd_vel[:n_rows].reshape((-1, n_factor, cmd_vel.shape[-1]))
            ts = ts[:n_rows].reshape((-1, n_factor, 1))

            self._train(scans, next_scans, cmd_vel, ts, verbose=False)

        if self.start_update_thr:
            print("-- Init update thread... ", end='')
            self.thr.start()
            print("done.\n")

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
                self._train(scans, next_scans, cmdv, ts, verbose=False)
                self.update_mtx.acquire()
                self.updating_model = False
                self.update_mtx.release()

        print("-- Terminating update thread.")

    def compute_transform(self, cmdv):
        return self.ls.compute_transform(cmdv, theta_axis=1)

    def compute_transforms(self, cmdv):
        return self.ls.compute_transforms(cmdv, theta_axis=1)

    # def _reshape_gan_input(self, encodings, cmd_vel, ts):
    #     latent = np.concatenate([encodings, cmd_vel], axis=-1)
    #     n_rows = int(latent.shape[0]/self.correlated_steps)*self.correlated_steps
    #     latent = latent[:n_rows].reshape((-1, self.correlated_steps, self.generator_input_shape[1]))
    #     correlated_cmdv, target_tf = self.ls.reshape_correlated_scans(cmd_vel, ts,
    #                                                                   correlated_steps=self.correlated_steps,
    #                                                                   integration_steps=self.generation_step,
    #                                                                   theta_axis=1, normalize_factor=self.projector_max_dist)
    #     n_rows = min(latent.shape[0], correlated_cmdv.shape[0])
    #     return latent[:n_rows], correlated_cmdv[:n_rows], target_tf[:n_rows]

    def _train_autoencoder(self, x, verbose=None):
        verbose = int(self.verbose if verbose is None else verboser)
        x = x.reshape((-1, x.shape[-1]))
        rnd_indices = np.arange(x.shape[0])
        np.random.shuffle(rnd_indices)
        return self.ae.train(x[rnd_indices], epochs=self.ae_epochs, verbose=verbose) if self.fit_ae else np.zeros((2,))

    def _train_gan(self, x, next_x, cmd_vel, ts, verbose=False):
        ae_x = x.reshape((-1, x.shape[-1]))
        encodings = self.encode_scan(ae_x).reshape((-1, self.correlated_steps,
                                                    self.generator_input_shape[1] - cmd_vel.shape[-1]))
        next_encodings = self.encode_scan(next_x)
        correlated_cmdv = cmd_vel[..., :self.correlated_steps, :]
        correlated_ts = 0.33*np.ones_like(correlated_cmdv[..., :1])
        next_correlated_cmdv = cmd_vel[..., self.correlated_steps:, :]

        latent = np.concatenate([encodings, correlated_cmdv], axis=-1)
        correlated_cmd = np.concatenate([correlated_cmdv, correlated_ts], axis=-1)
        _, target_tf = self.compute_transforms(next_correlated_cmdv)

        if self.projector_max_dist is not None:
            target_tf[:, :2] = np.clip(target_tf[:, :2], a_min=-self.projector_max_dist,
                                       a_max=self.projector_max_dist)/self.projector_max_dist
            target_tf[:, 2] /= np.pi

        rnd_indices = np.arange(latent.shape[0])
        np.random.shuffle(rnd_indices)

        projector_metrics = self.projector.train(correlated_cmd[rnd_indices], target_tf[rnd_indices],
                                                 epochs=10) if self.fit_projector else np.zeros((2,))
        gan_metrics = self.gan.train(latent[rnd_indices], next_encodings[rnd_indices],
                                     train_steps=self.gan_train_steps, batch_sz=self.gan_batch_sz,
                                     verbose=verbose) if self.fit_gan else np.zeros((3,))

        return np.concatenate([projector_metrics, np.mean(gan_metrics, axis=0)], axis=-1)

    def _train(self, x, next_x, cmd_vel, ts, verbose=False):
        print("-- Update model... ", end='')
        timer = ElapsedTimer()
        ae_metrics = self._train_autoencoder(x)
        gan_metrics = self._train_gan(x, next_x, cmd_vel, ts)
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
                self.ms.save()
        else:
            [sys.stdout.write("\033[F\033[K") for _ in range(4) if not self.verbose and elapsed_secs < 5.0]
        self.metrics_step += 1

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
        minibuffer_size = bn*min_correlated_data_num + self.generation_step

        buffer_scans = np.concatenate([s for s in self.data_buffer['scans']], axis=0)
        buffer_cmdv = np.concatenate([s for s in self.data_buffer['cmdv']], axis=0)
        buffer_ts = np.concatenate([s for s in self.data_buffer['ts']], axis=0)

        if buffer_scans.shape[0] < minibuffer_size:
            return False

        rnd_idx = np.random.randint(buffer_scans.shape[0] - (self.correlated_steps + self.generation_step + 1),
                                    size=(bn*self.gan_batch_sz - 1)).tolist()

        scans = np.array([buffer_scans[i:i + self.correlated_steps] for i in rnd_idx], dtype=np.float32)
        next_scans = np.array([buffer_scans[i + self.correlated_steps + self.generation_step] for i in rnd_idx], dtype=np.float32)
        cmdv = np.array([buffer_cmdv[i:i + self.correlated_steps + self.generation_step] for i in rnd_idx], dtype=np.float32)
        ts = np.array([buffer_ts[i:i + self.correlated_steps + self.generation_step] for i in rnd_idx], dtype=np.float32)

        if self.start_update_thr:
            self.update_mtx.acquire()
            if not self.updating_model:
                self.thr_data_buffer['scans'] = scans
                self.thr_data_buffer['next_scans'] = next_scans
                self.thr_data_buffer['cmdv'] = cmdv
                self.thr_data_buffer['ts'] = ts
            self.update_mtx.release()
        else:
            self._train(scans, next_scans, cmdv, ts, verbose=False)

        return True

    def add_raw_scans(self, raw_scans, cmd_vel, ts):
        if raw_scans.shape[0] < self.correlated_steps:
            return False

        scans = np.clip(raw_scans, a_min=0, a_max=self.clip_scans_at)/self.clip_scans_at
        return self.add_scans(scans, cmd_vel, ts)

    def encode_scan(self, scans):
        return self.ae.encode(scans)

    def decode_scan(self, latent, clip_max=True, interpolate=False):
        decoded = self.ae.decode(latent)

        if interpolate:
            decoded = np.vstack([self.ls.interpolateScanPoints(d) for d in decoded])
        if clip_max:
            decoded[decoded > 0.9] = 0.0

        return decoded

    def generate_scan(self, scans, cmd_vel, ts, clip_max=True):
        assert self.correlated_steps <= scans.shape[0], 'Not enough sample to generate scan latent.'

        if self.verbose:
            timer = ElapsedTimer()

        encodings = self.encode_scan(scans)
        decoded_scan = self.decode_scan(encodings, clip_max=clip_max,
                                        interpolate=self.interpolate_scans_pts)

        latent = np.concatenate([encodings, cmd_vel], axis=-1)
        latent = latent.reshape((-1, self.correlated_steps, self.generator_input_shape[1]))
        generated_latent = self.gan.generate(latent)
        generated_scan = self.decode_scan(generated_latent, clip_max=clip_max, interpolate=False)[0]

        correlated_cmdv = np.expand_dims(cmd_vel, axis=0)
        correlated_ts = 0.33*np.ones_like(correlated_cmdv[..., :1])
        correlated_cmd = np.concatenate([correlated_cmdv, correlated_ts], axis=-1)
        generated_tf_params = self.projector.predict(correlated_cmd)[0]

        generated_tf_params[..., :2] *= self.projector_max_dist
        generated_tf_params[..., 2] *= np.pi

        if self.verbose:
            print("-- Prediction in", timer.secs())

        return generated_scan, decoded_scan, generated_tf_params

    def generate_raw_scan(self, raw_scans, cmd_vel, ts):
        assert self.correlated_steps <= raw_scans.shape[0], 'Not enough sample to generate scan latent.'

        scans = np.clip(raw_scans, a_min=0, a_max=self.clip_scans_at)/self.clip_scans_at
        generated_scan, decoded_scan, generated_tf_params = self.generate_scan(scans, cmd_vel, ts)
        return generated_scan*self.clip_scans_at, decoded_scan*self.clip_scans_at, generated_tf_params

    def get_scans(self):
        return self.data_buffer['scans'] if 'scans' in self.data_buffer and \
            self.data_buffer['scans'] > self.correlated_steps else self.ls.get_scans()

    def get_cmd_vel(self):
        return self.data_buffer['cmdv'] if 'cmdv' in self.data_buffer and \
            self.data_buffer['cmdv'] > self.correlated_steps else self.ls.get_cmd_vel()[..., ::5]

    def timesteps(self):
        return self.data_buffer['ts'] if 'ts' in self.data_buffer and \
            self.data_buffer['ts'] > self.correlated_steps else self.ls.timesteps()

    def plot_scans(self, scan_specs, title, save_fig=""):
        self.ls.plot_scans(scan_specs, title=title, fig_path=save_fig)

    def plot_projections(self, scan, params, param_names, save_fig=""):
        self.ls.plot_projections(scan, params=params, param_names=param_names, fig_path=save_fig)

    def save_fig_prediction(self, file_path):
        scans_num = self.correlated_steps*self.gan_batch_sz + self.generation_step
        if self.b_scans.shape[0] < scans_num:
            return

        scans = self.ls.get_scans()[-scans_num:]
        cmdv = self.ls.get_cmd_vel()[-scans_num:, ::5]
        ts = self.ls.timesteps()[-scans_num:]

        target_tf = self.compute_transforms(cmdv[-self.generation_step:])
        gscan, dscan, pred_p = self.generate_scan(scans[-(self.correlated_steps + self.generation_step):-self.generation_step],
                                                 cmdv[-(self.correlated_steps + self.generation_step):-self.generation_step],
                                                 ts[-(self.correlated_steps + self.generation_step):-self.generation_step], clip_max=False)

        self.plot_scans(scans[-1], dscan[-1], save_fig=file_path + "target_vae.pdf")
        self.plot_scans(gscan, save_fig=file_path + "gen.pdf")
        self.plot_projections(scans[-1], hm_params=target_tf, gen_params=pred_p, save_fig=file_path + "tf.pdf")

    def simulate_step(self):
        scans_num = self.correlated_steps
        step_slice = slice(self.sim_step, self.sim_step + scans_num)

        scans = self.ls.get_scans()[step_slice]
        cmd_vel = self.ls.get_cmd_vel()[step_slice, ::5]
        ts = self.ls.timesteps()[step_slice]

        self.sim_step += self.correlated_steps
        return self.add_scans(scans, cmd_vel, ts)

if __name__ == "__main__":
    print("ScanGuesser test-main")
    scan_n = 10000
    test_id='afmk'

    scan_to_predict_idx = 1000
    minibuffer_batches_num = 5
    correlated_steps = 8
    generation_step = 10

    projector_lr = 1e-2

    ae_lr = 1e-2
    ae_batch_sz = 128
    ae_latent_dim = 32

    gan_discriminator_lr = 1e-3
    gan_generator_lr = 1e-3
    gan_batch_sz = 32
    gan_noise_dim = 8
    gan_smoothing_label_factor = 0.1

    save_experiment = False
    cwd = os.path.dirname(os.path.abspath(__file__))
    dtn = datetime.datetime.now()
    dt = str(dtn.month) + "-" + str(dtn.day) + "_" + str(dtn.hour) + "-" + str(dtn.minute)
    save_path_dir = os.path.join(cwd, "../../dataset/metrics/")
    save_path_dir = os.path.join(save_path_dir, test_id + "_" + dt) if save_experiment else ''
    dataset_file = os.path.join(os.path.join(cwd, "../../dataset/"), "diag_underground.txt")

    guesser = ScanGuesser(net_model="conv",  # conv; lstm
                          scan_dim=512, scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                          correlated_steps=correlated_steps, generation_step=generation_step,
                          projector_max_dist=generation_step*0.3*0.5,
                          minibuffer_batches_num=minibuffer_batches_num,
                          fit_ae=True, fit_projector=True, fit_gan=True,
                          # projector configs
                          projector_lr=projector_lr,
                          # autoencoder configs
                          ae_lr=ae_lr, ae_batch_sz=ae_batch_sz, ae_epochs=20, ae_variational=True,
                          ae_convolutional=False, ae_latent_dim=ae_latent_dim,
                          # gan configs
                          gan_batch_sz=gan_batch_sz, gan_train_steps=20, gan_noise_dim=gan_noise_dim,
                          gan_smoothing_label_factor=gan_smoothing_label_factor,
                          # run
                          start_update_thr=False, verbose=False,
                          metrics_save_path_dir=save_path_dir,
                          metrics_save_interleave=20)

    # diag_first_floor.txt ; diag_underground.txt ; diag_labrococo.txt
    guesser.init(dataset_file, init_models=True, init_batch_num=0, scan_offset=6)

    scans = guesser.get_scans()[:scan_n]
    scan_to_predict = scans[scan_to_predict_idx]
    cmdv = guesser.get_cmd_vel()[:scan_n]
    cmdv_to_predict = cmdv[scan_to_predict_idx - generation_step:scan_to_predict_idx]
    ts = guesser.timesteps()[:scan_n]

    prediction_input_slice = slice(scan_to_predict_idx - (generation_step + correlated_steps),
                                   scan_to_predict_idx - generation_step)
    gscan, _, _ = guesser.generate_scan(scans[prediction_input_slice], cmdv[prediction_input_slice],
                                        ts[prediction_input_slice], clip_max=False)

    save_pattern = '' if len(save_path_dir) == 0 else os.path.join(save_path_dir, 'it%d_gen.pdf')

    guesser.plot_scans([(scan_to_predict, '#e41a1c', 'scan'),
                        # (decoded_scans[scan_to_predict_idx], '#ff7f0e', 'decoded'),
                        (gscan, '#1f77b4', 'generated')], title='it 0',
                       save_fig=save_pattern if len(save_pattern) == 0 else save_pattern % 0)

    # fill buffer
    while not guesser.simulate_step():
        continue

    nsteps = 50
    for i in range(nsteps):
        if guesser.simulate_step():
            if i % int(0.45*nsteps) == 0 and False:
                gscan, dscan, _ = guesser.generate_scan(scans[prediction_input_slice], cmdv[prediction_input_slice],
                                                        ts[prediction_input_slice], clip_max=False)

                guesser.plot_scans([(scan_to_predict, '#e41a1c', 'scan'),
                                    (dscan[-1], '#ff7f0e', 'decoded'),
                                    (gscan, '#1f77b4', 'generated')], title='it %d' % i,
                                   save_fig=save_pattern if len(save_pattern) == 0 else save_pattern % i)

    _, target_tf = guesser.compute_transform(cmdv_to_predict)
    gscan, dscan, tfp_params = guesser.generate_scan(scans[prediction_input_slice],
                                                     cmdv[prediction_input_slice], ts[prediction_input_slice], clip_max=False)

    print("target_tf", target_tf)
    print("gen_tf", tfp_params)

    guesser.plot_scans([(scan_to_predict, '#e41a1c', 'scan'),
                        (dscan[-1], '#ff7f0e', 'decoded'),
                        (gscan, '#1f77b4', 'generated')], title='it %d' % nsteps,
                       save_fig=save_pattern if len(save_pattern) == 0 else save_pattern % nsteps)

    save_pattern = '' if len(save_path_dir) == 0 else os.path.join(save_path_dir, 'tf%d.pdf')
    guesser.plot_projections(scan_to_predict, params=[target_tf, tfp_params],
                             param_names=['projected', 'predicted'],
                             save_fig=save_pattern if len(save_pattern) == 0 else save_pattern % i)

    plt.show()

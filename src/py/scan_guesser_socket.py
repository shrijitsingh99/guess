#! /usr/bin/env python
# coding: utf-8

import socket
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import matplotlib.pyplot as plt

from scan_guesser import ScanGuesser

class Provider:
    def __init__(self, data_length, dip="127.0.0.1", dport=9559):
        self.data_length = data_length
        self.dip = dip
        self.dport = int(dport)
        self.socket = None

    def send(self, data):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.connect((self.dip, self.dport))

        assert len(data.shape) == 1, \
            'Incorrect data shape. Should be Nx1 array'
        assert data.shape[0] == self.data_length , \
            'Invalid data length. Should be (self.data_length, 1)'

        srz_data = np.array(data).tostring()
        self.socket.sendall(srz_data)
        self.socket.close()

class Receiver:
    def __init__(self, data_length, dip="127.0.0.1", dport=9559):
        self.data_length = data_length
        self.dip = dip
        self.dport = int(dport)
        self.socket = None

    def getData(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.dip, self.dport))
        self.socket.listen(1)
        c, _ = self.socket.accept()

        data = b''
        while True:
            block = c.recv(self.data_length*2)
            if not block: break
            data += block
        c.close()
        self.socket.close()

        dsrz_data = np.frombuffer(data, dtype=np.int16)
        return np.array(dsrz_data, dtype=np.float32)

if __name__ == "__main__":
    print("| ----------------------------- |")
    print("| -- ScanGuesser test-socket -- |")
    print("| ----------------------------- |\n")
    skt_pkg_scaling = 1000
    scan_seq_size = 8
    scan_generation_step = 60
    scan_length = 512
    clip_scans_at = 5.0
    module_rate = 1.0/30 # [1/freq]
    max_vel = 0.55  # [m/s]
    max_dist_proj = max_vel*scan_generation_step*module_rate # [m]
    # print("-- max distance projector:", max_dist_proj)

    add_scan = 0 # number of pkg to receive to update
    guesser = ScanGuesser(scan_length, # number of scan beams considered
                          net_model="afmk",  # conv; lstm
                          max_dist_projector=max_dist_proj,
                          scan_res=0.00653590704, scan_fov=(3/2)*np.pi,
                          scan_seq_sz=scan_seq_size,  # sequence of scans as input
                          gen_step=scan_generation_step, # \# of 'scan steps' to look ahead
                          clip_scans_at=clip_scans_at,  # max beam length [m]
                          ae_fit=True, proj_fit=True, gan_fit=True,
                          # autoencoder
                          ae_epochs=10, ae_variational=True, ae_convolutional=False,
                          ae_latent_dim=10,
                          # gan
                          gan_batch_sz=32, gan_train_steps=15, gan_noise_dim=1,
                          start_update_thr=True, run_id="simple_corridor",
                          metrics_save_rate=50)
    guesser.init(None, init_models=True)

    # 6D velocity + timestamp in seconds
    receiver = Receiver((7 + scan_length)*scan_seq_size, dport=9559)
    # generated and decoded scans + generated transform parameters
    provider = Provider(scan_length*2 + 3, dport=9558)

    handshake_port = 9550
    print("-- Requesting modules handshake on localhost:" + str(handshake_port))
    sp = Provider(1, dport=handshake_port)
    sp.send(np.array([1]))

    print("\n-- Staring main loop...\n")
    i = 0
    while i < 10:
        i = i + 0
        try: data_batch_srz = receiver.getData()*(1.0/skt_pkg_scaling)
        except Exception as e:
            print("Error", str(e))
            continue

        scan_batch = data_batch_srz[:scan_seq_size*scan_length]
        scan_batch = scan_batch.reshape(scan_seq_size, scan_length)
        cmdv_batch = data_batch_srz[scan_seq_size*scan_length:]
        cmdv_batch = cmdv_batch.reshape(scan_seq_size, 7)
        ts_batch = cmdv_batch[:, -1:]
        cmdv_batch = cmdv_batch[:, :-1]

        if add_scan + 1 == 2 or True:
            guesser.addRawScans(scan_batch, cmdv_batch, ts_batch)
            add_scan = 0
        else: add_scan = add_scan + 1

        try:
            gscan, vscan, hp = guesser.generateRawScan(scan_batch, cmdv_batch, ts_batch)
            if hp[2] < 0.0: hp[2] = min(-0.52359877559, hp[2])
            else: hp[2] = min(0.52359877559, hp[2])

            to_send = np.concatenate((gscan, vscan[-1]))
            to_send = np.concatenate((to_send, hp))
            provider.send((to_send*skt_pkg_scaling).astype(np.int16))
        except Exception as e: print("Message not sent. ", str(e))

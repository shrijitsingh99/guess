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
    scan_ahead_step = 100
    scan_seq_batch = 8
    scan_length = 512
    clip_scans_at = 5.0
    add_scan = 0 # number of pkg to receive to update
    guesser = ScanGuesser(scan_length, # original_scan_dim
                          net_model="lstm",  # default; thin; lstm
                          scan_batch_sz=scan_seq_batch,  # sequence of scans as input
                          gen_step_ahead=scan_ahead_step,  # \# of 'scansteps' to look ahead
                          clip_scans_at=clip_scans_at,  # max beam length [m]
                          scan_res=0.0085915, scan_fov=4.398848,#(3/2)*np.pi,
                          ae_epochs=40,
                          ae_variational=True, ae_convolutional=False,
                          gan_batch_sz=8, gan_train_steps=15, start_update_thr=True)

    guesser.init(None, init_models=True, init_scan_batch_num=1)

    # 6D velocity + timestamp in seconds
    receiver = Receiver((7 + scan_length)*scan_seq_batch, dport=9559)
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

        scan_batch = data_batch_srz[:scan_seq_batch*scan_length]
        scan_batch = scan_batch.reshape(scan_seq_batch, scan_length)
        cmdv_batch = data_batch_srz[scan_seq_batch*scan_length:]
        cmdv_batch = cmdv_batch.reshape(scan_seq_batch, 7)
        ts_batch = cmdv_batch[:, -1]
        cmdv_batch = cmdv_batch[:, :-1]

        if add_scan + 1 == 2 or True:
            guesser.addRawScans(scan_batch, cmdv_batch, ts_batch)
            add_scan = 0
        else: add_scan = add_scan + 1

        try:
            gscan, vscan, hp = guesser.generateRawScan(scan_batch, cmdv_batch, ts_batch)
            to_send = np.concatenate((gscan, vscan[-1]))
            to_send = np.concatenate((to_send, hp))
            provider.send((to_send*skt_pkg_scaling).astype(np.int16))
        except Exception as e: print("Message not sent. ", str(e))

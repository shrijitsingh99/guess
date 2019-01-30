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
            block = c.recv(2*8192)
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
    scan_ahead_step = 10
    scan_seq_batch = 8
    scan_length = 512
    clip_scans_at = 5
    guesser = ScanGuesser(scan_length, # original_scan_dim
                          net_model="lstm",  # default; thin; lstm
                          scan_batch_sz=scan_seq_batch,  # sequence of scans to concatenate to create one input
                          ae_epochs=30,
                          ae_variational=True, ae_convolutional=False,
                          clip_scans_at=clip_scans_at,  # max beam length [m]
                          gen_scan_ahead_step=scan_ahead_step,  # number of 'scansteps' to look ahaed for generation
                          gan_batch_sz=32, gan_train_steps=40, start_update_thr=True)

    guesser.setInitDataset(None, init_models=True, init_scan_batch_num=1)

    receiver = Receiver((6 + scan_length)*scan_seq_batch, dport=9559)
    provider = Provider(scan_length*2, dport=9558)

    print("-- Staring main loop...")
    i = 0
    while i < 1:
        data_batch_srz = receiver.getData()
        data_batch_srz = data_batch_srz*0.01
        scan_batch = data_batch_srz[:scan_seq_batch*scan_length]
        cmdv_batch = data_batch_srz[scan_seq_batch*scan_length:]
        scan_batch = scan_batch.reshape(scan_seq_batch, scan_length)
        cmdv_batch = cmdv_batch.reshape(scan_seq_batch, 6)

        guesser.addRawScans(scan_batch, cmdv_batch)
        gscan, vscan = guesser.generateRawScan(scan_batch, cmdv_batch)
        print("jesus")
        try:
            # guesser.plotScan(gscan)
            to_send = np.concatenate((gscan, vscan))
            provider.send((to_send*100).astype(np.int16))
        except: pass
        i = i + 0
        # plt.show()

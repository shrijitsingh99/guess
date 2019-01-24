
import socket
import numpy as np

def println(name):
        print name

__all__ = ['Provider', 'Receiver']


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

"""Connector classes for exchanging data between block runners."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import os
import tempfile

import numpy as np
from multiprocessing import Array, Event

# Note: this connector is currently slower than ZMQBlockConnector using
# IPC.
class MPBlockConnector(object):
    """Handles directed data exchange between two blocks using the
    multiprocessing module."""

    def __init__(self, send_array, recv_array, send_ev, recv_ev, conf_ev,
            remote_conf_ev):
        self._send_array = send_array
        self._recv_array = recv_array
        self._send_ev = send_ev
        self._recv_ev = recv_ev
        self._conf_ev = conf_ev
        self._remote_conf_ev = remote_conf_ev

    def send(self, data):
        self._remote_conf_ev.wait()
        self._send_array[:] = data
        self._remote_conf_ev.clear()
        self._send_ev.set()

    def recv(self, data, quit_ev):
        # If the quit event is set, do not wait for the data transfer.
        while self._recv_ev.wait(0.01) != True:
            # Necessary for py26- compatiblity.
            if self._recv_ev.is_set():
                break
            if quit_ev.is_set():
                return False
        data[:] = self._recv_array[:]
        self._recv_ev.clear()
        self._conf_ev.set()
        return True

    def init_runner(self, ctx):
        """Called from the block runner of the sender block."""
        pass

    @classmethod
    def make_pair(self, ctype, sizes, ids):
        array1 = Array(ctype, sizes[0])
        array2 = Array(ctype, sizes[1])
        ev1 = Event()
        ev2 = Event()
        ev3 = Event()
        ev4 = Event()
        ev3.set()
        ev4.set()

        return (MPBlockConnector(array1, array2, ev1, ev2, ev3, ev4),
                MPBlockConnector(array2, array1, ev2, ev1, ev4, ev3))


class ZMQBlockConnector(object):
    """Handles directed data exchange between two blocks using 0MQ."""

    def __init__(self, addr, receiver=False):
        """
        :param addr: ZMQ address string
        :param receiver: used to distinguish between bind/connect
        """
        self._addr = addr
        self._receiver = receiver

    def send(self, data):
        self.socket.send(data, copy=False)

    def recv(self, data, quit_ev):
        if quit_ev.is_set():
            return False

        msg = self.socket.recv(copy=False)
        data[:] = np.frombuffer(buffer(msg), dtype=data.dtype)
        return True

    def init_runner(self, ctx):
        """Called from the block runner of the sender block."""
        import zmq
        self.socket = ctx.socket(zmq.PAIR)
        if self._receiver:
            self.socket.connect(self._addr)
        else:
            self.socket.bind(self._addr)

    @classmethod
    def make_pair(self, ctype, sizes, ids):
        addr = 'ipc://%s/sailfish-master-%d_%d-%d' % (tempfile.gettempdir(),
                os.getpid(), ids[0], ids[1])
        return (ZMQBlockConnector(addr, False), ZMQBlockConnector(addr, True))

"""Connector classes for exchanging data between block runners."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

try:
    import blosc
except ImportError:
    pass
import os
import tempfile

import numpy as np
from multiprocessing import Array, Event

# Note: this connector is currently slower than ZMQSubdomainConnector using
# IPC.
class MPSubdomainConnector(object):
    """Handles directed data exchange between two subdomains using the
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

        return (MPSubdomainConnector(array1, array2, ev1, ev2, ev3, ev4),
                MPSubdomainConnector(array2, array1, ev2, ev1, ev4, ev3))


class ZMQSubdomainConnector(object):
    """Handles directed data exchange between two subdomains using 0MQ."""

    def __init__(self, addr, receiver=False):
        """
        :param addr: ZMQ address string
        :param receiver: used to distinguish between bind/connect
        """
        self._addr = addr
        self._receiver = receiver
        self.port = None

        if addr.startswith('ipc://'):
            self.ipc_file = addr.replace('ipc://', '')
        else:
            self.ipc_file = None

    def init_runner(self, ctx):
        """Called from the block runner of the sender block."""
        import zmq
        self.socket = ctx.socket(zmq.PAIR)
        if self._receiver:
            self.socket.connect(self._addr)
        else:
            self.socket.bind(self._addr)

    def send(self, data):
        self.socket.send(data, copy=False)

    def recv(self, data, quit_ev):
        if quit_ev.is_set():
            return False

        msg = self.socket.recv(copy=False)
        data[:] = np.frombuffer(buffer(msg), dtype=data.dtype)
        return True

    def is_ready(self):
        return True

    @classmethod
    def make_ipc_pair(self, ctype, sizes, ids):
        addr = 'ipc://%s/sailfish-master-%d_%d-%d' % (tempfile.gettempdir(),
                os.getpid(), ids[0], ids[1])
        return (ZMQSubdomainConnector(addr, False), ZMQSubdomainConnector(addr, True))


class ZMQRemoteSubdomainConnector(ZMQSubdomainConnector):
    """Handles directed data exchange between two subdomains on two different hosts."""

    def __init__(self, addr, receiver=False):
        """
        :param addr: if receiver == False, addr is tcp://<interface> or
            tcp://\*, otherwise it is tcp://<remote_node_ip_or_name>
        :param receiver: if True, use connect on the socket, otherwise bind
            it to a random port.
        """
        # Import to check if the module is available and fail early if not.
        import netifaces
        ZMQSubdomainConnector.__init__(self, addr, receiver)

    def is_ready(self):
        return self.port != None or not self._receiver

    def init_runner(self, ctx):
        import zmq
        self.socket = ctx.socket(zmq.PAIR)
        if self._receiver:
            self.socket.connect("{0}:{1}".format(self._addr, self.port))
        else:
            self.port = self.socket.bind_to_random_port(self._addr)

    def get_addr(self):
        iface = self._addr.replace('tcp://', '')
        # Local import so that other connectors can work without this module.
        import netifaces
        if iface in netifaces.interfaces():
            addrs = netifaces.ifaddresses(iface)
            return addrs[netifaces.AF_INET][0]['addr']
        else:
            return None

    def set_addr(self, addr):
        if addr is not None:
            self._addr = 'tcp://{0}'.format(addr)


class CompressedZMQRemoteSubdomainConnector(ZMQRemoteSubdomainConnector):
    """Like ZMQRemoteSubdomainConnector, but transfers compressed data."""

    def send(self, data):
        self.socket.send(blosc.pack_array(data), copy=False)

    def recv(self, data, quit_ev):
        if quit_ev.is_set():
            return False

        msg = self.socket.recv(copy=False)
        data[:] = blosc.unpack_array(bytes(msg))
        return True

"""Machine master.

Coordinates simulation of multiple subdomains on a single host."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import atexit
import ctypes
import logging
import operator
import os
import tempfile

import multiprocessing as mp
from multiprocessing import Process, Array, Event, Value

import zmq

from sailfish import subdomain_runner, util, io
from sailfish.connector import ZMQSubdomainConnector, ZMQRemoteSubdomainConnector

def _start_subdomain_runner(subdomain, config, sim, num_subdomains,
        backend_class, gpu_id, output,
        quit_event, master_addr, timing_info_to_master):
    """
    :param num_subdomains: number of subdomains handled by this machine
    """
    config.logger.debug('BlockRunner starting with PID {0}'.format(os.getpid()))
    # Make sure each subdomain has its own temporary directory.  This is
    # particularly important with Numpy 1.3.0, where there is a race
    # condition when saving npz files.
    # With only one subdomain, use the default to take advantage of pycuda
    # caching..
    if num_subdomains > 1:
        tempfile.tempdir = tempfile.mkdtemp()
    # We instantiate the backend class here (instead of in the machine
    # master), so that the backend object is created within the
    # context of the new process.
    backend = backend_class(config, gpu_id)

    if not timing_info_to_master:
        # If there is no controller channel, all processes are running on a
        # single host and we can communicate with the controller using the
        # loopback interface.
        summary_addr = 'tcp://127.0.0.1:{0}'.format(config._zmq_port)
    else:
        summary_addr = master_addr

    runner_cls = subdomain_runner.BlockRunner
    if sim.subdomain_runner is not None:
        runner_cls = sim.subdomain_runner

    runner = runner_cls(sim, subdomain, output, backend, quit_event, summary_addr,
            master_addr)
    runner.run()


class LBMachineMaster(object):
    """Controls execution of a LB simulation on a single physical machine
    (possibly with multiple GPUs and multiple LB subdomains being simulated)."""

    def __init__(self, config, subdomains, lb_class, subdomain_addr_map=None,
            channel=None, iface=None):
        """
        :param config: LBConfig object
        :param subdomains: list of SubdomainSpec objects assigned to this machine
        :param lb_class: simulation class descendant from LBSim
        :param subdomain_addr_map: dict mapping subdomain IDs to IP/DNS addresses
        :param channel: execnet Channel object for communication with the
                controller
        :param iface: network interface on which to listen for connections from
                other subdomains
        """

        self.subdomains = subdomains
        self.config = config
        self.lb_class = lb_class
        self._subdomain_addr_map = subdomain_addr_map
        self.runners = []
        self._subdomain_id_to_runner = {}
        self._pipes = []
        self._vis_process = None
        self._vis_quit_event = None
        self._quit_event = Event()
        self._channel = channel

        atexit.register(lambda event: event.set(), event=self._quit_event)

        if iface is not None:
            self._iface = iface
        else:
            self._iface = '*'
        self.config.logger = logging.getLogger('saifish')
        formatter = logging.Formatter("[%(relativeCreated)6d %(levelname)5s %(processName)s] %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        if config.verbose:
            stream_handler.setLevel(logging.DEBUG)
        elif config.quiet:
            stream_handler.setLevel(logging.WARNING)
        else:
            stream_handler.setLevel(logging.INFO)

        self.config.logger.addHandler(stream_handler)

        if self.config.log:
            handler = logging.FileHandler(self.config.log)
            handler.setFormatter(formatter)
            handler.setLevel(config.loglevel)
            self.config.logger.addHandler(handler)

        self.config.logger.setLevel(logging.DEBUG)

    def _assign_subdomains_to_gpus(self):
        subdomain2gpu = {}

        try:
            gpus = len(self.config.gpus)
            for i, subdomain in enumerate(self.subdomains):
                subdomain2gpu[subdomain.id] = self.config.gpus[i % gpus]
        except TypeError:
            for subdomain in self.subdomains:
                subdomain2gpu[subdomain.id] = 0

        return subdomain2gpu

    def _get_ctypes_float(self):
        if self.config.precision == 'double':
            return ctypes.c_double
        else:
            return ctypes.c_float

    def _init_connectors(self):
        """Creates subdomain connectors for all subdomains connections."""
        # A set to keep track which connections are already created.
        _subdomain_conns = set()

        # TOOD(michalj): Fix this for multi-grid models.
        grid = util.get_grid_from_config(self.config)

        # IDs of the subdomains that are local to this master.
        local_subdomain_ids = set([b.id for b in self.subdomains])
        local_subdomain_map = dict([(b.id, b) for b in self.subdomains])
        ipc_files = []

        for i, subdomain in enumerate(self.subdomains):
            connecting_subdomains = subdomain.connecting_subdomains()
            for face, nbid in connecting_subdomains:
                if (subdomain.id, nbid) in _subdomain_conns:
                    continue

                _subdomain_conns.add((subdomain.id, nbid))
                _subdomain_conns.add((nbid, subdomain.id))

                cpair = subdomain.get_connection(face, nbid)
                size1 = cpair.src.elements
                size2 = cpair.dst.elements
                ctype = self._get_ctypes_float()

                opp_face = subdomain.opposite_face(face)
                if (opp_face, nbid) in connecting_subdomains:
                    size1 *= 2
                    size2 *= 2
                    face_str = '{0} and {1}'.format(face, opp_face)
                else:
                    face_str = str(face)

                self.config.logger.debug("Block connection: {0} <-> {1}: {2}/{3}"
                        "-element buffer (face {4}).".format(
                            subdomain.id, nbid, size1, size2, face_str))

                if nbid in local_subdomain_ids:
                    c1, c2 = ZMQSubdomainConnector.make_ipc_pair(ctype, (size1, size2),
                                                             (subdomain.id, nbid))
                    subdomain.add_connector(nbid, c1)
                    ipc_files.append(c1.ipc_file)
                    local_subdomain_map[nbid].add_connector(subdomain.id, c2)
                else:
                    receiver = subdomain.id > nbid
                    if receiver:
                        addr = "tcp://{0}".format(self._subdomain_addr_map[nbid])
                    else:
                        addr = "tcp://{0}".format(self._iface)
                    c1 = ZMQRemoteSubdomainConnector(addr, receiver=subdomain.id > nbid)
                    subdomain.add_connector(nbid, c1)

        return ipc_files

    def _init_visualization_and_io(self, sim):
        if self.config.output:
            output_cls = io.format_name_to_cls[self.config.output_format]
        else:
            output_cls = io.LBOutput

        if self.config.mode != 'visualization':
            return lambda subdomain: output_cls(self.config, subdomain.id)

        # basic_fields = sim.fields()
        # XXX compute total storage requirements

        for subdomain in self.subdomains:
            size = reduce(operator.mul, subdomain.size)
            vis_lock = mp.Lock()
            vis_buffer = Array(ctypes.c_float, size, lock=vis_lock)
            vis_geo_buffer = Array(ctypes.c_uint8, size, lock=vis_lock)
            subdomain.set_vis_buffers(vis_buffer, vis_geo_buffer)

        vis_lock = mp.Lock()
        vis_config = Value(io.VisConfig, lock=vis_lock)
        vis_config.iteration = -1
        vis_config.field_name = ''
        vis_config.all_blocks = False
        vis_fields = sim.visualization_fields(sim.dim)

        # Start the visualizatione engine.
        vis_class = util.get_visualization_engines().next()

        # Event to signal that the visualization process should be terminated.
        self._vis_quit_event = Event()
        self._vis_process = Process(
                target=lambda: vis_class(
                    self.config, self.subdomains, self._vis_quit_event,
                    self._quit_event, vis_config).run(),
                name='VisEngine')
        self._vis_process.start()

        return lambda subdomain: io.VisualizationWrapper(
                self.config, subdomain, vis_config, output_cls)

    def _finish_visualization(self):
        if self.config.mode != 'visualization':
            return

        self._vis_quit_event.set()
        self._vis_process.join()

    def run(self):
        self.config.logger.info('Machine master starting with PID {0}'.format(os.getpid()))
        self.config.logger.info('Handling subdomains: {0}'.format([b.id for b in
            self.subdomains]))

        sim = self.lb_class(self.config)
        subdomain2gpu = self._assign_subdomains_to_gpus()

        self.config.logger.info('Subdomain -> GPU map: {0}'.format(subdomain2gpu))

        ipc_files = self._init_connectors()
        output_initializer = self._init_visualization_and_io(sim)
        try:
            backend_cls = util.get_backends(self.config.backends.split(',')).next()
        except StopIteration:
            self.config.logger.error('Failed to initialize compute backend.'
                    ' Make sure pycuda/pyopencl is installed.')
            return

        ctx = zmq.Context()
        sockets = []

        # Create subdomain runners for all subdomains.
        for subdomain in self.subdomains:
            output = output_initializer(subdomain)
            master_addr = 'ipc://{0}/sailfish-master-{1}_{2}'.format(
                    tempfile.gettempdir(), os.getpid(), subdomain.id)
            ipc_files.append(master_addr.replace('ipc://', ''))
            sock = ctx.socket(zmq.PAIR)
            sock.bind(master_addr)
            sockets.append(sock)
            p = Process(target=_start_subdomain_runner,
                        name='Block/{0}'.format(subdomain.id),
                        args=(subdomain, self.config, sim, len(self.subdomains),
                              backend_cls, subdomain2gpu[subdomain.id],
                              output, self._quit_event, master_addr,
                              self._channel is not None))
            self.runners.append(p)
            self._subdomain_id_to_runner[subdomain.id] = p

        # Start all subdomain runners.
        for runner in self.runners:
            runner.start()

        ports = {}
        for socket in sockets:
            runner_ports = socket.recv_pyobj()
            ports.update(runner_ports)

        # Only process rmeote port information if we have a channel open
        # back to the controller.
        if self._channel is not None:
            self._channel.send(ports)
            ports = self._channel.receive()
        else:
            # If there is no channel, we're the single master running in this
            # simulation and no port information should be necessary.
            assert not ports

        for socket in sockets:
            socket.send_pyobj(ports)

        if self._channel is not None and self.config.mode == 'benchmark':
            for socket in sockets:
                ti, min_ti, max_ti = socket.recv_pyobj()
                self._channel.send((tuple(ti), tuple(min_ti), tuple(max_ti)))
                socket.send('ack')

        # Wait for all subdomain runners to finish.
        for runner in self.runners:
            runner.join()

        self._finish_visualization()

        for ipcfile in ipc_files:
            os.unlink(ipcfile)

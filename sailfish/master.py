"""Machine master.

Coordinates simulation of multiple subdomains on a single host."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import ctypes
import logging
import operator
import os
import tempfile

import multiprocessing as mp
from multiprocessing import Process, Array, Event, Value

import zmq

from sailfish import block_runner, util, io
from sailfish.connector import ZMQBlockConnector, ZMQRemoteBlockConnector

def _start_block_runner(block, config, sim, backend_class, gpu_id, output,
        quit_event, master_addr, ctrl_channel):
    config.logger.debug('BlockRunner starting with PID {0}'.format(os.getpid()))
    # Make sure each block has its own temporary directory.  This is
    # particularly important with Numpy 1.3.0, where there is a race
    # condition when saving npz files.
    tempfile.tempdir = tempfile.mkdtemp()
    # We instantiate the backend class here (instead of in the machine
    # master), so that the backend object is created within the
    # context of the new process.
    backend = backend_class(config, gpu_id)

    summary_addr = None
    if ctrl_channel is None:
        # If there is no controller channel, all processes are running on a
        # single host and we can communicate with the controller using the
        # loopback interface.
        summary_addr = 'tcp://127.0.0.1:{0}'.format(config._zmq_port)

    runner = block_runner.BlockRunner(sim, block, output, backend, quit_event,
            summary_addr, master_addr, summary_channel=ctrl_channel)
    runner.run()


class LBMachineMaster(object):
    """Controls execution of a LB simulation on a single physical machine
    (possibly with multiple GPUs and multiple LB blocks being simulated)."""

    def __init__(self, config, blocks, lb_class, subdomain_addr_map=None,
            channel=None, iface=None):
        """
        Args:
        :param config: LBConfig object
        :param blocks: list of SubdomainSpec objects assigned to this machine
        :param lb_class: simulation class descendant from LBSim
        :param subdomain_addr_map: dict mapping subdomain IDs to IP/DNS addresses
        :param channel: execnet Channel object for communication with the
                controller
        :param iface: network interface on which to listen for connections from
                other blocks
        """

        self.blocks = blocks
        self.config = config
        self.lb_class = lb_class
        self._subdomain_addr_map = subdomain_addr_map
        self.runners = []
        self._block_id_to_runner = {}
        self._pipes = []
        self._vis_process = None
        self._vis_quit_event = None
        self._quit_event = Event()
        self._channel = channel

        if iface is not None:
            self._iface = iface
        else:
            self._iface = '*'
        self.config.logger = logging.getLogger('saifish')
        formatter = logging.Formatter("[%(relativeCreated)6d %(levelname)5s %(processName)s] %(message)s")
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        self.config.logger.addHandler(handler)

        if self.config.log:
            handler = logging.FileHandler(self.config.log)
            handler.setFormatter(formatter)
            self.config.logger.addHandler(handler)

        if config.verbose:
            self.config.logger.setLevel(logging.DEBUG)
        elif config.quiet:
            self.config.logger.setLevel(logging.WARNING)
        else:
            self.config.logger.setLevel(logging.INFO)

    def _assign_blocks_to_gpus(self):
        block2gpu = {}

        try:
            gpus = len(self.config.gpus)
            for i, block in enumerate(self.blocks):
                block2gpu[block.id] = self.config.gpus[i % gpus]
        except TypeError:
            for block in self.blocks:
                block2gpu[block.id] = 0

        return block2gpu

    def _get_ctypes_float(self):
        if self.config.precision == 'double':
            return ctypes.c_double
        else:
            return ctypes.c_float

    def _init_connectors(self):
        """Creates block connectors for all blocks connections."""
        # A set to keep track which connections are already created.
        _block_conns = set()

        # TOOD(michalj): Fix this for multi-grid models.
        grid = util.get_grid_from_config(self.config)

        # IDs of the blocks that are local to this master.
        local_block_ids = set([b.id for b in self.blocks])
        local_block_map = dict([(b.id, b) for b in self.blocks])

        for i, block in enumerate(self.blocks):
            connecting_blocks = block.connecting_blocks()
            for face, nbid in connecting_blocks:
                if (block.id, nbid) in _block_conns:
                    continue

                _block_conns.add((block.id, nbid))
                _block_conns.add((nbid, block.id))

                cpair = block.get_connection(face, nbid)
                size1 = cpair.src.elements
                size2 = cpair.dst.elements
                ctype = self._get_ctypes_float()

                opp_face = block.opposite_face(face)
                if (opp_face, nbid) in connecting_blocks:
                    size1 *= 2
                    size2 *= 2
                    face_str = '{0} and {1}'.format(face, opp_face)
                else:
                    face_str = str(face)

                self.config.logger.debug("Block connection: {0} <-> {1}: {2}/{3}"
                        "-element buffer (face {4}).".format(
                            block.id, nbid, size1, size2, face_str))

                if nbid in local_block_ids:
                    c1, c2 = ZMQBlockConnector.make_ipc_pair(ctype, (size1, size2),
                                                             (block.id, nbid))
                    block.add_connector(nbid, c1)
                    local_block_map[nbid].add_connector(block.id, c2)
                else:
                    receiver = block.id > nbid
                    if receiver:
                        addr = "tcp://{0}".format(self._subdomain_addr_map[nbid])
                    else:
                        addr = "tcp://{0}".format(self._iface)
                    c1 = ZMQRemoteBlockConnector(addr, receiver=block.id > nbid)
                    block.add_connector(nbid, c1)

    def _init_visualization_and_io(self):
        if self.config.output:
            output_cls = io.format_name_to_cls[self.config.output_format]
        else:
            output_cls = io.LBOutput

        if self.config.mode != 'visualization':
            return lambda block: output_cls(self.config, block.id)

        for block in self.blocks:
            size = reduce(operator.mul, block.size)
            vis_lock = mp.Lock()
            vis_buffer = Array(ctypes.c_float, size, lock=vis_lock)
            vis_geo_buffer = Array(ctypes.c_uint8, size, lock=vis_lock)
            block.set_vis_buffers(vis_buffer, vis_geo_buffer)

        vis_lock = mp.Lock()
        vis_config = Value(io.VisConfig, lock=vis_lock)
        vis_config.iteration = -1
        vis_config.field_name = ''
        vis_config.all_blocks = False

        # Start the visualizatione engine.
        vis_class = util.get_visualization_engines().next()

        # Event to singal that the visualization process should be terminated.
        self._vis_quit_event = Event()
        self._vis_process = Process(
                target=lambda: vis_class(
                    self.config, self.blocks, self._vis_quit_event,
                    self._quit_event, vis_config).run(),
                name='VisEngine')
        self._vis_process.start()

        return lambda block: io.VisualizationWrapper(
                self.config, block, vis_config, output_cls)

    def _finish_visualization(self):
        if self.config.mode != 'visualization':
            return

        self._vis_quit_event.set()
        self._vis_process.join()

    def run(self):
        self.config.logger.info('Machine master starting with PID {0}'.format(os.getpid()))
        self.config.logger.info('Handling blocks: {0}'.format([b.id for b in
            self.blocks]))

        sim = self.lb_class(self.config)
        block2gpu = self._assign_blocks_to_gpus()

        self.config.logger.info('Block -> GPU map: {0}'.format(block2gpu))

        self._init_connectors()
        output_initializer = self._init_visualization_and_io()
        try:
            backend_cls = util.get_backends().next()
        except StopIteration:
            self.config.logger.error('Failed to initialize compute backend.'
                    ' Make sure pycuda/pyopencl is installed.')
            return

        ctx = zmq.Context()
        sockets = []

        # Create block runners for all blocks.
        for block in self.blocks:
            output = output_initializer(block)
            master_addr = 'ipc://{0}/sailfish-master-{1}_{2}'.format(
                    tempfile.gettempdir(), os.getpid(), block.id)
            sock = ctx.socket(zmq.PAIR)
            sock.bind(master_addr)
            sockets.append(sock)
            p = Process(target=_start_block_runner,
                        name='Block/{0}'.format(block.id),
                        args=(block, self.config, sim,
                              backend_cls, block2gpu[block.id],
                              output, self._quit_event, master_addr,
                              self._channel))
            self.runners.append(p)
            self._block_id_to_runner[block.id] = p

        # Start all block runners.
        for runner in self.runners:
            runner.start()

        ports = {}
        for socket in sockets:
            runner_ports = socket.recv_pyobj()
            ports.update(runner_ports)

        # Only process rmeote port information if we have a channel open
        # back to the controller.
        if self._channel:
            self._channel.send(ports)
            ports = self._channel.receive()

            for socket in sockets:
                socket.send_pyobj(ports)
        else:
            # If there is no channel, we're the single master running in this
            # simulation and no port information should be necessary.
            assert not ports

        # Wait for all block runners to finish.
        for runner in self.runners:
            runner.join()

        self._finish_visualization()


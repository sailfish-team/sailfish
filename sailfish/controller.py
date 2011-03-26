import ctypes
import logging
import operator
import platform
import sys
import multiprocessing as mp
from multiprocessing import Process, Array, Event, Value

from sailfish import codegen, config, io, block_runner
from sailfish.geo import LBGeometry2D, LBGeometry3D

def _get_backends():
    for backend in ['cuda', 'opencl']:
        try:
            module = 'sailfish.backend_{}'.format(backend)
            __import__('sailfish', fromlist=['backend_{}'.format(backend)])
            yield sys.modules[module].backend
        except ImportError:
            pass

def _get_visualization_engines():
    for engine in ['2d']:
        try:
            module = 'sailfish.vis_{}'.format(engine)
            __import__('sailfish', fromlist=['vis_{}'.format(engine)])
            yield sys.modules[module].engine
        except ImportError:
            pass

def _start_block_runner(block, config, sim, backend_class, gpu_id, output,
        quit_event):
    # We instantiate the backend class here (instead in the machine
    # master), so that the backend object is created within the
    # context of the new process.
    backend = backend_class(config, gpu_id)
    runner = block_runner.BlockRunner(sim, block, output, backend, quit_event)
    runner.run()


class LBBlockConnector(object):
    """Handles data exchange between two blocks."""

    def __init__(self, send_array, recv_array, send_ev, recv_ev):
        self._send_array = send_array
        self._recv_array = recv_array
        self._send_ev = send_ev
        self._recv_ev = recv_ev

    def send(self, data):
        self._send_ev.set()

    def recv(self, data):
        self._recv_ev.wait()


class LBMachineMaster(object):
    """Controls execution of a LB simulation on a single physical machine
    (possibly with multiple GPUs and multiple LB blocks being simulated)."""

    def __init__(self, config, blocks, lb_class):
        self.blocks = blocks
        self.config = config
        self.lb_class = lb_class
        self.runners = []
        self._block_id_to_runner = {}
        self._pipes = []
        self._vis_process = None
        self._vis_quit_event = None
        self._quit_event = Event()
        self.config.logger = logging.getLogger('saifish')
        formatter = logging.Formatter("[%(relativeCreated)6d %(levelname)5s %(processName)s] %(message)s")
        handler = logging.StreamHandler()
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
        # TODO: actually assign to different GPUs here
        for gpu, block in enumerate(self.blocks):
            block2gpu[block.id] = 0

        return block2gpu


    def _init_connectors(self):
        """Creates block connectors for all blocks connections."""
        # A set to keep track which connections are already created.
        _block_conns = set()

        for i, block in enumerate(self.blocks):
            for axis, nbid in block.connecting_blocks():
                if (block.id, nbid) in _block_conns:
                    continue

                _block_conns.add((block.id, nbid))
                _block_conns.add((nbid, block.id))

                size = block.connection_buf_size(axis, nbid)
                # FIXME: make it work for doubles as well.
                array1 = Array('f', size)
                array2 = Array('f', size)
                ev1 = Event()
                ev2 = Event()

                block.add_connector(nbid,
                        LBBlockConnector(array1, array2, ev1, ev2))
                self.blocks[nbid].add_connector(block.id,
                        LBBlockConnector(array2, array1, ev2, ev1))

    def _init_block_envelope(self, sim):
        """Sets the size of the ghost node envelope for all blocks."""
        envelope_size = sim.nonlocality
        for vec in sim.grid.basis:
            for comp in vec:
                envelope_size = max(sim.nonlocality, abs(comp))

        # Get rid of any Sympy wrapper objects.
        envelope_size = int(envelope_size)

        for block in self.blocks:
            block.set_actual_size(envelope_size)

    def _init_visualization_and_io(self):
        if self.config.output:
            output_cls = io.format_name_to_cls[self.config.output_format]
        else:
            output_cls = io.LBOutput

        if self.config.mode != 'visualization':
            return lambda block: output_cls(self.config)

        # Compute the largest buffer size necessary to transfer
        # data to be visualized.
        max_size = reduce(max,
                (reduce(operator.mul, x.size) for x in self.blocks), 0)
        vis_lock = mp.RLock()
        vis_buffer = Array(ctypes.c_float, max_size, lock=vis_lock)
        vis_geo_buffer = Array(ctypes.c_uint8, max_size, lock=vis_lock)

        vis_config = Value(io.VisConfig, lock=vis_lock)
        vis_config.iteration = -1
        vis_config.field_name = ''

        # Start the visualizatione engine.
        vis_class = _get_visualization_engines().next()

        # Event to singal that the visualization process should be terminated.
        self._vis_quit_event = Event()
        self._vis_process = Process(
                target=lambda: vis_class(
                    self.config, self.blocks, self._vis_quit_event,
                    self._quit_event, vis_buffer,
                    vis_geo_buffer, vis_config).run(),
                name='VisEngine')
        self._vis_process.start()

        return lambda block: io.VisualizationWrapper(
                self.config, block, vis_buffer, vis_geo_buffer, vis_config, output_cls)

    def _finish_visualization(self):
        if self.config.mode != 'visualization':
            return

        self._vis_quit_event.set()
        self._vis_process.join()

    def run(self):
        self.config.logger.info('Machine master starting.')

        sim = self.lb_class(self.config)
        self._init_block_envelope(sim)

        block2gpu = self._assign_blocks_to_gpus()

        self._init_connectors()
        output_initializer = self._init_visualization_and_io()
        backend_cls = _get_backends().next()

        # Create block runners for all blocks.
        for block in self.blocks:
            output = output_initializer(block)
            p = Process(target=_start_block_runner,
                        name='Block/{}'.format(block.id),
                        args=(block, self.config, sim,
                              backend_cls, block2gpu[block.id],
                              output, self._quit_event))
            self.runners.append(p)
            self._block_id_to_runner[block.id] = p

        # Start all block runners.
        for runner in self.runners:
            runner.start()

        # Wait for all block runners to finish.
        for runner in self.runners:
            runner.join()

        self._finish_visualization()

# TODO: eventually, these arguments will be passed asynchronously
# in a different way
def _start_machine_master(config, blocks, lb_class):
    master = LBMachineMaster(config, blocks, lb_class)
    master.run()

class GeometryError(Exception):
    pass

class LBGeometryProcessor(object):
    """Transforms a set of LBBlocks into a another set covering the same
    physical domain, but optimized for execution on the available hardware.
    Initializes logical connections between the blocks based on their
    location."""

    def __init__(self, blocks, dim, geo):
        self.blocks = blocks
        self.dim = dim
        self.geo = geo

    def _annotate(self):
        # Assign IDs to blocks.  The block ID corresponds to its position
        # in the internal blocks list.
        for i, block in enumerate(self.blocks):
            block.id = i

    def _init_lower_coord_map(self):
        # List position corresponds to the principal axis (X, Y, Z).  List
        # items are maps from lower coordinate along the specific axis to
        # a list of block IDs.
        self._coord_map_list = [{}, {}, {}]
        for block in self.blocks:
            for i, coord in enumerate(block.location):
                self._coord_map_list[i].setdefault(coord, []).append(block)

    def _connect_blocks(self, config):
        if len(self.blocks) <= 1:
            return

        connected = [False] * len(self.blocks)

        def try_connect(block1, block2, geo=None):
            if block1.connect(block2, geo):
                connected[block1.id] = True
                connected[block2.id] = True

        for axis in range(self.dim):
            for block in sorted(self.blocks, key=lambda x: x.location[axis]):
                higher_coord = block.location[axis] + block.size[axis]
                if higher_coord not in self._coord_map_list[axis]:
                    continue
                for neighbor_candidate in \
                        self._coord_map_list[axis][higher_coord]:
                    try_connect(block, neighbor_candidate)

        # In case the simulation domain is globally periodic, try to connect
        # the blocks at the lower boundary of the domain along the periodic
        # axis (i.e. coordinate = 0) with blocks which have a boundary at the
        # highest global coordinate (gx, gy, gz).
        if config.periodic_x:
            for block in self._coord_map_list[0][0]:
                # If the block spans the whole X axis of the domain, mark it
                # as locally periodic and do not try to find any neigbor
                # candidates.
                if block.location[0] + block.size[0] == self.geo.gx:
                    block.enable_local_periodicity(0)
                    continue

                for x0, candidates in self._coord_map_list[0].iteritems():
                    for candidate in candidates:
                       if (candidate.location[0] + candidate.size[0]
                               == self.geo.gx):
                            try_connect(block, candidate, self.geo)

        if config.periodic_y:
            for block in self._coord_map_list[1][0]:
                if block.location[1] + block.size[1] == self.geo.gy:
                    block.enable_local_periodicity(1)
                    continue

                for y0, candidates in self._coord_map_list[1].iteritems():
                    for candidate in candidates:
                       if (candidate.location[1] + candidate.size[1]
                               == self.geo.gy):
                            try_connect(block, candidate, self.geo)

        if self.dim > 2 and config.periodic_z:
            for block in self._coord_map_list[2][0]:
                if block.location[2] + block.size[2] == self.geo.gz:
                    block.enable_local_periodicity(2)
                    continue

                for z0, candidates in self._coord_map_list[2].iteritems():
                    for candidate in candidates:
                       if (candidate.location[2] + candidate.size[2]
                               == self.geo.gz):
                            try_connect(block, candidate, self.geo)

        # Ensure every block is connected to at least one other block.
        if not all(connected) and len(connected) > 1:
            raise GeometryError()

    def transform(self, config):
        self._annotate()
        self._init_lower_coord_map()
        self._connect_blocks(config)
        return self.blocks


class LBSimulationController(object):
    """Controls the execution of a LB simulation."""

    def __init__(self, lb_class, lb_geo=None):
        self.conf = config.LBConfig()
        self._lb_class = lb_class

        # Use a default global geometry is one has not been
        # specified explicitly.
        if lb_geo is None:
            if self.dim == 2:
                lb_geo = LBGeometry2D
            else:
                lb_geo = LBGeometry3D

        self._lb_geo = lb_geo

        group = self.conf.add_group('Runtime mode settings')
        group.add_argument('--mode', help='runtime mode', type=str,
            choices=['batch', 'visualization', 'benchmark']),
        group.add_argument('--every',
            help='save/visualize simulation results every N iterations ',
            metavar='N', type=int, default=100)
        group.add_argument('--max_iters',
            help='number of iterations to run; use 0 to run indefinitely',
            type=int, default=0)
        group.add_argument('--output',
            help='save simulation results to FILE', metavar='FILE',
            type=str, default='')
        group.add_argument('--output_format',
            help='output format', type=str,
            choices=io.format_name_to_cls.keys(), default='npy')
        group.add_argument('--backends',
            type=str, default='cuda,opencl',
            help='computational backends to use; multiple backends '
                 'can be separated by a comma')
        group.add_argument('--visualize',
            type=str, default='2d',
            help='visualization engine to use')

        group = self.conf.add_group('Simulation-specific settings')
        lb_class.add_options(group, self.dim)

        group = self.conf.add_group('Geometry settings')
        lb_geo.add_options(group)

        group = self.conf.add_group('Code generator options')
        codegen.BlockCodeGenerator.add_options(group)

        # Backend options
        for backend in _get_backends():
            group = self.conf.add_group(
                    "'{}' backend options".format(backend.name))
            backend.add_options(group)

        for engine in _get_visualization_engines():
            group = self.conf.add_group(
                    "'{}' visualization engine".format(engine.name))
            engine.add_options(group)

        # Set default values defined by the simulation-specific class.
        defaults = {}
        lb_class.update_defaults(defaults)
        self.conf.set_defaults(defaults)

    @property
    def dim(self):
        """Dimensionality of the simulation: 2 or 3."""
        return self._lb_class.geo.dim

    def run(self):
        self.conf.parse()
        self.geo = self._lb_geo(self.conf)

        blocks = self.geo.blocks()
        proc = LBGeometryProcessor(blocks, self.dim, self.geo)
        blocks = proc.transform(self.conf)

        # TODO(michalj): do this over MPI
        p = Process(target=_start_machine_master,
                    name='Master/{}'.format(platform.node()),
                    args=(self.conf, blocks, self._lb_class))
        p.start()
        p.join()

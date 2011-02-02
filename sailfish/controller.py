import os
import sys
from multiprocessing import Process, Pipe, Array, Event

from sailfish import codegen, config, io, block_runner
from sailfish.geo import LBGeometry2D, LBGeometry3D

def _get_backends():
    for backend in ['cuda', 'opencl']:
        try:
            module = 'sailfish.backend_{}'.format(backend)
            __import__('sailfish', fromlist=["backend_{}".format(backend)])
            yield sys.modules[module].backend
        except ImportError:
            pass

def _start_block_runner(block, config, lb_class, backend_class, gpu_id):
    # We instantiate the backend class here (instead in the machine
    # master), so that the backend object is created within the
    # context of the new process.

    print 'block %d at PID %d' % (block.id, os.getpid())

    # FIXME: for some reason, this fails.
    backend = backend_class(config, gpu_id)
    #backend = None
    sim = lb_class(config)
    output = None
    runner = block_runner.BlockRunner(sim, block, output, backend)
    runner.run()

class LBBlockConnector(object):
    def __init__(self, array, send_ev, recv_ev):
        self._array = array
        self._send_ev = send_ev
        self._recv_ev = recv_ev

    def send(self, data):
        self._send_ev.set()

    def recv(self, data):
        self._recv_ev.wait()

class LBMachineMaster(object):
    def __init__(self, config, blocks, lb_class):
        self.blocks = blocks
        self.config = config
        self.lb_class = lb_class
        self.runners = []
        self._block_id_to_runner = {}
        self._pipes = []

    def _assign_blocks_to_gpus(self):
        block2gpu = {}
        # TODO: actually assign to different GPUs here
        for gpu, block in enumerate(self.blocks):
            block2gpu[block.id] = 0

        return block2gpu

    def run(self):
        block2gpu = self._assign_blocks_to_gpus()
        backend_class = _get_backends().next()

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
                array = Array('f', size)
                ev1 = Event()
                ev2 = Event()

                block.add_connector(nbid, LBBlockConnector(array, ev1, ev2))
                self.blocks[nbid].add_connector(block.id,
                        LBBlockConnector(array, ev2, ev1))

        for block in self.blocks:
            a, conn = Pipe()
            self._pipes.append(a)

            p = Process(target=_start_block_runner,
                        args=(block, self.config, self.lb_class,
                              backend_class, block2gpu[block.id]))
            self.runners.append(p)
            self._block_id_to_runner[block.id] = p

        for runner in self.runners:
            runner.start()

        for runner in self.runners:
            runner.join()

# TODO: eventually, these arguments will be passed asynchronously
# in a different way
def _start_machine_master(config, blocks, lb_class):
    master = LBMachineMaster(config, blocks, lb_class)
    master.run()

class GeometryError(Exception):
    pass

class LBGeometryProcessor(object):
    def __init__(self, blocks, dim):
        self.blocks = blocks
        self.dim = dim

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

    def _connect_blocks(self):
        connected = [False] * len(self.blocks)

        for axis in range(self.dim):
            for block in sorted(self.blocks, key=lambda x: x.location[axis]):
                higher_coord = block.location[axis] + block.size[axis]
                if higher_coord not in self._coord_map_list[axis]:
                    continue
                for neighbor_candidate in \
                        self._coord_map_list[axis][higher_coord]:
                    if block.connect(neighbor_candidate):
                        connected[block.id] = True
                        connected[neighbor_candidate.id] = True

        # Ensure every block is connected to at least one other block.
        if not all(connected):
            raise GeometryError()

    def transform(self):
        self._annotate()
        self._init_lower_coord_map()
        self._connect_blocks()
        return self.blocks

class LBSimulationController(object):
    """Controls the execution of a LB simulation."""

    def __init__(self, lb_class, lb_geo=None):
        self.conf = config.LBConfig()
        self._lb_class = lb_class
        self._lb_geo = lb_geo

        # Use a default global geometry is one has not been
        # specified explicitly.
        if lb_geo is None:
            if self.dim == 2:
                lb_geo = LBGeometry2D
            else:
                lb_geo = LBGeometry3D

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

        group = self.conf.add_group('Simulation-specific settings')
        lb_class.add_options(group)

        group = self.conf.add_group('Geometry settings')
        lb_geo.add_options(group)

        group = self.conf.add_group('Code generator options')
        codegen.BlockCodeGenerator.add_options(group)

        # Backend options
        for backend in _get_backends():
            group = self.conf.add_group(
                    "'{}' backend options".format(backend.name))
            backend.add_options(group)

        # TODO: visualization engine settings

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
        proc = LBGeometryProcessor(blocks, self.dim)
        blocks = proc.transform()

        # TODO: do this over MPI
        p = Process(target=_start_machine_master,
                    args=(self.conf, blocks, self._lb_class))
        p.start()
        p.join()

import sys
from multiprocessing import Process

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
    backend = backend_class(config, gpu_id)
    sim = lb_class(config)
    output = None
    runner = block_runner.BlockRunner(sim, block, output, backend)
    runner.run()

class LBMachineMaster(object):
    def __init__(self, config, blocks, lb_class):
        self.blocks = blocks
        self.config = config
        self.lb_class = lb_class
        self.runners = []
        self._block_id_to_runner = {}

    def _assign_blocks_to_gpus(self):
        block2gpu = {}
        # TODO: actually assign to different GPUs here
        for block in self.blocks:
            block2gpu[block.id] = 0

        return block2gpu

    def run(self):
        block2gpu = self._assign_blocks_to_gpus()
        backend_class = _get_backends().next()

        for block in self.blocks:
            p = Process(target=_start_block_runner,
                        args=(block, self.config, self.lb_class,
                              backend_class, block2gpu[block.id]))
            self.runners.append(p)
            self._block_id_to_runner[block.id] = p
            p.start()

        for runner in self.runners:
            runner.join()

# TODO: eventually, these arguments will be passed asynchronously
# in a different way
def _start_machine_master(config, blocks, lb_class):
    master = LBMachineMaster(config, blocks, lb_class)
    master.run()


class LBSimulationController(object):
    """Controls the execution of a LB simulation."""

    def __init__(self, lb_class, lb_geo=None):
        self.conf = config.LBConfig()
        self._lb_class = lb_class
        self._lb_geo = lb_geo

        # Use a default global geometry is one has not been
        # specified explicitly.
        if lb_geo is None:
            if lb_class.geo.dim == 2:
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

    def run(self):
        self.conf.parse()
        self.geo = self._lb_geo(self.conf)

        blocks = self.geo.blocks()

        # Assign IDs to blocks
        for i, block in enumerate(blocks):
            block.id = i

        # TODO: connect the blocks to each other here
        # XXX

        # TODO: do this over MPI
        p = Process(target=_start_machine_master,
                    args=(self.conf, blocks, self._lb_class))
        p.start()
        p.join()

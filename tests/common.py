import unittest
from dummy import *

from sailfish.backend_dummy import DummyBackend
from sailfish.config import LBConfig
from sailfish.lb_base import LBSim

class TestCase2D(unittest.TestCase):
    lattice_size = 64, 64

    def setUp(self):
        config = LBConfig()
        config.init_iters = 0
        config.seed = 0
        config.precision = 'single'
        config.block_size = 8
        config.mem_alignment = 8
        config.node_addressing = 'direct'
        config.lat_nx, config.lat_ny = self.lattice_size
        config.logger = DummyLogger()
        config.grid = 'D2Q9'
        config.mode = 'batch'
        config.periodic_x = False
        config.periodic_y = False
        config.use_link_tags = False
        self.sim = LBSim(config)
        self.config = config
        self.backend = DummyBackend()


class TestCase3D(unittest.TestCase):
    lattice_size = 32, 32, 16

    def setUp(self):
        config = LBConfig()
        config.init_iters = 0
        config.seed = 0
        config.precision = 'single'
        config.block_size = 8
        config.mem_alignment = 8
        config.node_addressing = 'direct'
        config.lat_nx, config.lat_ny, config.lat_nz = self.lattice_size
        config.logger = DummyLogger()
        config.grid = 'D3Q19'
        config.mode = 'batch'
        config.periodic_x = False
        config.periodic_y = False
        config.periodic_z = False
        config.use_link_tags = False
        self.sim = LBSim(config)
        self.backend = DummyBackend()

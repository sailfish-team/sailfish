import operator
import unittest
import numpy as np

from sailfish.config import LBConfig
from sailfish.lb_base import LBSim
from sailfish.backend_dummy import DummyBackend
from sailfish.block_runner import BlockRunner
from sailfish.geo_block import LBBlock2D, LBBlock3D

class TestBasicFunctionality(unittest.TestCase):
    location = 0, 0
    size = 10, 3

    location_3d = 0, 0, 0
    size_3d = 3, 5, 7

    def setUp(self):
        config = LBConfig()
        config.parse()
        config.precision = 'single'
        config.block_size = 8
        # Does not affect behaviour of any of the functions tested here.
        config.lat_nz, config.lat_nx, config.lat_ny = self.size_3d
        self.sim = LBSim(config)
        self.backend = DummyBackend()

    def get_block_runner(self, block):
        return BlockRunner(self.sim, block, output=None,
                backend=self.backend, quit_event=None)

    def test_block_connection(self):
        block = LBBlock2D(self.location, self.size)
        runner = self.get_block_runner(block)
        self.assertEqual(block.runner, runner)

    def test_strides_and_size_2d(self):
        block = LBBlock2D(self.location, self.size)
        block.set_actual_size(0)
        runner = self.get_block_runner(block)
        runner._init_shape()

        # Last dimension is rounded up to a multiple of block_size
        real_size = [3, 16]
        self.assertEqual(runner._physical_size, real_size)

        strides = runner._get_strides(np.float32)
        self.assertEqual(strides, [4 * 16, 4])
        nodes = runner._get_nodes()
        self.assertEqual(nodes, reduce(operator.mul, real_size))

    def test_strides_and_size_3d(self):
        block = LBBlock3D(self.location_3d, self.size_3d)
        block.set_actual_size(0)
        runner = self.get_block_runner(block)
        runner._init_shape()

        # Last dimension is rounded up to a multiple of block_size
        real_size = [7, 5, 8]
        self.assertEqual(runner._physical_size, real_size)
        strides = runner._get_strides(np.float64)
        self.assertEqual(strides, [8*8*5, 8*8, 8])
        nodes = runner._get_nodes()
        self.assertEqual(nodes, reduce(operator.mul, real_size))


if __name__ == '__main__':
    unittest.main()

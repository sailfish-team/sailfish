import numpy as np
import unittest
from sailfish.backend_dummy import DummyBackend
from sailfish.config import LBConfig
from sailfish.lb_base import LBSim
from sailfish.node_type import NTEquilibriumVelocity, multifield
from sailfish.subdomain import Subdomain2D, Subdomain3D, SubdomainSpec2D, SubdomainSpec3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.sym import D2Q9, D3Q19
from dummy import *


class SubdomainTest2D(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        where = (hx == hy)
        self.set_node(where,
                NTEquilibriumVelocity(
                    multifield((0.01 * (hx - self.gy / 2)**2, 0.0), where)))

class TestNodeTypeSetting2D(unittest.TestCase):
    lattice_size = 64, 64

    def setUp(self):
        config = LBConfig()
        config.precision = 'single'
        config.block_size = 8
        # Does not affect behaviour of any of the functions tested here.
        config.lat_nx, config.lat_ny = self.lattice_size
        config.logger = DummyLogger()
        self.sim = LBSim(config)
        self.config = config
        self.backend = DummyBackend()

    def test_array_setting(self):
        envelope = 1
        spec = SubdomainSpec2D((0, 0), self.lattice_size,
                envelope_size=envelope, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = SubdomainTest2D(list(reversed(self.lattice_size)), spec, D2Q9)
        sub.reset()

        center = 64 / 2
        for y in range(0, 64):
            np.testing.assert_array_almost_equal(
                    np.float64([0.01 * (y - center)**2, 0.0]),
                    np.float64(sub._encoder.get_param((y + envelope, y + envelope), 2)))


class SubdomainTest3D(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        where = np.logical_and((hx == hy), (hy == hz))
        self.set_node(where,
                NTEquilibriumVelocity(
                    multifield((0.01 * (hy - self.gy / 2)**2,
                        0.03 * (hz - self.gz / 2)**2, 0.0), where)))

class TestNodeTypeSetting3D(unittest.TestCase):
    lattice_size = 32, 32, 16

    def setUp(self):
        config = LBConfig()
        config.precision = 'single'
        config.block_size = 8
        # Does not affect behaviour of any of the functions tested here.
        config.lat_nx, config.lat_ny, config.lat_nz = self.lattice_size
        config.logger = DummyLogger()
        self.sim = LBSim(config)
        self.backend = DummyBackend()

    def test_array_setting(self):
        envelope = 1
        spec = SubdomainSpec3D((0, 0, 0), self.lattice_size,
                envelope_size=envelope, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = SubdomainTest3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.reset()

        center_y = 32 / 2
        center_z = 16 / 2
        for y in range(0, 16):
            np.testing.assert_array_almost_equal(
                    np.float64([0.01 * (y - center_y)**2,
                        0.03 * (y - center_z)**2, 0.0]),
                    np.float64(sub._encoder.get_param(
                        (y + envelope, y + envelope, y + envelope), 3)))


if __name__ == '__main__':
    unittest.main()

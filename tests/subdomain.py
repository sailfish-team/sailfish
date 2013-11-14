import numpy as np
import unittest
from sailfish.node_type import NTEquilibriumVelocity, multifield, NTFullBBWall, _NTUnused, _NTPropagationOnly
from sailfish.subdomain import Subdomain2D, Subdomain3D, SubdomainSpec2D, SubdomainSpec3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.sym import D2Q9, D3Q19
from common import TestCase2D, TestCase3D

class SubdomainTest2D(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        where = (hx == hy)
        self.set_node(where, NTEquilibriumVelocity(
            multifield((0.01 * (hx - self.gy / 2)**2, 0.0), where)))

        self.set_node((hx > 10) & (hy < 5), NTFullBBWall)

class TestNodeTypeSetting2D(TestCase2D):
    def test_array_setting(self):
        envelope = 1
        spec = SubdomainSpec2D((0, 0), self.lattice_size,
                               envelope_size=envelope, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = SubdomainTest2D(list(reversed(self.lattice_size)), spec, D2Q9)
        sub.allocate()
        sub.reset()

        center = 64 / 2
        for y in range(0, 64):
            np.testing.assert_array_almost_equal(
                    np.float64([0.01 * (y - center)**2, 0.0]),
                    np.float64(sub._encoder.get_param((y + envelope, y + envelope), 2)))

        np.testing.assert_equal(sub._type_map[1:2, 13:-1], _NTUnused.id)
        np.testing.assert_equal(sub._type_map[1:2, 12], _NTPropagationOnly.id)
        np.testing.assert_equal(sub._type_map[3, 12:-1], _NTPropagationOnly.id)

class SubdomainTest3D(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        where = np.logical_and((hx == hy), (hy == hz))
        self.set_node(where, NTEquilibriumVelocity(
            multifield((0.01 * (hy - self.gy / 2)**2,
                        0.03 * (hz - self.gz / 2)**2, 0.0), where)))
        self.set_node((hx > 10) & (hy < 5) & (hz < 7), NTFullBBWall)

class TestNodeTypeSetting3D(TestCase3D):
    def test_array_setting(self):
        envelope = 1
        spec = SubdomainSpec3D((0, 0, 0), self.lattice_size,
                               envelope_size=envelope, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = SubdomainTest3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.allocate()
        sub.reset()

        center_y = 32 / 2
        center_z = 16 / 2
        for y in range(0, 16):
            np.testing.assert_array_almost_equal(
                    np.float64([0.01 * (y - center_y)**2,
                        0.03 * (y - center_z)**2, 0.0]),
                    np.float64(sub._encoder.get_param(
                        (y + envelope, y + envelope, y + envelope), 3)))

        np.testing.assert_equal(sub._type_map[1:5, 1:2, 13:-1], _NTUnused.id)
        np.testing.assert_equal(sub._type_map[5, 3, 12:-1], _NTPropagationOnly.id)

if __name__ == '__main__':
    unittest.main()

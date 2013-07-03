import unittest
import numpy as np
from sailfish.node_type import NTEquilibriumDensity
from sailfish.subdomain import Subdomain2D, Subdomain3D, SubdomainSpec2D, SubdomainSpec3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.sym import D2Q9, D3Q19
from common import TestCase2D, TestCase3D

class TestSubdomain2D(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        self.set_node((hx == 0) | (hy == 0) | (hx == self.gx - 1) |
                      (hy == self.gy - 1), NTEquilibriumDensity(1.0))

class TestOrientationDetection2D(TestCase2D):
    def test_orientation(self):
        spec = SubdomainSpec2D((0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = TestSubdomain2D(list(reversed(self.lattice_size)), spec, D2Q9)
        sub.allocate()
        sub.reset()
        sub._encoder.detect_orientation(sub._orientation_base)

        nx, ny = self.lattice_size
        hx, hy = sub._get_mgrid()
        np.testing.assert_equal(sub._orientation[(hx == 0) & (hy > 0) &
                                                 (hy < ny - 1)],
                                D2Q9.vec_to_dir([1, 0]))
        np.testing.assert_equal(sub._orientation[(hx == nx - 1) & (hy > 0) &
                                                 (hy < ny - 1)],
                                D2Q9.vec_to_dir([-1, 0]))
        np.testing.assert_equal(sub._orientation[(hy == 0) & (hx > 0) &
                                                 (hx < nx - 1)],
                                D2Q9.vec_to_dir([0, 1]))
        np.testing.assert_equal(sub._orientation[(hy == ny - 1) & (hx > 0) &
                                                 (hx < nx - 1)],
                                D2Q9.vec_to_dir([0, -1]))

        # No orientation vector for corner nodes.
        np.testing.assert_equal(sub._orientation[0, 0], 0)
        np.testing.assert_equal(sub._orientation[0, nx - 1], 0)
        np.testing.assert_equal(sub._orientation[ny - 1, 0], 0)
        np.testing.assert_equal(sub._orientation[ny - 1, nx - 1], 0)


class TestSubdomain3D(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        self.set_node((hx == 0) | (hy == 0) | (hz == 0) | (hz == self.gz - 1) |
                      (hx == self.gx - 1) | (hy == self.gy - 1),
                      NTEquilibriumDensity(1.0))

class TestOrientationDetection3D(TestCase3D):
    def test_orientation(self):
        spec = SubdomainSpec3D((0, 0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = TestSubdomain3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.allocate()
        sub.reset()
        sub._encoder.detect_orientation(sub._orientation_base)

        nx, ny, nz = self.lattice_size
        hx, hy, hz = sub._get_mgrid()

        xx = np.logical_not((hx == 0) | (hx == nx - 1))
        yy = np.logical_not((hy == 0) | (hy == ny - 1))
        zz = np.logical_not((hz == 0) | (hz == nz - 1))

        np.testing.assert_equal(sub._orientation[(hx == 0) & yy & zz],
                                D3Q19.vec_to_dir([1, 0, 0]))
        np.testing.assert_equal(sub._orientation[(hy == 0) & xx & zz],
                                D3Q19.vec_to_dir([0, 1, 0]))
        np.testing.assert_equal(sub._orientation[(hz == 0) & yy & xx],
                                D3Q19.vec_to_dir([0, 0, 1]))
        np.testing.assert_equal(sub._orientation[(hx == nx - 1) & yy & zz],
                                D3Q19.vec_to_dir([-1, 0, 0]))
        np.testing.assert_equal(sub._orientation[(hy == ny - 1) & xx & zz],
                                D3Q19.vec_to_dir([0, -1, 0]))
        np.testing.assert_equal(sub._orientation[(hz == nz - 1) & yy & xx],
                                D3Q19.vec_to_dir([0, 0, -1]))

        # No orientation vector for edge nodes.
        np.testing.assert_equal(sub._orientation[(hx == 0) & (hy == 0)], 0)
        np.testing.assert_equal(sub._orientation[(hx == 0) & (hz == 0)], 0)
        np.testing.assert_equal(sub._orientation[(hz == 0) & (hy == 0)], 0)

        np.testing.assert_equal(sub._orientation[(hx == 0) & (hy == ny - 1)], 0)
        np.testing.assert_equal(sub._orientation[(hx == 0) & (hz == nz - 1)], 0)
        np.testing.assert_equal(sub._orientation[(hz == 0) & (hy == ny - 1)], 0)

        np.testing.assert_equal(sub._orientation[(hx == nx - 1) & (hy == 0)], 0)
        np.testing.assert_equal(sub._orientation[(hx == nx - 1) & (hz == 0)], 0)
        np.testing.assert_equal(sub._orientation[(hz == nz - 1) & (hy == 0)], 0)

        np.testing.assert_equal(sub._orientation[(hx == nx - 1) & (hy == ny - 1)], 0)
        np.testing.assert_equal(sub._orientation[(hx == nx - 1) & (hz == nz - 1)], 0)
        np.testing.assert_equal(sub._orientation[(hz == nz - 1) & (hy == ny - 1)], 0)

        # No orientation vector for corner nodes.
        np.testing.assert_equal(sub._orientation[0, 0, 0], 0)
        np.testing.assert_equal(sub._orientation[0, 0, nx - 1], 0)
        np.testing.assert_equal(sub._orientation[0, ny - 1, 0], 0)
        np.testing.assert_equal(sub._orientation[0, ny - 1, nx - 1], 0)
        np.testing.assert_equal(sub._orientation[nz - 1, 0, 0], 0)
        np.testing.assert_equal(sub._orientation[nz - 1, 0, nx - 1], 0)
        np.testing.assert_equal(sub._orientation[nz - 1, ny - 1, 0], 0)
        np.testing.assert_equal(sub._orientation[nz - 1, ny - 1, nx - 1], 0)

if __name__ == '__main__':
    unittest.main()

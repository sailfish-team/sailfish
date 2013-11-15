import numpy as np
import operator
import unittest
from sailfish.node_type import NTEquilibriumDensity, NTEquilibriumVelocity, multifield, NTFullBBWall, _NTUnused, _NTPropagationOnly, NTHalfBBWall
from sailfish.subdomain import Subdomain2D, Subdomain3D, SubdomainSpec2D, SubdomainSpec3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.sym import D2Q9, D3Q19
from common import TestCase2D, TestCase3D


# Basic node type setting.
# ========================

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


# Orientation detection.
# ======================

class OrientationSubdomain2D(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        self.set_node((hx == 0) | (hy == 0) | (hx == self.gx - 1) |
                      (hy == self.gy - 1), NTEquilibriumDensity(1.0))

class TestOrientationDetection2D(TestCase2D):
    def test_orientation(self):
        spec = SubdomainSpec2D((0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = OrientationSubdomain2D(list(reversed(self.lattice_size)), spec, D2Q9)
        sub.allocate()
        sub.reset()

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


class OrientationSubdomain3D(Subdomain3D):
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
        sub = OrientationSubdomain3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.allocate()
        sub.reset()

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


# Link tagging.
# =============

class TestLinkTagging3D(TestCase3D):
    def test_periodic_yz(self):
        self.sim.config.use_link_tags = True
        self.sim.config.periodic_y = True
        self.sim.config.periodic_z = True

        spec = SubdomainSpec3D((0, 0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec.enable_local_periodicity(1)
        spec.enable_local_periodicity(2)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()

        class _LinkTaggingSubdomain3D(Subdomain3D):
            def boundary_conditions(self, hx, hy, hz):
                self.set_node((hx == 0) | (hx == self.gx - 1), NTHalfBBWall)

        sub = _LinkTaggingSubdomain3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.allocate()
        sub.reset()

        nx, ny, nz = self.lattice_size
        hx, hy, hz = sub._get_mgrid()

        hx_tags = reduce(operator.or_, ((1 << i) for i, vec in
                         enumerate(D3Q19.basis[1:]) if vec[0] >= 0))
        np.testing.assert_equal(hx_tags, sub._orientation[hx == 0])

        hx_tags = reduce(operator.or_, ((1 << i) for i, vec in
                         enumerate(D3Q19.basis[1:]) if vec[0] <= 0))
        np.testing.assert_equal(hx_tags, sub._orientation[hx == nx - 1])

    def test_periodic_xz(self):
        self.sim.config.use_link_tags = True
        self.sim.config.periodic_x = True
        self.sim.config.periodic_z = True

        spec = SubdomainSpec3D((0, 0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec.enable_local_periodicity(0)
        spec.enable_local_periodicity(2)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()

        class _LinkTaggingSubdomain3D(Subdomain3D):
            def boundary_conditions(self, hx, hy, hz):
                self.set_node((hy == 0) | (hy == self.gy - 1), NTHalfBBWall)

        sub = _LinkTaggingSubdomain3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.allocate()
        sub.reset()

        nx, ny, nz = self.lattice_size
        hx, hy, hz = sub._get_mgrid()

        hy_tags = reduce(operator.or_, ((1 << i) for i, vec in
                         enumerate(D3Q19.basis[1:]) if vec[1] >= 0))
        np.testing.assert_equal(hy_tags, sub._orientation[hy == 0])

        hy_tags = reduce(operator.or_, ((1 << i) for i, vec in
                         enumerate(D3Q19.basis[1:]) if vec[1] <= 0))
        np.testing.assert_equal(hy_tags, sub._orientation[hy == ny - 1])


if __name__ == '__main__':
    unittest.main()

import numpy as np
import operator
import unittest
from sailfish.controller import LBGeometryProcessor
from sailfish.node_type import NTEquilibriumDensity, NTEquilibriumVelocity, multifield, NTFullBBWall, _NTUnused, _NTPropagationOnly, NTHalfBBWall, DynamicValue, LinearlyInterpolatedTimeSeries
from sailfish.subdomain import Subdomain2D, Subdomain3D, SubdomainSpec2D, SubdomainSpec3D, SubdomainPair
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.sym import D2Q9, D3Q19, S
from common import TestCase2D, TestCase3D


# Basic node type setting.
# ========================

class SubdomainTest2D(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        where = (hx == hy)
        self.set_node(where, NTEquilibriumVelocity(
            multifield((0.01 * (hx - self.gy / 2)**2, 0.0), where)))

        where = ((hx == 5) & (hy == 7))
        self.set_node(where, NTEquilibriumDensity(
            DynamicValue(0.1 * S.gx)))

        # Interpolated time series.
        data = np.linspace(0, 50, 10)
        where = ((hx == 5) & (hy == 8))
        self.set_node(where, NTEquilibriumDensity(
            DynamicValue(0.1 * S.gx * LinearlyInterpolatedTimeSeries(data, 40))))

        # Same underlying data, but different time step.
        where = ((hx == 5) & (hy == 9))
        self.set_node(where, NTEquilibriumDensity(
            DynamicValue(0.1 * S.gx * LinearlyInterpolatedTimeSeries(data, 30))))

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
        sub.reset(encode=False)

        center = 64 / 2
        for y in range(0, 64):
            np.testing.assert_array_almost_equal(
                    np.float64([0.01 * (y - center)**2, 0.0]),
                    np.float64(sub._encoder.get_param((y + envelope, y + envelope), 2)))

        np.testing.assert_equal(sub._type_map[1:2, 13:-1], _NTUnused.id)
        np.testing.assert_equal(sub._type_map[1:2, 12], _NTPropagationOnly.id)
        np.testing.assert_equal(sub._type_map[3, 12:-1], _NTPropagationOnly.id)
        self.assertTrue(sub.config.time_dependence)
        self.assertTrue(sub.config.space_dependence)

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
        sub.reset(encode=False)

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

class ChannelSubdomain3D(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        self.set_node((hx == 0) | (hx == self.gx-1), NTHalfBBWall)

class TestOrientationDetection3D(TestCase3D):
    def test_orientation_channel_pbc(self):
        self.sim.config.periodic_z = True
        self.sim.config.periodic_y = True
        spec = SubdomainSpec3D((0, 0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec.enable_local_periodicity(1)
        spec.enable_local_periodicity(2)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()
        sub = ChannelSubdomain3D(list(reversed(self.lattice_size)), spec, D3Q19)
        sub.allocate()
        sub.reset()

        nx, ny, nz = self.lattice_size
        hx, hy, hz = sub._get_mgrid()

        np.testing.assert_equal(sub._orientation[(hx == 0)],
                                D3Q19.vec_to_dir([1, 0, 0]))
        np.testing.assert_equal(sub._orientation[(hx == nx - 1)],
                                D3Q19.vec_to_dir([-1, 0, 0]))

    def test_orientation_channel_2subdomains(self):
        self.sim.config.periodic_z = True
        self.sim.config.periodic_y = True
        spec0 = SubdomainSpec3D((0, 0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec0.enable_local_periodicity(1)
        spec1 = SubdomainSpec3D((0, 0, self.lattice_size[2]),
                                 self.lattice_size, envelope_size=1, id_=1)
        spec1.enable_local_periodicity(1)

        self.assertTrue(spec0.connect(spec1, grid=D3Q19))

        spec0.runner = SubdomainRunner(self.sim, spec0, output=None,
                                       backend=self.backend, quit_event=None)
        spec0.runner._init_shape()
        spec1.runner = SubdomainRunner(self.sim, spec1, output=None,
                                       backend=self.backend, quit_event=None)
        spec1.runner._init_shape()

        sub0 = ChannelSubdomain3D(list(reversed(self.lattice_size)), spec0, D3Q19)
        sub0.allocate()
        sub0.reset()

        sub1 = ChannelSubdomain3D(list(reversed(self.lattice_size)), spec1, D3Q19)
        sub1.allocate()
        sub1.reset()

        nx, ny, nz = self.lattice_size
        hx, hy, hz = sub0._get_mgrid()

        np.testing.assert_equal(sub0._orientation[(hx == 0)],
                                D3Q19.vec_to_dir([1, 0, 0]))
        np.testing.assert_equal(sub0._orientation[(hx == nx - 1)],
                                D3Q19.vec_to_dir([-1, 0, 0]))
        np.testing.assert_equal(sub1._orientation[(hx == 0)],
                                D3Q19.vec_to_dir([1, 0, 0]))
        np.testing.assert_equal(sub1._orientation[(hx == nx - 1)],
                                D3Q19.vec_to_dir([-1, 0, 0]))

    def test_orientation_box(self):
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

class TestLinkTaggingInterior2D(TestCase2D):
    def setUp(self):
        TestCase2D.setUp(self)
        self.sim.config.use_link_tags = True

    def testSolidInteriorNodes(self):
        spec = SubdomainSpec2D((0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()

        class _LinkTaggingSubdomain2D(Subdomain2D):
            def boundary_conditions(self, hx, hy):
                box = (hx >= 5) & (hx <= 10) & (hy >= 5) & (hy <= 10)
                self.set_node(box, NTHalfBBWall)

        sub = _LinkTaggingSubdomain2D(list(reversed(self.lattice_size)), spec,
                                      D2Q9)
        sub.allocate()
        sub.reset(encode=False)

        # Outer layer is bounce-back nodes.
        np.testing.assert_equal(sub._type_map[5:11, 5], NTHalfBBWall.id)
        np.testing.assert_equal(sub._type_map[5:11, 10], NTHalfBBWall.id)
        np.testing.assert_equal(sub._type_map[5, 5:11], NTHalfBBWall.id)
        np.testing.assert_equal(sub._type_map[10, 5:11], NTHalfBBWall.id)

        # Inner layer are propagation-only nodes.
        np.testing.assert_equal(sub._type_map[6:10, 6], _NTPropagationOnly.id)
        np.testing.assert_equal(sub._type_map[6:10, 9], _NTPropagationOnly.id)
        np.testing.assert_equal(sub._type_map[6, 6:10], _NTPropagationOnly.id)
        np.testing.assert_equal(sub._type_map[9, 6:10], _NTPropagationOnly.id)

        # Core of unused nodes.
        np.testing.assert_equal(sub._type_map[7:9, 7:9], _NTUnused.id)


class TestLinkTagging3D(TestCase3D):
    def setUp(self):
        TestCase3D.setUp(self)
        self.sim.config.use_link_tags = True

    def testBoundaryNodes(self):
        spec = SubdomainSpec3D((0, 0, 0), (5, 8, 3), envelope_size=1, id_=0)
        spec.runner = SubdomainRunner(self.sim, spec, output=None,
                                      backend=self.backend, quit_event=None)
        spec.runner._init_shape()

        class _LinkTaggingSubdomain3D(Subdomain3D):
            def boundary_conditions(self, hx, hy, hz):
                wall_map = np.array([
                    [[0, 0, 0, 0, 1],
                     [0, 0, 0, 0, 1],
                     [0, 0, 0, 1, 1],
                     [0, 0, 0, 1, 1],  # There was a bug once that caused the
                                       # middle node here to be marked
                                       # PropagationOnly.
                     [0, 0, 1, 1, 1],
                     [0, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[0, 0, 0, 1, 1],
                     [0, 0, 0, 1, 1],
                     [0, 0, 1, 1, 1],
                     [0, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[0, 0, 1, 1, 1],
                     [0, 1, 1, 1, 1],
                     [0, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]]], dtype=np.bool)
                wall_map = np.pad(wall_map, (1, 1), mode='constant',
                                  constant_values=0)
                self.set_node(wall_map, NTFullBBWall)
                self.set_node(np.logical_not(wall_map) & (hz == 0),
                            NTEquilibriumDensity(1.0))

        sub = _LinkTaggingSubdomain3D((3, 8, 5), spec, D3Q19)
        sub.allocate()
        sub.reset(encode=False)

        expected = np.array([
             [ 3,  3,  3,  3,  5],
             [ 3,  3,  3,  3,  5],
             [ 3,  3,  3,  5,  5],
             [ 3,  3,  3,  5, 19],
             [ 3,  3,  5,  5, 19],
             [ 3,  5,  5, 19, 19],
             [ 5,  5, 19, 19, 20],
             [19, 19, 19, 20, 20]])
        expected[expected == 3] = NTEquilibriumDensity.id
        expected[expected == 5] = NTFullBBWall.id
        expected[expected == 19] = _NTPropagationOnly.id
        expected[expected == 20] = _NTUnused.id
        np.testing.assert_equal(expected, sub._type_map[0,:,:])

    def test_periodic_yz(self):
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

    def test_periodic_yz_2subdomains(self):
        self.sim.config.periodic_y = True
        self.sim.config.periodic_z = True

        spec0 = SubdomainSpec3D((0, 0, 0), self.lattice_size, envelope_size=1, id_=0)
        spec1 = SubdomainSpec3D((0, 0, self.lattice_size[2]), self.lattice_size, envelope_size=1, id_=1)

        nx, ny, nz = self.lattice_size
        spec0, spec1 = LBGeometryProcessor([spec0, spec1], 3, (nx, ny, 2 * nz)).transform(self.sim.config)

        spec0.runner = SubdomainRunner(self.sim, spec0, output=None,
                                       backend=self.backend, quit_event=None)
        spec0.runner._init_shape()
        spec1.runner = SubdomainRunner(self.sim, spec1, output=None,
                                       backend=self.backend, quit_event=None)
        spec1.runner._init_shape()

        class _LinkTaggingSubdomain3D(Subdomain3D):
            def boundary_conditions(self, hx, hy, hz):
                self.set_node((hx == 0) | (hx == self.gx - 1), NTHalfBBWall)

        sub0 = _LinkTaggingSubdomain3D(list(reversed(self.lattice_size)), spec0, D3Q19)
        sub0.allocate()
        sub0.reset()
        sub1 = _LinkTaggingSubdomain3D(list(reversed(self.lattice_size)), spec1, D3Q19)
        sub1.allocate()
        sub1.reset()

        hx, hy, hz = sub1._get_mgrid()
        hx_tags = reduce(operator.or_, ((1 << i) for i, vec in
                         enumerate(D3Q19.basis[1:]) if vec[0] >= 0))
        np.testing.assert_equal(hx_tags, sub0._orientation[hx == 0])
        np.testing.assert_equal(hx_tags, sub1._orientation[hx == 0])

        hx_tags = reduce(operator.or_, ((1 << i) for i, vec in
                         enumerate(D3Q19.basis[1:]) if vec[0] <= 0))
        np.testing.assert_equal(hx_tags, sub0._orientation[hx == nx - 1])
        np.testing.assert_equal(hx_tags, sub1._orientation[hx == nx - 1])

    def test_periodic_xz(self):
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

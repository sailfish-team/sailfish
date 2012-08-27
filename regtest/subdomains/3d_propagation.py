#!/usr/bin/env python

import os
import shutil
import tempfile
import unittest

import numpy as np

from sailfish import io
from sailfish.geo import LBGeometry3D
from sailfish.subdomain import SubdomainSpec3D, Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.sym import D3Q19, D3Q15
from regtest.subdomains import util


class BlockTest(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        pass

    def initial_conditions(self, sim, hx, hy, hz):
        pass

class TwoBlocksXConnGeoTest(LBGeometry3D):
    def subdomains(self, n=None):
        blocks = []
        blocks.append(SubdomainSpec3D((0, 0, 0), (64, 64, 66)))
        blocks.append(SubdomainSpec3D((64, 0, 0), (64, 64, 66)))
        return blocks

class TwoBlocksYConnGeoTest(LBGeometry3D):
    def subdomains(self, n=None):
        blocks = []
        blocks.append(SubdomainSpec3D((0, 0, 0), (64, 64, 66)))
        blocks.append(SubdomainSpec3D((0, 64, 0), (64, 64, 66)))
        return blocks


class TwoBlocksZConnGeoTest(LBGeometry3D):
    def subdomains(self, n=None):
        blocks = []
        blocks.append(SubdomainSpec3D((0, 0, 0), (64, 66, 64)))
        blocks.append(SubdomainSpec3D((0, 0, 64), (64, 66, 64)))
        return blocks


mem_align = 32
block_size = 64
tmpdir = None
periodic_x = False
vi = lambda x, y, z: D3Q19.vec_idx([x, y, z])
vi2 = lambda x, y, z: D3Q15.vec_idx([x, y, z])


class SimulationTest(LBFluidSim):
    subdomain = BlockTest
    axis = 0

    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False

    @classmethod
    def update_defaults(cls, defaults):
        if cls.axis == 0:
            lat_nx = 128
            lat_ny = 64
            lat_nz = 66
        elif cls.axis == 1:
            lat_ny = 128
            lat_nx = 64
            lat_nz = 66
        elif cls.axis == 2:
            lat_nz = 128
            lat_nx = 64
            lat_ny = 66

        global block_size, tmpdir
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'lat_nx': lat_nx,
            'lat_ny': lat_ny,
            'lat_nz': lat_nz,
            'grid': 'D3Q19',
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'test_out'),
            'debug_dump_dists': True,
            'cuda_cache': False,
            'check_invalid_results_gpu': False,
            'check_invalid_results_host': False,
        })

    # Permutation functions.
    @classmethod
    def p1(cls, x, y, z):
        if cls.axis == 0:
            return x, y, z
        elif cls.axis == 1:
            return y, x, z
        elif cls.axis == 2:
            return y, z, x

    @classmethod
    def p2(cls, d, z, y, x):
        if cls.axis == 0:
            return d, z, y, x
        elif cls.axis == 1:
            return d, z, x, y
        elif cls.axis == 2:
            return d, x, z, y

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        p1 = self.p1
        p2 = self.p2

        # dbuf indices are: dist, z, y, x
        # vec indices are: x, y, z
        if runner._block.id == 0:
            # All distributions from a single node in the
            # middle of the face
            dbuf[p2(vi(*p1(1, 0, 0)), 32, 32, 64)] = 0.11
            dbuf[p2(vi(*p1(1, 1, 0)), 32, 32, 64)] = 0.12
            dbuf[p2(vi(*p1(1, -1, 0)), 32, 32, 64)] = 0.13
            dbuf[p2(vi(*p1(1, 0, 1)), 32, 32, 64)] = 0.21
            dbuf[p2(vi(*p1(1, 0, -1)), 32, 32, 64)] = 0.23
        elif runner._block.id == 1:
            # Corner
            dbuf[p2(vi(*p1(-1, 0, 0)), 1, 1, 1)] = 0.33
            dbuf[p2(vi(*p1(-1, 1, 0)), 1, 1, 1)] = 0.34
            dbuf[p2(vi(*p1(-1, 0, 1)), 1, 1, 1)] = 0.35

            # Edge
            dbuf[p2(vi(*p1(-1, 0, 0)), 1, 32, 1)] = 0.44
            dbuf[p2(vi(*p1(-1, 1, 0)), 1, 32, 1)] = 0.45
            dbuf[p2(vi(*p1(-1, -1, 0)), 1, 32, 1)] = 0.46
            dbuf[p2(vi(*p1(-1, 0, 1)), 1, 32, 1)] = 0.47

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)


class TwoBlockPropagationTest(unittest.TestCase):
    def _verify(self, b0, b1, cls):
        p1 = cls.p1
        p2 = cls.p2
        ae = np.testing.assert_equal

        ae(b1[p2(vi(*p1(1, 0, 0)), 32, 32, 1)], np.float32(0.11))
        ae(b1[p2(vi(*p1(1, 1, 0)), 32, 33, 1)], np.float32(0.12))
        ae(b1[p2(vi(*p1(1, -1, 0)), 32, 31, 1)], np.float32(0.13))
        ae(b1[p2(vi(*p1(1, 0, 1)), 33, 32, 1)], np.float32(0.21))
        ae(b1[p2(vi(*p1(1, 0, -1)), 31, 32, 1)], np.float32(0.23))

        ae(b0[p2(vi(*p1(-1, 0, 0)), 1, 1, 64)], np.float32(0.33))
        ae(b0[p2(vi(*p1(-1, 1, 0)), 1, 2, 64)], np.float32(0.34))
        ae(b0[p2(vi(*p1(-1, 0, 1)), 2, 1, 64)], np.float32(0.35))

        ae(b0[p2(vi(*p1(-1, 0, 0)), 1, 32, 64)], np.float32(0.44))
        ae(b0[p2(vi(*p1(-1, 1, 0)), 1, 33, 64)], np.float32(0.45))
        ae(b0[p2(vi(*p1(-1, -1, 0)), 1, 31, 64)], np.float32(0.46))
        ae(b0[p2(vi(*p1(-1, 0, 1)), 2, 32, 64)], np.float32(0.47))

    def test_horiz_spread(self):
        global tmpdir
        HorizTest = type('HorizTest', (SimulationTest,), {'axis': 0})
        ctrl = LBSimulationController(HorizTest, TwoBlocksXConnGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        b1 = np.load(io.dists_filename(output, 1, 1, 1))
        self._verify(b0, b1, HorizTest)

    def test_vert_spread(self):
        global tmpdir
        VertTest = type('VertTest', (SimulationTest,), {'axis': 1})
        ctrl = LBSimulationController(VertTest, TwoBlocksYConnGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        b1 = np.load(io.dists_filename(output, 1, 1, 1))
        self._verify(b0, b1, VertTest)

    def test_depth_spread(self):
        global tmpdir
        DepthTest = type('DepthTest', (SimulationTest,), {'axis': 2})
        ctrl = LBSimulationController(DepthTest, TwoBlocksZConnGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        b1 = np.load(io.dists_filename(output, 1, 1, 1))
        self._verify(b0, b1, DepthTest)


#############################################################################

class TwoBlocksShiftedYConnGeoTest(LBGeometry3D):
    def subdomains(self, n=None):
        return [SubdomainSpec3D((0, 0, 0), (50, 30, 13)),
                SubdomainSpec3D((5, 30, 0), (50, 10, 13))]

class TwoBlocksShiftedXConnGeoTest(LBGeometry3D):
    def subdomains(self, n=None):
        return [SubdomainSpec3D((0, 0, 0), (20, 15, 30)),
                SubdomainSpec3D((20, 2, 7), (10, 15, 30))]

class MisalignedHorizTest(SimulationTest):
    @classmethod
    def update_defaults(cls, defaults):
        SimulationTest.update_defaults(defaults)
        defaults.update({
            'lat_nx': 30,
            'lat_ny': 17,
            'lat_nz': 37,
            })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, z, y, x
        # vec indices are: x, y, z
        if runner._block.id == 0:
            dbuf[vi(1, 0, 0),  15, 7, 20] = 0.11
            dbuf[vi(1, 1, 0),  15, 7, 20] = 0.12
            dbuf[vi(1, -1, 0), 15, 7, 20] = 0.13
            dbuf[vi(1, 0, 1),  15, 7, 20] = 0.14
            dbuf[vi(1, 0, -1), 15, 7, 20] = 0.15
        elif runner._block.id == 1:
            dbuf[vi(-1, 0, 0),  10, 7, 1] = 0.21
            dbuf[vi(-1, 1, 0),  10, 7, 1] = 0.22
            dbuf[vi(-1, -1, 0), 10, 7, 1] = 0.23
            dbuf[vi(-1, 0, 1),  10, 7, 1] = 0.24
            dbuf[vi(-1, 0, -1), 10, 7, 1] = 0.25

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class MisalignedVertTest(SimulationTest):
    @classmethod
    def update_defaults(cls, defaults):
        SimulationTest.update_defaults(defaults)
        defaults.update({
            'lat_nx': 55,
            'lat_ny': 40,
            'lat_nz': 13,
            })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, z, y, x
        # vec indices are: x, y, z
        if runner._block.id == 0:
            dbuf[vi(0, 1, 0),  5, 30, 25] = 0.11
            dbuf[vi(1, 1, 0),  5, 30, 25] = 0.12
            dbuf[vi(-1, 1, 0), 5, 30, 25] = 0.13
            dbuf[vi(0, 1, 1),  5, 30, 25] = 0.14
            dbuf[vi(0, 1, -1), 5, 30, 25] = 0.15
        elif runner._block.id == 1:
            dbuf[vi(0, -1, 0),  5, 1, 25] = 0.21
            dbuf[vi(1, -1, 0),  5, 1, 25] = 0.22
            dbuf[vi(-1, -1, 0), 5, 1, 25] = 0.23
            dbuf[vi(0, -1, 1),  5, 1, 25] = 0.24
            dbuf[vi(0, -1, -1), 5, 1, 25] = 0.25

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class TwoBlockMisalignedPropagationTest(unittest.TestCase):
    def test_vert_spread(self):
        ctrl = LBSimulationController(MisalignedVertTest,
                TwoBlocksShiftedYConnGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        b1 = np.load(io.dists_filename(output, 1, 1, 1))

        ae = np.testing.assert_equal
        ae(b1[vi(0, 1, 0),  5, 1, 20], np.float32(0.11))
        ae(b1[vi(1, 1, 0),  5, 1, 21], np.float32(0.12))
        ae(b1[vi(-1, 1, 0), 5, 1, 19], np.float32(0.13))
        ae(b1[vi(0, 1, 1),  6, 1, 20], np.float32(0.14))
        ae(b1[vi(0, 1, -1), 4, 1, 20], np.float32(0.15))

        ae(b0[vi(0, -1, 0),  5, 30, 30], np.float32(0.21))
        ae(b0[vi(1, -1, 0),  5, 30, 31], np.float32(0.22))
        ae(b0[vi(-1, -1, 0), 5, 30, 29], np.float32(0.23))
        ae(b0[vi(0, -1, 1),  6, 30, 30], np.float32(0.24))
        ae(b0[vi(0, -1, -1), 4, 30, 30], np.float32(0.25))

    def test_horiz_spread(self):
        ctrl = LBSimulationController(MisalignedHorizTest,
                TwoBlocksShiftedXConnGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        b1 = np.load(io.dists_filename(output, 1, 1, 1))

        ae = np.testing.assert_equal
        ae(b1[vi(1, 0, 0),  8, 5, 1], np.float32(0.11))
        ae(b1[vi(1, 1, 0),  8, 6, 1], np.float32(0.12))
        ae(b1[vi(1, -1, 0), 8, 4, 1], np.float32(0.13))
        ae(b1[vi(1, 0, 1),  9, 5, 1], np.float32(0.14))
        ae(b1[vi(1, 0, -1), 7, 5, 1], np.float32(0.15))

        ae(b0[vi(-1, 0, 0),  17, 9, 20], np.float32(0.21))
        ae(b0[vi(-1, 1, 0),  17, 10, 20], np.float32(0.22))
        ae(b0[vi(-1, -1, 0), 17, 8, 20], np.float32(0.23))
        ae(b0[vi(-1, 0, 1),  18, 9, 20], np.float32(0.24))
        ae(b0[vi(-1, 0, -1), 16, 9, 20], np.float32(0.25))

#############################################################################

class PeriodicSimulationTest(LBFluidSim):
    subdomain = BlockTest

    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False
        config.periodic_x = True

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'lat_nx': 128,
            'lat_ny': 64,
            'lat_nz': 66,
            'grid': 'D3Q19',
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'per_horiz_out'),
            'debug_dump_dists': True,
            'cuda_cache': False,
            'check_invalid_results_gpu': False,
            'check_invalid_results_host': False,
        })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, z, y, x
        # vec indices are: x, y, z
        if runner._block.id == 0:
            dbuf[vi(-1, 0, 0), 32, 32, 1] = 0.11
            dbuf[vi(-1, 1, 0), 32, 32, 1] = 0.12
            dbuf[vi(-1, -1, 0), 32, 32, 1] = 0.13
            dbuf[vi(-1, 0, 1), 32, 32, 1] = 0.14
            dbuf[vi(-1, 0, -1), 32, 32, 1] = 0.15
        elif runner._block.id == 1:
            # Corner
            dbuf[vi(1, 0, 0), 1, 1, 64] = 0.21
            dbuf[vi(1, 1, 0), 1, 1, 64] = 0.22
            dbuf[vi(1, 0, 1), 1, 1, 64] = 0.23

            # Edge
            dbuf[vi(1, 0, 0), 1, 32, 64] = 0.33
            dbuf[vi(1, 1, 0), 1, 32, 64] = 0.34
            dbuf[vi(1, -1, 0), 1, 32, 64] = 0.35
            dbuf[vi(1, 0, 1), 1, 32, 64] = 0.36

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class PeriodicPropagationTest(unittest.TestCase):
    def test_horiz_spread(self):
        ctrl = LBSimulationController(PeriodicSimulationTest, TwoBlocksXConnGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'per_horiz_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        b1 = np.load(io.dists_filename(output, 1, 1, 1))

        ae = np.testing.assert_equal

        ae(b1[vi(-1, 0, 0), 32, 32, 64], np.float32(0.11))
        ae(b1[vi(-1, 1, 0), 32, 33, 64], np.float32(0.12))
        ae(b1[vi(-1, -1, 0), 32, 31, 64], np.float32(0.13))
        ae(b1[vi(-1, 0, 1), 33, 32, 64], np.float32(0.14))
        ae(b1[vi(-1, 0, -1), 31, 32, 64], np.float32(0.15))

        ae(b0[vi(1, 0, 0), 1, 1, 1], np.float32(0.21))
        ae(b0[vi(1, 1, 0), 1, 2, 1], np.float32(0.22))
        ae(b0[vi(1, 0, 1), 2, 1, 1], np.float32(0.23))

        ae(b0[vi(1, 0, 0), 1, 32, 1], np.float32(0.33))
        ae(b0[vi(1, 1, 0), 1, 33, 1], np.float32(0.34))
        ae(b0[vi(1,-1, 0), 1, 31, 1], np.float32(0.35))
        ae(b0[vi(1, 0, 1), 2, 32, 1], np.float32(0.36))


class PartialPeriodicSimulationTest(PeriodicSimulationTest):
    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False
        config.periodic_x = False
        config.periodic_y = True
        config.periodic_z = True

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        if runner._block.id == 0:
            dbuf[vi(1, 1, 0), 32, 64, 64] = 0.11
            dbuf[vi(1, 0, 1), 66, 32, 64] = 0.12
            dbuf[vi(1, -1, 0), 32, 1, 64] = 0.13
            dbuf[vi(1, 0, -1), 1, 32, 64] = 0.14
        elif runner._block.id == 1:
            dbuf[vi(-1, 1, 0), 20, 64, 1] = 0.21
            dbuf[vi(-1, 0, 1), 66, 20, 1] = 0.22
            dbuf[vi(-1, -1, 0), 20, 1, 1] = 0.23
            dbuf[vi(-1, 0, -1), 1, 20, 1] = 0.24

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class PartialPeriodicPropagationTest(unittest.TestCase):
    def test_x_conn(self):
        ctrl = LBSimulationController(PartialPeriodicSimulationTest,
                TwoBlocksXConnGeoTest).run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'per_horiz_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        b1 = np.load(io.dists_filename(output, 1, 1, 1))

        ae = np.testing.assert_equal
        ae(b1[vi(1, 1, 0), 32, 1, 1], np.float32(0.11))
        ae(b1[vi(1, 0, 1), 1, 32, 1], np.float32(0.12))
        ae(b1[vi(1, -1, 0), 32, 64, 1], np.float32(0.13))
        ae(b1[vi(1, 0, -1), 66, 32, 1], np.float32(0.14))

        ae(b0[vi(-1, 1, 0), 20, 1, 64], np.float32(0.21))
        ae(b0[vi(-1, 0, 1), 1, 20, 64], np.float32(0.22))
        ae(b0[vi(-1, -1, 0), 20, 64, 64], np.float32(0.23))
        ae(b0[vi(-1, 0, -1), 66, 20, 64], np.float32(0.24))


#############################################################################

class SingleBlockGeoTest(LBGeometry3D):
    def subdomains(self, n=None):
        return [SubdomainSpec3D((0,0,0), (64, 62, 66))]


class SingleBlockPeriodicSimulationTest(LBFluidSim):
    subdomain = BlockTest

    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False
        config.periodic_x = True
        config.periodic_y = True
        config.periodic_z = True

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'lat_nx': 64,
            'lat_ny': 62,
            'lat_nz': 66,
            'grid': 'D3Q19',
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'per_single_out'),
            'debug_dump_dists': True,
            'cuda_cache': False,
            'check_invalid_results_gpu': False,
            'check_invalid_results_host': False,
        })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # X-face
        dbuf[vi(-1, 0, 0), 32, 32, 1] = 0.11
        dbuf[vi(-1, 1, 0), 32, 32, 1] = 0.12
        dbuf[vi(-1, -1, 0), 32, 32, 1] = 0.13
        dbuf[vi(-1, 0, 1), 32, 32, 1] = 0.14
        dbuf[vi(-1, 0, -1), 32, 32, 1] = 0.15

        # Y-face
        dbuf[vi(0, 1, 0), 33, 62, 31] = 0.41

        # Z-face
        dbuf[vi(0, 0, 1), 66, 35, 31] = 0.42

        # Corner
        dbuf[vi(-1, 0, 0), 1, 1, 1] = 0.21
        dbuf[vi(0, -1, 0), 1, 1, 1] = 0.22
        dbuf[vi(0, 0, -1), 1, 1, 1] = 0.23
        dbuf[vi(-1, -1, 0), 1, 1, 1] = 0.24
        dbuf[vi(0, -1, -1), 1, 1, 1] = 0.25
        dbuf[vi(-1, 0, -1), 1, 1, 1] = 0.26

        # Edge
        dbuf[vi(-1, 0, 0), 32, 1, 1] = 0.31
        dbuf[vi(-1, -1, 0), 32, 2, 1] = 0.32

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class SingleBlockPeriodicTest(unittest.TestCase):
    def test_x_face_propagation(self):
        global tmpdir
        ctrl = LBSimulationController(SingleBlockPeriodicSimulationTest,
                SingleBlockGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'per_single_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        ae = np.testing.assert_equal

        ae(b0[vi(-1, 0, 0), 32, 32, 64], np.float32(0.11))
        ae(b0[vi(-1, 1, 0), 32, 33, 64], np.float32(0.12))
        ae(b0[vi(-1, -1, 0), 32, 31, 64], np.float32(0.13))
        ae(b0[vi(-1, 0, 1), 33, 32, 64], np.float32(0.14))
        ae(b0[vi(-1, 0, -1), 31, 32, 64], np.float32(0.15))

        ae(b0[vi(0, 1, 0), 33, 1, 31], np.float32(0.41))
        ae(b0[vi(0, 0, 1), 1, 35, 31], np.float32(0.42))

        ae(b0[vi(-1, 0, 0), 1, 1, 64], np.float32(0.21))
        ae(b0[vi(0, -1, 0), 1, 62, 1], np.float32(0.22))
        ae(b0[vi(0, 0, -1), 66, 1, 1], np.float32(0.23))

        ae(b0[vi(-1, -1, 0), 1, 62, 64], np.float32(0.24))
        ae(b0[vi(0, -1, -1), 66, 62, 1], np.float32(0.25))
        ae(b0[vi(-1, 0, -1), 66, 1, 64], np.float32(0.26))

        ae(b0[vi(-1, 0, 0), 32, 1, 64], np.float32(0.31))
        ae(b0[vi(-1, -1, 0), 32, 1, 64], np.float32(0.32))


class SingleBlockPeriodicSimulationQ15Test(SingleBlockPeriodicSimulationTest):
    @classmethod
    def update_defaults(cls, defaults):
        SingleBlockPeriodicSimulationTest.update_defaults(defaults)
        defaults.update({
            'grid': 'D3Q15',
            })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # X-face, left, middle
        dbuf[vi2(-1, 0, 0), 32, 32, 1] = 0.11
        dbuf[vi2(-1, -1, -1), 32, 32, 1] = 0.12
        dbuf[vi2(-1, -1, 1), 32, 32, 1] = 0.13
        dbuf[vi2(-1, 1, -1), 32, 32, 1] = 0.14
        dbuf[vi2(-1, 1, 1), 32, 32, 1] = 0.15

        # Y-face, low, middle
        dbuf[vi2(0, -1, 0), 32, 1, 32] = 0.21
        dbuf[vi2(1, -1, 1), 32, 1, 32] = 0.22
        dbuf[vi2(-1, -1, 1), 32, 1, 32] = 0.23
        dbuf[vi2(-1, -1, -1), 32, 1, 32] = 0.24
        dbuf[vi2(1, -1, -1), 32, 1, 32] = 0.25

        # X-face, left, edge
        dbuf[vi2(-1, 0, 0), 32, 1, 1] = 0.16
        dbuf[vi2(-1, 0, 0), 1, 1, 1] = 0.17
        dbuf[vi2(-1, 0, 0), 1, 32, 1] = 0.18
        dbuf[vi2(-1, 0, 0), 66, 1, 1] = 0.19
        dbuf[vi2(-1, 0, 0), 1, 62, 1] = 0.20

        # corners
        dbuf[vi2(-1, -1, -1), 1, 1, 1] = 0.30
        dbuf[vi2(-1, -1, 1), 66, 1, 1] = 0.31
        dbuf[vi2(-1, 1, -1), 1, 62, 1] = 0.32
        dbuf[vi2(-1, 1, 1), 66, 62, 1] = 0.33

        dbuf[vi2(-1, 1, 1), 65, 2, 1] = 0.40
        dbuf[vi2(-1, 1, 1), 2, 61, 1] = 0.41
        dbuf[vi2(-1, 1, 1), 66, 2, 1] = 0.42
        dbuf[vi2(-1, 1, 1), 2, 62, 1] = 0.43

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class SingleBlockPeriodicQ15Test(unittest.TestCase):
    def test_x_face_propagation(self):
        global tmpdir
        ctrl = LBSimulationController(SingleBlockPeriodicSimulationQ15Test,
                SingleBlockGeoTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'per_single_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))
        ae = np.testing.assert_equal

        ae(b0[vi2(-1, 0, 0), 32, 32, 64], np.float32(0.11))
        ae(b0[vi2(-1, -1, -1), 31, 31, 64], np.float32(0.12))
        ae(b0[vi2(-1, -1, 1), 33, 31, 64], np.float32(0.13))
        ae(b0[vi2(-1, 1, -1), 31, 33, 64], np.float32(0.14))
        ae(b0[vi2(-1, 1, 1), 33, 33, 64], np.float32(0.15))

        ae(b0[vi2(0, -1, 0), 32, 62, 32], np.float32(0.21))
        ae(b0[vi2(1, -1, 1), 33, 62, 33], np.float32(0.22))
        ae(b0[vi2(-1, -1, 1), 33, 62, 31], np.float32(0.23))
        ae(b0[vi2(-1, -1, -1), 31, 62, 31], np.float32(0.24))
        ae(b0[vi2(1, -1, -1), 31, 62, 33], np.float32(0.25))

        ae(b0[vi2(-1, 0, 0), 32, 1, 64], np.float32(0.16))
        ae(b0[vi2(-1, 0, 0), 1, 1, 64], np.float32(0.17))
        ae(b0[vi2(-1, 0, 0), 1, 32, 64], np.float32(0.18))
        ae(b0[vi2(-1, 0, 0), 66, 1, 64], np.float32(0.19))
        ae(b0[vi2(-1, 0, 0), 1, 62, 64], np.float32(0.20))

        ae(b0[vi2(-1, -1, -1), 66, 62, 64], np.float32(0.30))
        ae(b0[vi2(-1, -1, 1), 1, 62, 64], np.float32(0.31))
        ae(b0[vi2(-1, 1, -1), 66, 1, 64], np.float32(0.32))
        ae(b0[vi2(-1, 1, 1), 1, 1, 64], np.float32(0.33))

        ae(b0[vi2(-1, 1, 1), 66, 3, 64], np.float32(0.40))
        ae(b0[vi2(-1, 1, 1), 3, 62, 64], np.float32(0.41))
        ae(b0[vi2(-1, 1, 1), 1, 3, 64], np.float32(0.42))
        ae(b0[vi2(-1, 1, 1), 3, 1, 64], np.float32(0.43))


def setUpModule():
    global tmpdir
    tmpdir = tempfile.mkdtemp()

def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    if block_size < mem_align:
        mem_align = block_size
    unittest.main()

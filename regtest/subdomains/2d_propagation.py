#!/usr/bin/env python

import os
import shutil
import tempfile
import unittest

import numpy as np

from sailfish import io
from sailfish.geo import LBGeometry2D
from sailfish.subdomain import SubdomainSpec2D, Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.sym import D2Q9
from regtest.subdomains import util

#
#  b1  b3
#
#  b0  b2
#
class GeometryTest(LBGeometry2D):
    def subdomains(self, n=None):
        blocks = []
        q = 128
        for i in range(0, 2):
            for j in range(0, 2):
                blocks.append(SubdomainSpec2D((i * q, j * q), (q, q)))

        return blocks

class DoubleBlockGeometryTest(LBGeometry2D):
    def subdomains(self, n=None):
        blocks = []
        q = 128
        blocks.append(SubdomainSpec2D((0, 0), (q, 2*q)))
        blocks.append(SubdomainSpec2D((q, 0), (q, 2*q)))
        return blocks

class Vertical2BlockGeo(LBGeometry2D):
    def subdomains(self, n=None):
        q = 128
        return [SubdomainSpec2D((0, 0), (q, q)),
                SubdomainSpec2D((0, q), (q, q))]

class ThreeBlocksGeometryTest(LBGeometry2D):
    def subdomains(self, n=None):
        blocks = []
        q = 128

        # +-------+
        # | 1 |   |
        # |---| 2 |
        # | 0 |   |
        # +-------+

        blocks.append(SubdomainSpec2D((0, 0), (q, q)))
        blocks.append(SubdomainSpec2D((0, q), (q, q)))
        blocks.append(SubdomainSpec2D((q, 0), (q, 2*q)))
        return blocks

class BlockTest(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        pass

mem_align = 32
block_size = 64
tmpdir = None
periodic_x = False
periodic_y = False
vi = lambda x, y: D2Q9.vec_idx([x, y])

class SimulationTest(LBFluidSim):
    subdomain = BlockTest

    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False
        config.periodic_x = periodic_x
        config.periodic_y = periodic_y

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'lat_nx': 256,
            'lat_ny': 256,
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'test_out'),
            'debug_dump_dists': True,
            'cuda_cache': False,
            'check_invalid_results_gpu': False,
            'check_invalid_results_host': False,
        })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, y, x
        # vec indices are: x, y
        if runner._spec.id == 0:
            dbuf[vi(1, 1), 128, 128] = 0.11
        elif runner._spec.id == 1:
            dbuf[vi(1, -1), 1, 128] = 0.22
        elif runner._spec.id == 2:
            dbuf[vi(-1, 1), 128, 1] = 0.33
        else:
            dbuf[vi(-1, -1), 1, 1] = 0.44

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class MixedPeriodicPropagationTest(unittest.TestCase):
    def setUp(self):
        global periodic_x, periodic_y
        periodic_x = False
        periodic_y = True

    def test_horiz_spread(self):
        """Two blocks connected along the X axis, with Y PBC enabled.

        This test verifies distribution streaming from the corner nodes
        on the surface connecting the two blocks."""
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._spec.id == 1:
                # At the top
                dbuf[vi(-1, 0), 256, 1] = 0.31
                dbuf[vi(-1, 1), 256, 1] = 0.32
                dbuf[vi(-1, -1), 256, 1] = 0.33
            elif runner._spec.id == 0:
                # At the bottom
                dbuf[vi(1, 0), 1, 128] = 0.41
                dbuf[vi(1, 1), 1, 128] = 0.42
                dbuf[vi(1, -1), 1, 128] = 0.43

            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        HorizTest = type('HorizTest', (SimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(HorizTest, DoubleBlockGeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        ae = np.testing.assert_equal

        ae(b0[vi(-1, 0), 256, 128], np.float32(0.31))
        ae(b0[vi(-1, -1), 255, 128], np.float32(0.33))
        ae(b0[vi(-1, 1), 1, 128], np.float32(0.32))

        ae(b1[vi(1, 0), 1, 1], np.float32(0.41))
        ae(b1[vi(1, 1), 2, 1], np.float32(0.42))
        ae(b1[vi(1, -1), 256, 1], np.float32(0.43))

class AASimulationTest(SimulationTest):
    subdomain = BlockTest

    @classmethod
    def update_defaults(cls, defaults):
        SimulationTest.update_defaults(defaults)
        defaults.update({
            'access_pattern': 'AA',
            })

class ABPropagationTest(unittest.TestCase):
    def test_horiz_spread(self):
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            # In-warp location x = 20.
            dbuf[vi(1, 0), 30, 20] = 0.11
            dbuf[vi(1, 1), 30, 20] = 0.12
            dbuf[vi(1, -1), 30, 20] = 0.13

            # End of warp location x = 31.
            dbuf[vi(1, 0), 30, 31] = 0.14

            # Beginning of warp location x = 32.
            dbuf[vi(1, 0), 30, 32] = 0.16

            # End of block location x = 64.
            dbuf[vi(1, 0), 30, 63] = 0.15

            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        HorizTest = type('HorizTest', (SimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(HorizTest, LBGeometry2D, {'block_size': 64})
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        ae = np.testing.assert_equal

        ae(b0[vi(1, 0), 30, 21], np.float32(0.11))
        ae(b0[vi(1, 1), 31, 21], np.float32(0.12))
        ae(b0[vi(1, -1), 29, 21], np.float32(0.13))
        ae(b0[vi(1, 0), 30, 32], np.float32(0.14))
        ae(b0[vi(1, 0), 30, 64], np.float32(0.15))
        ae(b0[vi(1, 0), 30, 33], np.float32(0.16))

class AAPropagationTest(unittest.TestCase):
    def test_horiz_spread(self):
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._spec.id == 0:
                dbuf[vi(1, 0), 64, 128] = 0.11
                dbuf[vi(1, 1), 64, 128] = 0.12
                dbuf[vi(1, -1), 64, 128] = 0.13

            runner._debug_set_dist(dbuf)

        HorizTest = type('HorizTest', (AASimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(HorizTest, DoubleBlockGeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        ae = np.testing.assert_equal

        # No propagation in the first step, but the distributions are stored
        # in opposite slots.
        ae(b1[vi(-1, 0), 64, 0], np.float32(0.11))
        ae(b1[vi(-1, -1), 64, 0], np.float32(0.12))
        ae(b1[vi(-1, 1), 64, 0], np.float32(0.13))

    def test_vert_spread(self):
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._spec.id == 0:
                dbuf[vi(-1, 1), 128, 64] = 0.11
                dbuf[vi(0, 1), 128, 64] = 0.12
                dbuf[vi(1, 1), 128, 64] = 0.13

            runner._debug_set_dist(dbuf)

        VertTest = type('VertTest', (AASimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(VertTest, Vertical2BlockGeo)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        ae = np.testing.assert_equal

        # No propagation in the first step, but the distributions are stored
        # in opposite slots.
        ae(b1[vi(1, -1), 0, 64], np.float32(0.11))
        ae(b1[vi(0, -1), 0, 64], np.float32(0.12))
        ae(b1[vi(-1, -1), 0, 64], np.float32(0.13))

    def test_3blocks(self):
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._spec.id == 0:
                dbuf[vi(1, 1), 128, 128] = 0.11
                dbuf[vi(0, 1), 128, 128] = 0.12
                dbuf[vi(1, 0), 128, 128] = 0.13
                dbuf[vi(1, -1), 128, 128] = 0.14
                dbuf[vi(-1, 1), 128, 128] = 0.15
            elif runner._spec.id == 1:
                dbuf[vi(-1, -1), 1, 128] = 0.16
                dbuf[vi(0, -1), 1, 128] = 0.17
            elif runner._spec.id == 2:
                dbuf[vi(-1, -1), 129, 1] = 0.18
                dbuf[vi(-1, -1), 128, 1] = 0.19
                dbuf[vi(-1, 0), 128, 1] = 0.20

            runner._debug_set_dist(dbuf)

        SimTest = type('SimTest', (AASimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(SimTest, ThreeBlocksGeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        b2 = np.load(io.dists_filename(output, 1, 2, 1))['arr_0']
        ae = np.testing.assert_equal

        # No propagation in the first step, but the distributions are stored
        # in opposite slots.
        ae(b2[vi(-1, -1), 128, 0], np.float32(0.11))
        ae(b2[vi(-1, 0), 128, 0], np.float32(0.13))
        ae(b2[vi(-1, 1), 128, 0], np.float32(0.14))
        ae(b1[vi(1, -1), 0, 128], np.float32(0.15))
        ae(b1[vi(0, -1), 0, 128], np.float32(0.12))

        # From b1
        ae(b0[vi(1, 1), 129, 128], np.float32(0.16))
        ae(b0[vi(0, 1), 129, 128], np.float32(0.17))

        # From b2
        ae(b0[vi(1, 1), 128, 129], np.float32(0.19))
        ae(b0[vi(1, 1), 129, 129], np.float32(0.18))
        ae(b0[vi(1, 0), 128, 129], np.float32(0.20))


class PeriodicCornerPropagationTest(unittest.TestCase):
    def setUp(self):
        global periodic_x, periodic_y
        periodic_x = True
        periodic_y = True

    def test_spread(self):
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._spec.id == 1:
                dbuf[vi(1, 0), 1, 128] = 0.11
                dbuf[vi(1, 1), 1, 128] = 0.12
                dbuf[vi(1, -1), 1, 128] = 0.13

                # At the top
                dbuf[vi(1, 0), 256, 128] = 0.31
                dbuf[vi(1, 1), 256, 128] = 0.32
                dbuf[vi(1, -1), 256, 128] = 0.33

            elif runner._spec.id == 0:
                dbuf[vi(-1, 0), 256, 1] = 0.21
                dbuf[vi(-1, 1), 256, 1] = 0.22
                dbuf[vi(-1, -1), 256, 1] = 0.23

                # At the bottom
                dbuf[vi(-1, 0), 1, 1] = 0.41
                dbuf[vi(-1, 1), 1, 1] = 0.42
                dbuf[vi(-1, -1), 1, 1] = 0.43

            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        HorizTest = type('HorizTest', (SimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(HorizTest, DoubleBlockGeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        ae = np.testing.assert_equal

        ae(b0[vi(1, 0), 1, 1], np.float32(0.11))
        ae(b0[vi(1, 1), 2, 1], np.float32(0.12))
        ae(b0[vi(1, -1), 256, 1], np.float32(0.13))

        ae(b0[vi(1, 0), 256, 1], np.float32(0.31))
        ae(b0[vi(1, 1), 1, 1], np.float32(0.32))
        ae(b0[vi(1, -1), 255, 1], np.float32(0.33))

        ae(b1[vi(-1, 0), 256, 128], np.float32(0.21))
        ae(b1[vi(-1, 1), 1, 128], np.float32(0.22))
        ae(b1[vi(-1, -1), 255, 128], np.float32(0.23))

        ae(b1[vi(-1, 0), 1, 128], np.float32(0.41))
        ae(b1[vi(-1, 1), 2, 128], np.float32(0.42))
        ae(b1[vi(-1, -1), 256, 128], np.float32(0.43))

class PeriodicPropagationTest(unittest.TestCase):
    def setUp(self):
        global periodic_x, periodic_y
        periodic_x = True
        periodic_y = False

    def test_horiz_spread(self):

        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._spec.id == 1:
                dbuf[vi(1, 0), 128, 128] = 0.11
                dbuf[vi(1, 1), 128, 128] = 0.12
                dbuf[vi(1, -1), 128, 128] = 0.13

                # At the top
                dbuf[vi(1, 0), 256, 128] = 0.31
                dbuf[vi(1, 1), 256, 128] = 0.32
                dbuf[vi(1, -1), 256, 128] = 0.33

                dbuf[vi(-1, -1), 256, 128] = 0.66   # should not be overwritten
            elif runner._spec.id == 0:
                dbuf[vi(-1, 0), 128, 1] = 0.21
                dbuf[vi(-1, 1), 128, 1] = 0.22
                dbuf[vi(-1, -1), 128, 1] = 0.23

                # At the bottom
                dbuf[vi(-1, 0), 1, 1] = 0.41
                dbuf[vi(-1, 1), 1, 1] = 0.42
                dbuf[vi(-1, -1), 1, 1] = 0.43

                dbuf[vi(1, 1), 1, 1] = 0.77     # should not be overwritten

            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        HorizTest = type('HorizTest', (SimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(HorizTest, DoubleBlockGeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']

        ae = np.testing.assert_equal

        ae(b0[vi(1, 0), 128, 1], np.float32(0.11))
        ae(b0[vi(1, 1), 129, 1], np.float32(0.12))
        ae(b0[vi(1, -1), 127, 1], np.float32(0.13))

        ae(b0[vi(1, 0), 256, 1], np.float32(0.31))
        ae(b0[vi(1, -1), 255, 1], np.float32(0.33))

        ae(b1[vi(-1, 0), 128, 128], np.float32(0.21))
        ae(b1[vi(-1, 1), 129, 128], np.float32(0.22))
        ae(b1[vi(-1, -1), 127, 128], np.float32(0.23))

        ae(b1[vi(-1, 0), 1, 128], np.float32(0.41))
        ae(b1[vi(-1, 1), 2, 128], np.float32(0.42))

        ae(b1[vi(-1, -1), 256, 128], np.float32(0.66))
        ae(b0[vi(1, 1), 1, 1], np.float32(0.77))

    # Like the test above but for a single block that is globally periodic.
    def test_horiz_global_periodic(self):
        global tmpdir

        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            dbuf[vi(1, 0), 128, 256] = 0.11
            dbuf[vi(1, 1), 128, 256] = 0.12
            dbuf[vi(1, -1), 128, 256] = 0.13

            # At the top
            dbuf[vi(1, 0), 256, 256] = 0.31
            dbuf[vi(1, 1), 256, 256] = 0.32
            dbuf[vi(1, -1), 256, 256] = 0.33

            dbuf[vi(-1, -1), 256, 256] = 0.66   # should not be overwritten
            dbuf[vi(-1, 0), 128, 1] = 0.21
            dbuf[vi(-1, 1), 128, 1] = 0.22
            dbuf[vi(-1, -1), 128, 1] = 0.23

            # At the bottom
            dbuf[vi(-1, 0), 1, 1] = 0.41
            dbuf[vi(-1, 1), 1, 1] = 0.42
            dbuf[vi(-1, -1), 1, 1] = 0.43

            dbuf[vi(1, 1), 1, 1] = 0.77     # should not be overwritten

            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        HorizTest = type('HorizTest', (SimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(HorizTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        ae = np.testing.assert_equal

        ae(b0[vi(1, 0), 128, 1], np.float32(0.11))
        ae(b0[vi(1, 1), 129, 1], np.float32(0.12))
        ae(b0[vi(1, -1), 127, 1], np.float32(0.13))
        ae(b0[vi(1, 0), 256, 1], np.float32(0.31))
        ae(b0[vi(1, -1), 255, 1], np.float32(0.33))

        ae(b0[vi(-1, 0), 128, 256], np.float32(0.21))
        ae(b0[vi(-1, 1), 129, 256], np.float32(0.22))
        ae(b0[vi(-1, -1), 127, 256], np.float32(0.23))
        ae(b0[vi(-1, 0), 1, 256], np.float32(0.41))
        ae(b0[vi(-1, 1), 2, 256], np.float32(0.42))

        ae(b0[vi(1, 1), 1, 1], np.float32(0.77))
        ae(b0[vi(-1, -1), 256, 256], np.float32(0.66))

    # Single block, periodic in both X and Y.
    def test_corner_global_periodic(self):
        global tmpdir, periodic_y
        periodic_y = True

        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0
            dbuf[vi(1, 1), 256, 256] = 0.11
            dbuf[vi(-1, -1), 1, 1] = 0.12
            dbuf[vi(1, -1), 1, 256] = 0.13
            dbuf[vi(-1, 1), 256, 1] = 0.14
            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        CornerTest = type('CornerTest', (SimulationTest,),
                {'initial_conditions': ic})
        ctrl = LBSimulationController(CornerTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        ae = np.testing.assert_equal

        ae(b0[vi(1, 1), 1, 1], np.float32(0.11))
        ae(b0[vi(-1, -1), 256, 256], np.float32(0.12))
        ae(b0[vi(1, -1), 256, 1], np.float32(0.13))
        ae(b0[vi(-1, 1), 1, 256], np.float32(0.14))


class TestCornerPropagation(unittest.TestCase):
    """Tests mass fraction movements between the corners of 4 subdomains."""
    def setUp(self):
        global periodic_x, periodic_y
        periodic_x = False
        periodic_y = False

    def test_4corners(self):
        global tmpdir
        ctrl = LBSimulationController(SimulationTest, GeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        b2 = np.load(io.dists_filename(output, 1, 2, 1))['arr_0']
        b3 = np.load(io.dists_filename(output, 1, 3, 1))['arr_0']

        np.testing.assert_equal(b0[vi(-1, -1), 128, 128], np.float32(0.44))
        np.testing.assert_equal(b1[vi(-1, 1), 1, 128], np.float32(0.33))
        np.testing.assert_equal(b2[vi(1, -1), 128, 1], np.float32(0.22))
        np.testing.assert_equal(b3[vi(1, 1), 1, 1], np.float32(0.11))

    def test_b0_spread(self):
        global tmpdir
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._spec.id == 0:
                # Top right corner
                dbuf[vi(1, 1), 128, 128] = 0.11
                dbuf[vi(0, 1), 128, 128] = 0.01
                dbuf[vi(1, 0), 128, 128] = 0.10
                dbuf[vi(0, -1), 128, 128] = 0.02
                dbuf[vi(-1, 0), 128, 128] = 0.20
                dbuf[vi(1, -1), 128, 128] = 0.30
                dbuf[vi(-1, 1), 128, 128] = 0.40

                # Bottom right corner
                dbuf[vi(1, 1), 1, 128] = 0.50
                dbuf[vi(1, -1), 1, 128] = 0.51
                dbuf[vi(1, 0), 1, 128] = 0.52
            elif runner._spec.id == 1:
                dbuf[vi(1, 0), 127, 128] = 0.60
                dbuf[vi(1, 1), 127, 128] = 0.61
                dbuf[vi(1, -1), 127, 128] = 0.62
                dbuf[vi(1 ,0), 128, 128] = 0.70
                dbuf[vi(1, -1), 128, 128] = 0.71

            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        RightSide = type('RightSide', (SimulationTest,), {'initial_conditions': ic})
        ctrl = LBSimulationController(RightSide, GeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        b2 = np.load(io.dists_filename(output, 1, 2, 1))['arr_0']
        b3 = np.load(io.dists_filename(output, 1, 3, 1))['arr_0']
        ae = np.testing.assert_equal

        ae(b3[vi(1, 1), 1, 1], np.float32(0.11))
        ae(b0[vi(-1, 0), 128, 127], np.float32(0.20))
        ae(b0[vi(0, -1), 127, 128], np.float32(0.02))
        ae(b2[vi(1, 0), 128, 1], np.float32(0.10))
        ae(b1[vi(0, 1), 1, 128], np.float32(0.01))
        ae(b2[vi(1, -1), 127, 1], np.float32(0.30))
        ae(b1[vi(-1, 1), 1, 127], np.float32(0.40))
        ae(b2[vi(1, 1), 2, 1], np.float32(0.50))
        ae(b2[vi(1, 0), 1, 1], np.float32(0.52))

        ae(b3[vi(1, 0), 127, 1], np.float32(0.60))
        ae(b3[vi(1, 0), 128, 1], np.float32(0.70))
        ae(b3[vi(1, 1), 128, 1], np.float32(0.61))
        ae(b3[vi(1, -1), 127, 1], np.float32(0.71))
        ae(b3[vi(1, -1), 126, 1], np.float32(0.62))


class ThreeBlocksSimulationTest(LBFluidSim):
    subdomain = BlockTest

    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'lat_nx': 256,
            'lat_ny': 256,
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'test_out'),
            'debug_dump_dists': True,
            'cuda_cache': False,
            'check_invalid_results_gpu': False,
            'check_invalid_results_host': False,
        })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, y, x
        # vec indices are: x, y
        if runner._spec.id == 0:
            dbuf[vi(1, 1), 128, 128] = 0.11
            dbuf[vi(1, 0), 64, 128] = 0.12
            dbuf[vi(0, 1), 128, 64] = 0.13
        elif runner._spec.id == 1:
            dbuf[vi(1, -1), 1, 128] = 0.22
            dbuf[vi(0, -1), 1, 64] = 0.21
            dbuf[vi(1, 0), 64, 128] = 0.23
        elif runner._spec.id == 2:
            dbuf[vi(-1, 0), 64, 1] = 0.31

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)


class TestThreeBlockPropagation(unittest.TestCase):
    def test_propagation(self):
        global tmpdir
        ctrl = LBSimulationController(ThreeBlocksSimulationTest, ThreeBlocksGeometryTest)
        ctrl.run(ignore_cmdline=True)

        output = os.path.join(tmpdir, 'test_out')
        b0 = np.load(io.dists_filename(output, 1, 0, 1))['arr_0']
        b1 = np.load(io.dists_filename(output, 1, 1, 1))['arr_0']
        b2 = np.load(io.dists_filename(output, 1, 2, 1))['arr_0']

        ae = np.testing.assert_equal

        ae(b0[vi(0, -1), 128, 64], np.float32(0.21))
        ae(b1[vi(0, 1),  1, 64], np.float32(0.13))
        ae(b2[vi(1, 0), 64, 1], np.float32(0.12))
        ae(b2[vi(1, 0), 64+128, 1], np.float32(0.23))
        ae(b0[vi(-1, 0), 64, 128], np.float32(0.31))
        ae(b2[vi(1, 1), 129, 1], np.float32(0.11))
        ae(b2[vi(1, -1), 128, 1], np.float32(0.22))


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

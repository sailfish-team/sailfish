#!/usr/bin/python

import os
import tempfile
import unittest

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D, GeoBlock2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim
from sailfish.sym import D2Q9

#
#  b1  b3
#
#  b0  b2
#
class GeometryTest(LBGeometry2D):
    def blocks(self, n=None):
        blocks = []
        q = 128
        for i in range(0, 2):
            for j in range(0, 2):
                blocks.append(LBBlock2D((i * q, j * q), (q, q)))

        return blocks

class DoubleBlockGeometryTest(LBGeometry2D):
    def blocks(self, n=None):
        blocks = []
        q = 128
        blocks.append(LBBlock2D((0, 0), (q, 2*q)))
        blocks.append(LBBlock2D((q, 0), (q, 2*q)))
        return blocks

class ThreeBlocksGeometryTest(LBGeometry2D):
    def blocks(self, n=None):
        blocks = []
        q = 128
        blocks.append(LBBlock2D((0, 0), (q, q)))
        blocks.append(LBBlock2D((0, q), (q, q)))
        blocks.append(LBBlock2D((q, 0), (q, 2*q)))
        return blocks

class BlockTest(GeoBlock2D):
    def _define_nodes(self, hx, hy):
        pass

    def _init_fields(self, sim, hx, hy):
        pass

tmpdir = None
periodic_x = False
vi = lambda x, y: D2Q9.vec_idx([x, y])

class SimulationTest(LBFluidSim, LBForcedSim):
    geo = BlockTest

    @classmethod
    def modify_config(cls, config):
        global periodic_x
        config.relaxation_enabled = False
        config.periodic_x = periodic_x

    @classmethod
    def update_defaults(cls, defaults):
        global tmpdir
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'test_out'),
            'debug_dump_dists': True,
        })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, y, x
        # vec indices are: x, y
        if runner._block.id == 0:
            dbuf[vi(1, 1), 128, 128] = 0.11
        elif runner._block.id == 1:
            dbuf[vi(1, -1), 1, 128] = 0.22
        elif runner._block.id == 2:
            dbuf[vi(-1, 1), 128, 1] = 0.33
        else:
            dbuf[vi(-1, -1), 1, 1] = 0.44

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)

class PeriodicPropagationTest(object): #unittest.TestCase):
    def setUp(self):
        global periodic_x
        periodic_x = True

    def test_horiz_spread(self):
        global tmpdir

        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._block.id == 1:
                dbuf[vi(1, 0), 128, 128] = 0.11
                dbuf[vi(1, 1), 128, 128] = 0.12
                dbuf[vi(1, -1), 128, 128] = 0.13

                # At the top
                dbuf[vi(1, 0), 256, 128] = 0.31
                dbuf[vi(1, 1), 256, 128] = 0.32
                dbuf[vi(1, -1), 256, 128] = 0.33

                dbuf[vi(-1, -1), 256, 128] = 0.66   # should not be overwritten
            elif runner._block.id == 0:
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
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
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
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
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


class TestCornerPropagation(object): #unittest.TestCase):
    def setUp(self):
        global periodic_x
        periodic_x = False

    def test_4corners(self):
        global tmpdir
        ctrl = LBSimulationController(SimulationTest, GeometryTest)
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
        b2 = np.load(os.path.join(tmpdir, 'test_out_blk2_dist_dump1.npy'))
        b3 = np.load(os.path.join(tmpdir, 'test_out_blk3_dist_dump1.npy'))

        np.testing.assert_equal(b0[vi(-1, -1), 128, 128], np.float32(0.44))
        np.testing.assert_equal(b1[vi(-1, 1), 1, 128], np.float32(0.33))
        np.testing.assert_equal(b2[vi(1, -1), 128, 1], np.float32(0.22))
        np.testing.assert_equal(b3[vi(1, 1), 1, 1], np.float32(0.11))

    def test_b0_spread(self):
        global tmpdir
        def ic(self, runner):
            dbuf = runner._debug_get_dist()
            dbuf[:] = 0.0

            if runner._block.id == 0:
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
            elif runner._block.id == 1:
                dbuf[vi(1, 0), 127, 128] = 0.60
                dbuf[vi(1, 1), 127, 128] = 0.61
                dbuf[vi(1, -1), 127, 128] = 0.62
                dbuf[vi(1 ,0), 128, 128] = 0.70
                dbuf[vi(1, -1), 128, 128] = 0.71

            runner._debug_set_dist(dbuf)
            runner._debug_set_dist(dbuf, False)

        RightSide = type('RightSide', (SimulationTest,), {'initial_conditions': ic})
        ctrl = LBSimulationController(RightSide, GeometryTest)
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
        b2 = np.load(os.path.join(tmpdir, 'test_out_blk2_dist_dump1.npy'))
        b3 = np.load(os.path.join(tmpdir, 'test_out_blk3_dist_dump1.npy'))

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


class ThreeBlocksSimulationTest(LBFluidSim, LBForcedSim):
    geo = BlockTest

    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False

    @classmethod
    def update_defaults(cls, defaults):
        global tmpdir
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'test_out'),
            'debug_dump_dists': True,
        })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, y, x
        # vec indices are: x, y
        if runner._block.id == 0:
            dbuf[vi(1, 1), 128, 128] = 0.11
            dbuf[vi(1, 0), 64, 128] = 0.12
            dbuf[vi(0, 1), 128, 64] = 0.13
        elif runner._block.id == 1:
            dbuf[vi(1, -1), 1, 128] = 0.22
            dbuf[vi(0, -1), 1, 64] = 0.21
            dbuf[vi(1, 0), 64, 128] = 0.23
        elif runner._block.id == 2:
            dbuf[vi(-1, 0), 64, 1] = 0.31

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)


class TestThreeBlockPropagation(unittest.TestCase):
    def test_propagation(self):
        global tmpdir
        ctrl = LBSimulationController(ThreeBlocksSimulationTest, ThreeBlocksGeometryTest)
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
        b2 = np.load(os.path.join(tmpdir, 'test_out_blk2_dist_dump1.npy'))

        ae = np.testing.assert_equal

        ae(b0[vi(0, -1), 128, 64], np.float32(0.21))
        ae(b1[vi(0, 1),  1, 64], np.float32(0.13))
        ae(b2[vi(1, 0), 64, 1], np.float32(0.12))
        ae(b2[vi(1, 0), 64+128, 1], np.float32(0.23))
        ae(b0[vi(-1, 0), 64, 128], np.float32(0.31))


if __name__ == '__main__':
    tmpdir = tempfile.mkdtemp()
    unittest.main()

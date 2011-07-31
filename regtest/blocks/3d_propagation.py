#!/usr/bin/env python

import os
import tempfile
import unittest

import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.geo_block import LBBlock3D, GeoBlock3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim
from sailfish.sym import D3Q19


class BlockTest(GeoBlock3D):
    def _define_nodes(self, hx, hy, hz):
        pass

    def _init_fields(self, sim, hx, hy, hz):
        pass

class TwoBlocksXConnGeoTest(LBGeometry3D):
    def blocks(self, n=None):
        blocks = []
        blocks.append(LBBlock3D((0, 0, 0), (64, 64, 66)))
        blocks.append(LBBlock3D((64, 0, 0), (64, 64, 66)))
        return blocks

class TwoBlocksYConnGeoTest(LBGeometry3D):
    def blocks(self, n=None):
        blocks = []
        blocks.append(LBBlock3D((0, 0, 0), (64, 64, 66)))
        blocks.append(LBBlock3D((0, 64, 0), (64, 64, 66)))
        return blocks

class TwoBlocksZConnGeoTest(LBGeometry3D):
    def blocks(self, n=None):
        blocks = []
        blocks.append(LBBlock3D((0, 0, 0), (64, 66, 64)))
        blocks.append(LBBlock3D((0, 0, 64), (64, 66, 64)))
        return blocks


tmpdir = None
periodic_x = False
vi = lambda x, y, z: D3Q19.vec_idx([x, y, z])

class SimulationTest(LBFluidSim, LBForcedSim):
    geo = BlockTest

    @classmethod
    def modify_config(cls, config):
        global periodic_x
        config.relaxation_enabled = False
        config.periodic_x = periodic_x

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

        global tmpdir
        defaults.update({
            'lat_nx': lat_nx,
            'lat_ny': lat_ny,
            'lat_nz': lat_nz,
            'grid': 'D3Q19',
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'test_out'),
            'debug_dump_dists': True,
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
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
        self._verify(b0, b1, HorizTest)

    def test_vert_spread(self):
        global tmpdir
        VertTest = type('VertTest', (SimulationTest,), {'axis': 1})
        ctrl = LBSimulationController(VertTest, TwoBlocksYConnGeoTest)
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
        self._verify(b0, b1, VertTest)

    def test_depth_spread(self):
        global tmpdir
        DepthTest = type('DepthTest', (SimulationTest,), {'axis': 2})
        ctrl = LBSimulationController(DepthTest, TwoBlocksZConnGeoTest)
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
        self._verify(b0, b1, DepthTest)

if __name__ == '__main__':
    tmpdir = tempfile.mkdtemp()
    unittest.main()

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

class TwoBlocksGeometryTest(LBGeometry3D):
    def blocks(self, n=None):
        blocks = []
        q = 64
        blocks.append(LBBlock3D((0, 0, 0), (q, q, q)))
        blocks.append(LBBlock3D((q, 0, 0), (q, q, q)))
        return blocks

class BlockTest(GeoBlock3D):
    def _define_nodes(self, hx, hy, hz):
        pass

    def _init_fields(self, sim, hx, hy, hz):
        pass

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
        global tmpdir
        defaults.update({
            'lat_nx': 128,
            'lat_ny': 64,
            'lat_nz': 64,
            'grid': 'D3Q19',
            'max_iters': 2,
            'every': 1,
            'quiet': True,
            'output': os.path.join(tmpdir, 'test_out'),
            'debug_dump_dists': True,
        })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        # dbuf indices are: dist, z, y, x
        # vec indices are: x, y, z
        if runner._block.id == 0:
            dbuf[vi(1, 0, 0), 32, 32, 64] = 0.11
            dbuf[vi(1, 1, 0), 32, 32, 64] = 0.12
            dbuf[vi(1, -1, 0), 32, 32, 64] = 0.13
            dbuf[vi(1, 0, 1), 32, 32, 64] = 0.21
            dbuf[vi(1, 0, -1), 32, 32, 64] = 0.23
        elif runner._block.id == 1:
            dbuf[vi(-1, 0, 0), 1, 1, 1] = 0.33
            dbuf[vi(-1, 1, 0), 1, 1, 1] = 0.34
            dbuf[vi(-1, 0, 1), 1, 1, 1] = 0.35

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)


class TwoBlockPropagationTest(unittest.TestCase):
    def test_horiz_spread(self):
        global tmpdir
        ctrl = LBSimulationController(SimulationTest, TwoBlocksGeometryTest)
        ctrl.run()

        b0 = np.load(os.path.join(tmpdir, 'test_out_blk0_dist_dump1.npy'))
        b1 = np.load(os.path.join(tmpdir, 'test_out_blk1_dist_dump1.npy'))
        ae = np.testing.assert_equal

        ae(b1[vi(1, 0, 0), 32, 32, 1], np.float32(0.11))
        ae(b1[vi(1, 1, 0), 32, 33, 1], np.float32(0.12))
        ae(b1[vi(1, -1, 0), 32, 31, 1], np.float32(0.13))
        ae(b1[vi(1, 0, 1), 33, 32, 1], np.float32(0.21))
        ae(b1[vi(1, 0, -1), 31, 32, 1], np.float32(0.23))

        ae(b0[vi(-1, 0, 0), 1, 1, 64], np.float32(0.33))
        ae(b0[vi(-1, 0, 1), 2, 1, 64], np.float32(0.35))
        ae(b0[vi(-1, 1, 0), 1, 2, 64], np.float32(0.34))


if __name__ == '__main__':
    tmpdir = tempfile.mkdtemp()
    unittest.main()

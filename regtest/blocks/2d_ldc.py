#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.lbm_ldc_multi import LDCGeometry, LDCSim
from sailfish.controller import LBSimulationController
from regtest.blocks import util

block_size = 64
tmpdir = tempfile.mkdtemp()
blocks = 1
output = ''

class SimulationTest(LDCSim):
    @classmethod
    def update_defaults(cls, defaults):
        global block_size, blocks, output
        LDCSim.update_defaults(defaults)
        defaults['block_size'] = block_size
        defaults['blocks'] = blocks
        defaults['max_iters'] = 100
        defaults['quiet'] = True
        defaults['output'] = output
        defaults['cuda_cache'] = False

# NOTE: This test class is not thread safe.
class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, output
        output = os.path.join(tmpdir, 'ref')
        blocks = 1
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run(ignore_cmdline=True)
        cls.ref = np.load('%s_blk0_100.npz' % output)
        cls.rho = cls.ref['rho']
        cls.vx  = cls.ref['v'][0]
        cls.vy  = cls.ref['v'][1]

    def test_4blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_4block')
        blocks = 4
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run(ignore_cmdline=True)
        testdata0 = np.load('%s_blk0_100.npz' % output)
        testdata1 = np.load('%s_blk1_100.npz' % output)
        testdata2 = np.load('%s_blk2_100.npz' % output)
        testdata3 = np.load('%s_blk3_100.npz' % output)

        rho_p1 = np.vstack([testdata0['rho'], testdata1['rho']])
        rho_p2 = np.vstack([testdata2['rho'], testdata3['rho']])
        rho    = np.hstack([rho_p1, rho_p2])

        vx_p1  = np.vstack([testdata0['v'][0], testdata1['v'][0]])
        vx_p2  = np.vstack([testdata2['v'][0], testdata3['v'][0]])
        vx     = np.hstack([vx_p1, vx_p2])

        vy_p1  = np.vstack([testdata0['v'][1], testdata1['v'][1]])
        vy_p2  = np.vstack([testdata2['v'][1], testdata3['v'][1]])
        vy     = np.hstack([vy_p1, vy_p2])

        np.testing.assert_array_almost_equal(rho, self.rho)
        np.testing.assert_array_almost_equal(vx, self.vx)
        np.testing.assert_array_almost_equal(vy, self.vy)

    def test_3blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_3block')
        blocks = 3
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run(ignore_cmdline=True)
        testdata0 = np.load('%s_blk0_100.npz' % output)
        testdata1 = np.load('%s_blk1_100.npz' % output)
        testdata2 = np.load('%s_blk2_100.npz' % output)

        rho_p1 = np.vstack([testdata0['rho'], testdata1['rho']])
        rho    = np.hstack([rho_p1, testdata2['rho']])

        vx_p1  = np.vstack([testdata0['v'][0], testdata1['v'][0]])
        vx     = np.hstack([vx_p1, testdata2['v'][0]])

        vy_p1  = np.vstack([testdata0['v'][1], testdata1['v'][1]])
        vy     = np.hstack([vy_p1, testdata2['v'][1]])

        np.testing.assert_array_almost_equal(rho, self.rho)
        np.testing.assert_array_almost_equal(vx, self.vx)
        np.testing.assert_array_almost_equal(vy, self.vy)


def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    unittest.main()

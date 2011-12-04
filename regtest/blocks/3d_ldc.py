#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.lbm_ldc_multi_3d import LDCGeometry, LDCSim
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
        defaults['max_iters'] = 200
        defaults['quiet'] = True
        defaults['output'] = output

# NOTE: This test class is not thread safe.
class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, output
        output = os.path.join(tmpdir, 'ref')
        blocks = 1
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run()
        cls.ref = np.load('%s_blk0_200.npz' % output)
        cls.rho = cls.ref['rho']
        cls.vx  = cls.ref['v'][0]
        cls.vy  = cls.ref['v'][1]
        cls.vz  = cls.ref['v'][2]

    def test_8blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_8block')
        blocks = 8
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run()
        testdata0 = np.load('%s_blk0_200.npz' % output)
        testdata1 = np.load('%s_blk1_200.npz' % output)
        testdata2 = np.load('%s_blk2_200.npz' % output)
        testdata3 = np.load('%s_blk3_200.npz' % output)
        testdata4 = np.load('%s_blk4_200.npz' % output)
        testdata5 = np.load('%s_blk5_200.npz' % output)
        testdata6 = np.load('%s_blk6_200.npz' % output)
        testdata7 = np.load('%s_blk7_200.npz' % output)

        rho_p1 = np.vstack([testdata0['rho'], testdata1['rho']])
        rho_p2 = np.vstack([testdata2['rho'], testdata3['rho']])
        rho_p3 = np.vstack([testdata4['rho'], testdata5['rho']])
        rho_p4 = np.vstack([testdata6['rho'], testdata7['rho']])
        rho_pp1 = np.hstack([rho_p1, rho_p2])
        rho_pp2 = np.hstack([rho_p3, rho_p4])
        rho = np.dstack([rho_pp1, rho_pp2])

        vx_p1 = np.vstack([testdata0['v'][0], testdata1['v'][0]])
        vx_p2 = np.vstack([testdata2['v'][0], testdata3['v'][0]])
        vx_p3 = np.vstack([testdata4['v'][0], testdata5['v'][0]])
        vx_p4 = np.vstack([testdata6['v'][0], testdata7['v'][0]])
        vx_pp1 = np.hstack([vx_p1, vx_p2])
        vx_pp2 = np.hstack([vx_p3, vx_p4])
        vx = np.dstack([vx_pp1, vx_pp2])

        vy_p1 = np.vstack([testdata0['v'][1], testdata1['v'][1]])
        vy_p2 = np.vstack([testdata2['v'][1], testdata3['v'][1]])
        vy_p3 = np.vstack([testdata4['v'][1], testdata5['v'][1]])
        vy_p4 = np.vstack([testdata6['v'][1], testdata7['v'][1]])
        vy_pp1 = np.hstack([vy_p1, vy_p2])
        vy_pp2 = np.hstack([vy_p3, vy_p4])
        vy = np.dstack([vy_pp1, vy_pp2])

        vz_p1 = np.vstack([testdata0['v'][2], testdata1['v'][2]])
        vz_p2 = np.vstack([testdata2['v'][2], testdata3['v'][2]])
        vz_p3 = np.vstack([testdata4['v'][2], testdata5['v'][2]])
        vz_p4 = np.vstack([testdata6['v'][2], testdata7['v'][2]])
        vz_pp1 = np.hstack([vz_p1, vz_p2])
        vz_pp2 = np.hstack([vz_p3, vz_p4])
        vz = np.dstack([vz_pp1, vz_pp2])

        np.testing.assert_array_almost_equal(rho, self.rho)
        np.testing.assert_array_almost_equal(vx, self.vx)
        np.testing.assert_array_almost_equal(vy, self.vy)
        np.testing.assert_array_almost_equal(vz, self.vz)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    unittest.main()
    shutil.rmtree(tmpdir)

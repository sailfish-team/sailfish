#!/usr/bin/python

import os
import tempfile
import unittest

import numpy as np

from examples.lbm_ldc_multi import LDCGeometry, LDCSim
from sailfish.controller import LBSimulationController

tmpdir = tempfile.mkdtemp()
blocks = 1
output = ''

class SimulationTest(LDCSim):
    @classmethod
    def update_defaults(cls, defaults):
        global blocks, vertical, output
        LDCSim.update_defaults(defaults)
        defaults['blocks'] = blocks
        defaults['max_iters'] = 100
        defaults['quiet'] = True
        defaults['output'] = output

# NOTE: This test class is not thread safe.
class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'ref')
        blocks = 1
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run()
        cls.ref = np.load('%s_blk0_100.npz' % output)
        cls.rho = cls.ref['rho']
        cls.vx  = cls.ref['v'][0]
        cls.vy  = cls.ref['v'][1]

    def test_4blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'horiz_2block')
        blocks = 4
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run()
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


if __name__ == '__main__':
    unittest.main()

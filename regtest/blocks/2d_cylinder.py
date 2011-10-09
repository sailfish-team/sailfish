#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.lbm_cylinder_multi import CylinderSimulation, CylinderGeometry
from sailfish.controller import LBSimulationController
from regtest.blocks import util

tmpdir = tempfile.mkdtemp()
block_size = 64
blocks = 2
vertical = False
output = ''

class SimulationTest(CylinderSimulation):
    @classmethod
    def update_defaults(cls, defaults):
        global block_size, blocks, vertical, output
        CylinderSimulation.update_defaults(defaults)
        defaults['block_size'] = block_size
        defaults['blocks'] = blocks
        defaults['vertical'] = vertical
        defaults['max_iters'] = 100
        defaults['quiet'] = True
        defaults['output'] = output

# NOTE: This test class is not thread safe.
class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'href')
        blocks = 1
        ctrl = LBSimulationController(SimulationTest, CylinderGeometry)
        ctrl.run()
        cls.href = np.load('%s_blk0_100.npz' % output)
        cls.hrho = cls.href['rho']
        cls.hvx  = cls.href['v'][0]
        cls.hvy  = cls.href['v'][1]

        output = os.path.join(tmpdir, 'vref')
        vertical = True
        ctrl = LBSimulationController(SimulationTest, CylinderGeometry)
        ctrl.run()
        cls.vref = np.load('%s_blk0_100.npz' % output)
        cls.vrho = cls.vref['rho']
        cls.vvx  = cls.vref['v'][0]
        cls.vvy  = cls.vref['v'][1]

    def test_horiz_2blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'horiz_2block')
        blocks = 2
        vertical = False
        ctrl = LBSimulationController(SimulationTest, CylinderGeometry)
        ctrl.run()
        testdata0 = np.load('%s_blk0_100.npz' % output)
        testdata1 = np.load('%s_blk1_100.npz' % output)

        rho = np.hstack([testdata0['rho'], testdata1['rho']])
        vx = np.hstack([testdata0['v'][0], testdata1['v'][0]])
        vy = np.hstack([testdata0['v'][1], testdata1['v'][1]])

        np.testing.assert_array_almost_equal(rho, self.hrho)
        np.testing.assert_array_almost_equal(vx, self.hvx)
        np.testing.assert_array_almost_equal(vy, self.hvy)

    def test_horiz_3blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'horiz_3block')
        blocks = 3
        vertical = False
        ctrl = LBSimulationController(SimulationTest, CylinderGeometry)
        ctrl.run()
        testdata0 = np.load('%s_blk0_100.npz' % output)
        testdata1 = np.load('%s_blk1_100.npz' % output)
        testdata2 = np.load('%s_blk2_100.npz' % output)

        rho = np.hstack([testdata0['rho'], testdata1['rho'], testdata2['rho']])
        vx = np.hstack([testdata0['v'][0], testdata1['v'][0], testdata2['v'][0]])
        vy = np.hstack([testdata0['v'][1], testdata1['v'][1], testdata2['v'][1]])

        np.testing.assert_array_almost_equal(rho, self.hrho)
        np.testing.assert_array_almost_equal(vx, self.hvx)
        np.testing.assert_array_almost_equal(vy, self.hvy)

    def test_vert_2blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'vert_2block')
        blocks = 2
        vertical = True
        ctrl = LBSimulationController(SimulationTest, CylinderGeometry)
        ctrl.run()
        testdata0 = np.load('%s_blk0_100.npz' % output)
        testdata1 = np.load('%s_blk1_100.npz' % output)

        rho = np.vstack([testdata0['rho'], testdata1['rho']])
        vx = np.vstack([testdata0['v'][0], testdata1['v'][0]])
        vy = np.vstack([testdata0['v'][1], testdata1['v'][1]])

        np.testing.assert_array_almost_equal(rho, self.vrho)
        np.testing.assert_array_almost_equal(vx, self.vvx)
        np.testing.assert_array_almost_equal(vy, self.vvy)

    def test_vert_3blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'vert_3block')
        blocks = 3
        vertical = True
        ctrl = LBSimulationController(SimulationTest, CylinderGeometry)
        ctrl.run()
        testdata0 = np.load('%s_blk0_100.npz' % output)
        testdata1 = np.load('%s_blk1_100.npz' % output)
        testdata2 = np.load('%s_blk2_100.npz' % output)

        rho = np.vstack([testdata0['rho'], testdata1['rho'], testdata2['rho']])
        vx = np.vstack([testdata0['v'][0], testdata1['v'][0], testdata2['v'][0]])
        vy = np.vstack([testdata0['v'][1], testdata1['v'][1], testdata2['v'][1]])

        np.testing.assert_array_almost_equal(rho, self.vrho)
        np.testing.assert_array_almost_equal(vx, self.vvx)
        np.testing.assert_array_almost_equal(vy, self.vvy)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    unittest.main()
    shutil.rmtree(tmpdir)

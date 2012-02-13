#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.sphere_3d import SphereSimulation, SphereGeometry
from sailfish.controller import LBSimulationController
from regtest.blocks import util

block_size = 64
blocks = 2
output = ''

class SimulationTest(SphereSimulation):
    @classmethod
    def update_defaults(cls, defaults):
        global block_size, blocks, output
        SphereSimulation.update_defaults(defaults)
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
        output = os.path.join(tmpdir, 'href')
        blocks = 1
        ctrl = LBSimulationController(SimulationTest, SphereGeometry)
        ctrl.run(ignore_cmdline=True)
        cls.href = np.load('%s_blk0_100.npz' % output)
        cls.hrho = cls.href['rho']
        cls.hvx  = cls.href['v'][0]
        cls.hvy  = cls.href['v'][1]
        cls.hvz  = cls.href['v'][2]

    def test_horiz_2blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_2block')
        blocks = 2
        ctrl = LBSimulationController(SimulationTest, SphereGeometry)
        ctrl.run(ignore_cmdline=True)
        testdata0 = np.load('%s_blk0_100.npz' % output)
        testdata1 = np.load('%s_blk1_100.npz' % output)

        rho = np.dstack([testdata0['rho'], testdata1['rho']])
        vx = np.dstack([testdata0['v'][0], testdata1['v'][0]])
        vy = np.dstack([testdata0['v'][1], testdata1['v'][1]])
        vz = np.dstack([testdata0['v'][2], testdata1['v'][2]])

        np.testing.assert_array_almost_equal(rho, self.hrho)
        np.testing.assert_array_almost_equal(vx, self.hvx)
        np.testing.assert_array_almost_equal(vy, self.hvy)
        np.testing.assert_array_almost_equal(vz, self.hvz)


def setUpModule():
    global tmpdir
    tmpdir = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    unittest.main()

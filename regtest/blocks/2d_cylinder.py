#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.cylinder import CylinderSimulation, CylinderGeometry
from sailfish import io
from sailfish.controller import LBSimulationController
from regtest.blocks import util
from utils.merge_subdomains import merge_subdomains

tmpdir = tempfile.mkdtemp()
block_size = 64
blocks = 2
vertical = False
output = ''
MAX_ITERS = 100

class SimulationTest(CylinderSimulation):
    @classmethod
    def update_defaults(cls, defaults):
        global block_size, blocks, vertical, output
        CylinderSimulation.update_defaults(defaults)
        defaults['block_size'] = block_size
        defaults['blocks'] = blocks
        defaults['vertical'] = vertical
        defaults['max_iters'] = MAX_ITERS
        defaults['quiet'] = True
        defaults['output'] = output
        defaults['cuda_cache'] = False

# NOTE: This test class is not thread safe.
class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'href')
        blocks = 1
        LBSimulationController(SimulationTest, CylinderGeometry).run(ignore_cmdline=True)
        cls.digits = io.filename_iter_digits(MAX_ITERS)
        cls.href = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))
        cls.hrho = cls.href['rho']
        cls.hvx  = cls.href['v'][0]
        cls.hvy  = cls.href['v'][1]

        output = os.path.join(tmpdir, 'vref')
        vertical = True
        LBSimulationController(SimulationTest, CylinderGeometry).run(ignore_cmdline=True)
        cls.vref = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))
        cls.vrho = cls.vref['rho']
        cls.vvx  = cls.vref['v'][0]
        cls.vvy  = cls.vref['v'][1]

    def test_horiz_2blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'horiz_2block')
        blocks = 2
        vertical = False
        LBSimulationController(SimulationTest, CylinderGeometry).run(ignore_cmdline=True)

        merged = merge_subdomains(output, self.digits, MAX_ITERS, save=False)
        rho = merged['rho']
        vx  = merged['v'][0]
        vy  = merged['v'][1]

        np.testing.assert_array_almost_equal(rho, self.hrho)
        np.testing.assert_array_almost_equal(vx, self.hvx)
        np.testing.assert_array_almost_equal(vy, self.hvy)

    def test_horiz_3blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'horiz_3block')
        blocks = 3
        vertical = False
        LBSimulationController(SimulationTest, CylinderGeometry).run(ignore_cmdline=True)

        merged = merge_subdomains(output, self.digits, MAX_ITERS, save=False)
        rho = merged['rho']
        vx  = merged['v'][0]
        vy  = merged['v'][1]

        np.testing.assert_array_almost_equal(rho, self.hrho)
        np.testing.assert_array_almost_equal(vx, self.hvx)
        np.testing.assert_array_almost_equal(vy, self.hvy)

    def test_vert_2blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'vert_2block')
        blocks = 2
        vertical = True
        LBSimulationController(SimulationTest, CylinderGeometry).run(ignore_cmdline=True)

        merged = merge_subdomains(output, self.digits, MAX_ITERS, save=False)
        rho = merged['rho']
        vx  = merged['v'][0]
        vy  = merged['v'][1]

        np.testing.assert_array_almost_equal(rho, self.vrho)
        np.testing.assert_array_almost_equal(vx, self.vvx)
        np.testing.assert_array_almost_equal(vy, self.vvy)

    def test_vert_3blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'vert_3block')
        blocks = 3
        vertical = True
        LBSimulationController(SimulationTest, CylinderGeometry).run(ignore_cmdline=True)

        merged = merge_subdomains(output, self.digits, MAX_ITERS, save=False)
        rho = merged['rho']
        vx  = merged['v'][0]
        vy  = merged['v'][1]

        np.testing.assert_array_almost_equal(rho, self.vrho)
        np.testing.assert_array_almost_equal(vx, self.vvx)
        np.testing.assert_array_almost_equal(vy, self.vvy)


def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    unittest.main()

#!/usr/bin/python
"""Runs the cylinder test case with 2 or 3 subdomains vertically/horizontally
and compares the results against a reference solution with a single subdomain.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry2D
from examples.cylinder import CylinderSimulation
from sailfish import io
from sailfish.controller import LBSimulationController
from regtest.subdomains import util
from utils.merge_subdomains import merge_subdomains

tmpdir = tempfile.mkdtemp()
mem_align = 32
block_size = 64
blocks = 2
vertical = False
output = ''
MAX_ITERS = 100

class SimulationTest(CylinderSimulation):
    @classmethod
    def update_defaults(cls, defaults):
        CylinderSimulation.update_defaults(defaults)
        defaults.update({
            'access_pattern': access_pattern,
            'node_addressing': node_addressing,
            'block_size': block_size,
            'mem_alignment': mem_align,
            'subdomains': blocks,
            'conn_axis': 'y' if vertical else 'x',
            'vertical': vertical,
            'max_iters': MAX_ITERS,
            'quiet': True,
            'output': output,
            'cuda_cache': False})

# NOTE: This test class is not thread safe.
class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'href')
        blocks = 1
        LBSimulationController(SimulationTest, EqualSubdomainsGeometry2D).run(ignore_cmdline=True)
        cls.digits = io.filename_iter_digits(MAX_ITERS)
        cls.href = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))
        cls.hrho = cls.href['rho']
        cls.hvx  = cls.href['v'][0]
        cls.hvy  = cls.href['v'][1]

        output = os.path.join(tmpdir, 'vref')
        vertical = True
        LBSimulationController(SimulationTest, EqualSubdomainsGeometry2D).run(ignore_cmdline=True)
        cls.vref = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))
        cls.vrho = cls.vref['rho']
        cls.vvx  = cls.vref['v'][0]
        cls.vvy  = cls.vref['v'][1]

    def test_horiz_2blocks(self):
        global blocks, vertical, output
        output = os.path.join(tmpdir, 'horiz_2block')
        blocks = 2
        vertical = False
        LBSimulationController(SimulationTest, EqualSubdomainsGeometry2D).run(ignore_cmdline=True)

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
        LBSimulationController(SimulationTest, EqualSubdomainsGeometry2D).run(ignore_cmdline=True)

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
        LBSimulationController(SimulationTest, EqualSubdomainsGeometry2D).run(ignore_cmdline=True)

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
        LBSimulationController(SimulationTest, EqualSubdomainsGeometry2D).run(ignore_cmdline=True)

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
    if block_size < mem_align:
        mem_align = block_size
    access_pattern = args.access_pattern
    node_addressing = args.node_addressing
    unittest.main()

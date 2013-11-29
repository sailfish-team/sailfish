#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.sphere_3d import SphereSimulation
from sailfish import io
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.controller import LBSimulationController
from regtest.subdomains import util

mem_align = 32
block_size = 64
blocks = 2
output = ''
MAX_ITERS = 100

class SimulationTest(SphereSimulation):
    @classmethod
    def update_defaults(cls, defaults):
        global block_size, blocks, output
        SphereSimulation.update_defaults(defaults)
        defaults['block_size'] = block_size
        defaults['mem_alignment'] = mem_align
        defaults['subdomains'] = blocks
        defaults['max_iters'] = MAX_ITERS
        defaults['output'] = output
        defaults['cuda_cache'] = False
        defaults.update({
            'access_pattern': access_pattern,
            'node_addressing': node_addressing,
            'silent': True,
            })

# NOTE: This test class is not thread safe.
class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, output
        output = os.path.join(tmpdir, 'href')
        blocks = 1
        ctrl = LBSimulationController(SimulationTest, EqualSubdomainsGeometry3D)
        ctrl.run(ignore_cmdline=True)
        cls.digits = io.filename_iter_digits(MAX_ITERS)
        cls.ref = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))

    def test_horiz_2blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_2block')
        blocks = 2
        ctrl = LBSimulationController(SimulationTest, EqualSubdomainsGeometry3D)
        ctrl.run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)


def setUpModule():
    global tmpdir
    tmpdir = tempfile.mkdtemp()


def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    access_pattern = args.access_pattern
    node_addressing = args.node_addressing
    if block_size < mem_align:
        mem_align = block_size
    unittest.main()

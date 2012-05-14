#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.ldc_2d import LDCGeometry, LDCSim
from sailfish import io
from sailfish.controller import LBSimulationController
from regtest.blocks import util

block_size = 64
tmpdir = tempfile.mkdtemp()
blocks = 1
output = ''
MAX_ITERS = 100

class SimulationTest(LDCSim):
    @classmethod
    def update_defaults(cls, defaults):
        global block_size, blocks, output
        LDCSim.update_defaults(defaults)
        defaults['block_size'] = block_size
        defaults['subdomains'] = blocks
        defaults['max_iters'] = MAX_ITERS
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
        LBSimulationController(SimulationTest, LDCGeometry).run(ignore_cmdline=True)
        cls.digits = io.filename_iter_digits(MAX_ITERS)
        cls.ref = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))

    def test_4blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_4block')
        blocks = 4
        LBSimulationController(SimulationTest, LDCGeometry).run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)

    def test_3blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_3block')
        blocks = 3
        LBSimulationController(SimulationTest, LDCGeometry).run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)


def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    unittest.main()

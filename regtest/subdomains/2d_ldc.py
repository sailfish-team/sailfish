#!/usr/bin/python

import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.ldc_2d import LDCSim
from sailfish import io
from sailfish.geo import LBGeometry2D
from sailfish.subdomain import SubdomainSpec2D
from sailfish.controller import LBSimulationController
from regtest.subdomains import util

mem_align = 32
block_size = 64
tmpdir = tempfile.mkdtemp()
blocks = 1
output = ''
MAX_ITERS = 100


class LDCGeometry(LBGeometry2D):
    def subdomains(self, n=None):
        subdomains = []
        bps = int(math.sqrt(blocks))

        # Special case.
        if blocks == 3:
            w1 = self.gx / 2
            w2 = self.gx - w1
            h1 = self.gy / 2
            h2 = self.gy - h1

            subdomains.append(SubdomainSpec2D((0, 0), (w1, h1)))
            subdomains.append(SubdomainSpec2D((0, h1), (w1, h2)))
            subdomains.append(SubdomainSpec2D((w1, 0), (w2, self.gy)))
            return subdomains

        if bps**2 != blocks:
            print ('Only configurations with '
                    'square-of-interger numbers of subdomains are supported. '
                    'Falling back to {0} x {0} subdomains.'.format(bps))

        yq = self.gy / bps
        ydiff = self.gy % bps
        xq = self.gx / bps
        xdiff = self.gx % bps

        for i in range(0, bps):
            xsize = xq
            if i == bps - 1:
                xsize += xdiff

            for j in range(0, bps):
                ysize = yq
                if j == bps - 1:
                    ysize += ydiff

                subdomains.append(SubdomainSpec2D((i * xq, j * yq), (xsize, ysize)))

        return subdomains


class SimulationTest(LDCSim):
    @classmethod
    def update_defaults(cls, defaults):
        LDCSim.update_defaults(defaults)
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'max_iters': MAX_ITERS,
            'silent': True,
            'output': output,
            'cuda_cache': False,
            'access_pattern': access_pattern,
            'node_addressing': node_addressing})

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
    access_pattern = args.access_pattern
    node_addressing = args.node_addressing
    if block_size < mem_align:
        mem_align = block_size
    unittest.main()

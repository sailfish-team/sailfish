#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.ldc_3d import LDCSim
from sailfish import io
from sailfish.geo import LBGeometry3D
from sailfish.subdomain import SubdomainSpec3D
from sailfish.controller import LBSimulationController
from regtest.subdomains import util

mem_align = 32
block_size = 64
tmpdir = tempfile.mkdtemp()
blocks = 1
output = ''
MAX_ITERS = 200


class LDCGeometry(LBGeometry3D):
    def subdomains(self, n=None):
        subdomains = []
        bps = int(blocks**(1.0/3))

        if bps**3 != blocks:
            print ('Only configurations with '
                    'a third power of an integer number of subdomains are '
                    'supported.  Falling back to {0} x {0} subdomains.'.
                    format(bps))

        xq = self.gx / bps
        xd = self.gx % bps
        yq = self.gy / bps
        yd = self.gy % bps
        zq = self.gz / bps
        zd = self.gz % bps

        for i in range(0, bps):
            xsize = xq
            if i == bps - 1:
                xsize += xd
            for j in range(0, bps):
                ysize = yq
                if j == bps - 1:
                    ysize += yd
                for k in range(0, bps):
                    zsize = zq
                    if k == bps - 1:
                        zsize += zd
                    subdomains.append(SubdomainSpec3D((i * xq, j * yq, k * zq),
                                (xsize, ysize, zsize)))
        return subdomains



class SimulationTest(LDCSim):
    @classmethod
    def update_defaults(cls, defaults):
        global block_size, blocks, output
        LDCSim.update_defaults(defaults)
        defaults['block_size'] = block_size
        defaults['mem_alignment'] = mem_align
        defaults['max_iters'] = MAX_ITERS
        defaults['quiet'] = True
        defaults['output'] = output
        defaults['cuda_cache'] = False
        defaults.update({
            'access_pattern': access_pattern
            })

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

    def test_8blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, 'horiz_8block')
        blocks = 8
        ctrl = LBSimulationController(SimulationTest, LDCGeometry)
        ctrl.run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)


def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    access_pattern = args.access_pattern
    if block_size < mem_align:
        mem_align = block_size
    unittest.main()

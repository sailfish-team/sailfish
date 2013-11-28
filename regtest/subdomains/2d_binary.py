#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.binary_fluid.fe_separation_2d import SeparationFESim
from examples.binary_fluid.sc_separation_2d import SeparationSCSim, SeparationDomain
from sailfish import io
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry2D
from sailfish.subdomain import SubdomainSpec2D
from regtest.subdomains import util

MAX_ITERS = 10
mem_align = 32
block_size = 64
output = ''
tmpdir = tempfile.mkdtemp()


class SCTestDomain(SeparationDomain):
    def initial_conditions(self, sim, hx, hy):
        np.random.seed(1234)

        rho = np.random.rand(self.gy, self.gx) / 1000.0
        phi = np.random.rand(self.gy, self.gx) / 1000.0

        x1 = np.min(hx)
        x2 = np.max(hx)
        y1 = np.min(hy)
        y2 = np.max(hy)

        sim.rho[:] = 1.0 + rho[y1:y2+1, x1:x2+1]
        sim.phi[:] = 1.0 + phi[y1:y2+1, x1:x2+1]


class SCSimulationTest(SeparationSCSim):
    subdomain = SCTestDomain

    @classmethod
    def update_defaults(cls, defaults):
        SeparationSCSim.update_defaults(defaults)
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'every': 1,
            'silent': True,
            'max_iters': MAX_ITERS,
            'cuda_cache': False,
            'output': output,
            'access_pattern': access_pattern,
            'node_addressing': node_addressing})


class FETestDomain(SeparationDomain):
    def initial_conditions(self, sim, hx, hy):
        np.random.seed(1234)

        phi = np.random.rand(self.gy, self.gx) / 1000.0

        x1 = np.min(hx)
        x2 = np.max(hx)
        y1 = np.min(hy)
        y2 = np.max(hy)

        sim.rho[:] = 1.0
        sim.phi[:] = 1.0 + phi[y1:y2+1, x1:x2+1]


class FESimulationTest(SeparationFESim):
    subdomain = FETestDomain

    @classmethod
    def update_defaults(cls, defaults):
        SeparationFESim.update_defaults(defaults)
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'every': 10,
            'max_iters': MAX_ITERS,
            'silent': True,
            'cuda_cache': False,
            'output': output,
            'access_pattern': access_pattern,
            'node_addressing': node_addressing})


class Geometry4Blocks(LBGeometry2D):
    def subdomains(self, n=None):
        y1 = self.gy / 2
        y2 = self.gy - y1
        x1 = self.gx / 2
        x2 = self.gx - x1

        return [SubdomainSpec2D((0, 0), (x1, y1)),
                SubdomainSpec2D((0, y1), (x1, y2)),
                SubdomainSpec2D((x1, 0), (x2, y1)),
                SubdomainSpec2D((x1, y1), (x2, y2))]

class Geometry2BlocksHoriz(LBGeometry2D):
    def subdomains(self, n=None):
        x1 = self.gx / 2
        x2 = self.gx - x1

        return [SubdomainSpec2D((0, 0), (x1, self.gy)),
                SubdomainSpec2D((x1, 0), (x2, self.gy))]


class Geometry2BlocksVertical(LBGeometry2D):
    def subdomains(self, n=None):
        y1 = self.gy / 2
        y2 = self.gy - y1

        return [SubdomainSpec2D((0, 0), (self.gx, y1)),
                SubdomainSpec2D((0, y1), (self.gx, y2))]



class TestSCInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global output
        output = os.path.join(tmpdir, 'ref')
        LBSimulationController(SCSimulationTest, LBGeometry2D).run(ignore_cmdline=True)
        cls.digits = io.filename_iter_digits(MAX_ITERS)
        cls.ref = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))

    def test_4blocks(self):
        global output
        output = os.path.join(tmpdir, '4blocks')
        LBSimulationController(SCSimulationTest, Geometry4Blocks).run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)

    def test_2blocks_horiz(self):
        global output
        output = os.path.join(tmpdir, '2blocks_horiz')
        LBSimulationController(SCSimulationTest, Geometry2BlocksHoriz).run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)

    def test_2blocks_vert(self):
        global output
        output = os.path.join(tmpdir, '2blocks_vert')
        LBSimulationController(SCSimulationTest, Geometry2BlocksVertical).run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)


class TestFEInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global output
        output = os.path.join(tmpdir, 'ref')
        LBSimulationController(FESimulationTest, LBGeometry2D).run(ignore_cmdline=True)
        cls.digits = io.filename_iter_digits(MAX_ITERS)
        cls.ref = np.load(io.filename(output, cls.digits, 0, MAX_ITERS))

    def test_4blocks(self):
        global output
        output = os.path.join(tmpdir, '4blocks')
        LBSimulationController(FESimulationTest, Geometry4Blocks).run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)

    def test_2blocks_horiz(self):
        global output
        output = os.path.join(tmpdir, '2blocks_horiz')
        LBSimulationController(FESimulationTest, Geometry2BlocksHoriz).run(ignore_cmdline=True)
        util.verify_fields(self.ref, output, self.digits, MAX_ITERS)

    def test_2blocks_vert(self):
        global output
        output = os.path.join(tmpdir, '2blocks_vert')
        LBSimulationController(FESimulationTest, Geometry2BlocksVertical).run(ignore_cmdline=True)
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

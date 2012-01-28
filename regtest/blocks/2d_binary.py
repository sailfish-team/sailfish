#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.binary_fluid.fe_separation_2d import SeparationFESim
from examples.binary_fluid.sc_separation_2d import SeparationSCSim, SeparationDomain
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import SubdomainSpec2D

MAX_ITERS = 10
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
            'every': 1,
            'max_iters': MAX_ITERS,
            'quiet': True,
            'cuda_cache': False,
            'save_src': '/tmp/foo.cu',
            'output': output,
            'debug_dump_dists': True})


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
            'periodic_x': False,
            'periodic_y': False,
            'every': 10,
            'max_iters': MAX_ITERS,
            'quiet': True,
            'cuda_cache': False,
            'output': output})


class Geometry4Blocks(LBGeometry2D):
    def blocks(self, n=None):
        y1 = self.gy / 2
        y2 = self.gy - y1
        x1 = self.gx / 2
        x2 = self.gx - x1

        return [SubdomainSpec2D((0, 0), (x1, y1)),
                SubdomainSpec2D((0, y1), (x1, y2)),
                SubdomainSpec2D((x1, 0), (x2, y1)),
                SubdomainSpec2D((x1, y1), (x2, y2))]

class Geometry2BlocksHoriz(LBGeometry2D):
    def blocks(self, n=None):
        x1 = self.gx / 2
        x2 = self.gx - x1

        return [SubdomainSpec2D((0, 0), (x1, self.gy)),
                SubdomainSpec2D((x1, 0), (x2, self.gy))]


class Geometry2BlocksVertical(LBGeometry2D):
    def blocks(self, n=None):
        y1 = self.gy / 2
        y2 = self.gy - y1

        return [SubdomainSpec2D((0, 0), (self.gx, y1)),
                SubdomainSpec2D((0, y1), (self.gx, y2))]

def rebuild_4blocks_field(f1, f2, f3, f4):
    p1 = np.vstack([f1, f2])
    p2 = np.vstack([f3, f4])
    return np.hstack([p1, p2])

def test_4blocks(ref):
    t0 = np.load('%s_blk0_%d.npz' % (output, MAX_ITERS))
    t1 = np.load('%s_blk1_%d.npz' % (output, MAX_ITERS))
    t2 = np.load('%s_blk2_%d.npz' % (output, MAX_ITERS))
    t3 = np.load('%s_blk3_%d.npz' % (output, MAX_ITERS))

    rho = rebuild_4blocks_field(t0['rho'], t1['rho'], t2['rho'], t3['rho'])
    phi = rebuild_4blocks_field(t0['phi'], t1['phi'], t2['phi'], t3['phi'])
    vx  = rebuild_4blocks_field(t0['v'][0], t1['v'][0], t2['v'][0], t3['v'][0])
    vy  = rebuild_4blocks_field(t0['v'][1], t1['v'][1], t2['v'][1], t3['v'][1])

    np.testing.assert_array_almost_equal(rho, ref['rho'])
    np.testing.assert_array_almost_equal(phi, ref['phi'])
    np.testing.assert_array_almost_equal(vx, ref['v'][0])
    np.testing.assert_array_almost_equal(vy, ref['v'][1])

def test_2blocks_horiz(ref):
    t0 = np.load('%s_blk0_%d.npz' % (output, MAX_ITERS))
    t1 = np.load('%s_blk1_%d.npz' % (output, MAX_ITERS))

    rho  = np.hstack([t0['rho'], t1['rho']])
    phi  = np.hstack([t0['phi'], t1['phi']])
    vx   = np.hstack([t0['v'][0], t1['v'][0]])
    vy   = np.hstack([t0['v'][1], t1['v'][1]])

    np.testing.assert_array_almost_equal(rho, ref['rho'])
    np.testing.assert_array_almost_equal(phi, ref['phi'])
    np.testing.assert_array_almost_equal(vx, ref['v'][0])
    np.testing.assert_array_almost_equal(vy, ref['v'][1])

def test_2blocks_vert(ref):
    t0 = np.load('%s_blk0_%d.npz' % (output, MAX_ITERS))
    t1 = np.load('%s_blk1_%d.npz' % (output, MAX_ITERS))

    rho  = np.vstack([t0['rho'], t1['rho']])
    phi  = np.vstack([t0['phi'], t1['phi']])
    vx   = np.vstack([t0['v'][0], t1['v'][0]])
    vy   = np.vstack([t0['v'][1], t1['v'][1]])

    np.testing.assert_array_almost_equal(rho, ref['rho'])
    np.testing.assert_array_almost_equal(phi, ref['phi'])
    np.testing.assert_array_almost_equal(vx, ref['v'][0])
    np.testing.assert_array_almost_equal(vy, ref['v'][1])


class TestSCInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global output
        output = os.path.join(tmpdir, 'ref')
        LBSimulationController(SCSimulationTest, LBGeometry2D).run()
        cls.ref = np.load('%s_blk0_%d.npz' % (output, MAX_ITERS))

    def test_4blocks(self):
        global output
        output = os.path.join(tmpdir, '4blocks')
        LBSimulationController(SCSimulationTest, Geometry4Blocks).run()
        test_4blocks(self.ref)

    def test_2blocks_horiz(self):
        global output
        return
        output = os.path.join(tmpdir, '2blocks_horiz')
        LBSimulationController(SCSimulationTest, Geometry2BlocksHoriz).run()
        test_2blocks_horiz(self.ref)

    def test_2blocks_vert(self):
        global output
        return
        output = os.path.join(tmpdir, '2blocks_vert')
        LBSimulationController(SCSimulationTest, Geometry2BlocksVertical).run()
        test_2blocks_vert(self.ref)


class TestFEInterblockPropagation(object): #unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global output
        output = os.path.join(tmpdir, 'ref')
        LBSimulationController(FESimulationTest, LBGeometry2D).run()
        cls.ref = np.load('%s_blk0_%d.npz' % (output, MAX_ITERS))

    def test_4blocks(self):
        global output
        output = os.path.join(tmpdir, '4blocks')
        LBSimulationController(FESimulationTest, Geometry4Blocks).run()
        test_4blocks(self.ref)

    def test_2blocks_horiz(self):
        global output
        output = os.path.join(tmpdir, '2blocks_horiz')
        LBSimulationController(FESimulationTest, Geometry2BlocksHoriz).run()
        test_2blocks_horiz(self.ref)

    def test_2blocks_vert(self):
        global output
        output = os.path.join(tmpdir, '2blocks_vert')
        LBSimulationController(FESimulationTest, Geometry2BlocksVertical).run()
        test_2blocks_vert(self.ref)


def tearDownModule():
    #shutil.rmtree(tmpdir)
    print tmpdir

if __name__ == '__main__':
    unittest.main()

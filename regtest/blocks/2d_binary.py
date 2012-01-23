#!/usr/bin/python

import os
import shutil
import tempfile
import unittest

import numpy as np

from examples.binary_fluid.sc_separation_2d import SeparationSCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import SubdomainSpec2D

output = ''
tmpdir = tempfile.mkdtemp()


class SCSimulationTest(SeparationSCSim):
    @classmethod
    def update_defaults(cls, defaults):
        SeparationSCSim.update_defaults(defaults)
        defaults.update({
            'max_iters': 100,
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
                SubdomainSpec2D((x1, y1), (x1, y2))]


def rebuild_4blocks_field(f1, f2, f3, f4):
    p1 = np.vstack([f1, f2])
    p2 = np.vstack([f3, f4])
    return np.hstack([p1, p2])

def test_4blocks(ref):
    t0 = np.load('%s_blk0_100.npz' % output)
    t1 = np.load('%s_blk1_100.npz' % output)
    t2 = np.load('%s_blk2_100.npz' % output)
    t3 = np.load('%s_blk3_100.npz' % output)

    rho = rebuild_4blocks_field(t0['rho'], t1['rho'], t2['rho'], t3['rho'])
    phi = rebuild_4blocks_field(t0['phi'], t1['phi'], t2['phi'], t3['phi'])
    vx  = rebuild_4blocks_field(t0['v'][0], t1['v'][0], t2['v'][0], t3['v'][0])
    vy  = rebuild_4blocks_field(t0['v'][1], t1['v'][1], t2['v'][1], t3['v'][1])

    np.testing.assert_array_almost_equal(rho, ref['rho'])
    np.testing.assert_array_almost_equal(phi, ref['phi'])
    np.testing.assert_array_almost_equal(vx, ref['v'][0])
    np.testing.assert_array_almost_equal(vy, ref['v'][1])


class TestInterblockPropagation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global blocks, output
        output = os.path.join(tmpdir, 'ref')
        LBSimulationController(SCSimulationTest, LBGeometry2D).run()
        cls.sc_ref = np.load('%s_blk0_100.npz' % output)

    def test_4blocks(self):
        global blocks, output
        output = os.path.join(tmpdir, '4blocks')
        LBSimulationController(SCSimulationTest, Geometry4Blocks).run()
        test_4blocks(self.sc_ref)


def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    unittest.main()

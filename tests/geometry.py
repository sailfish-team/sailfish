#!/usr/bin/python

import sys
import unittest
import numpy

sys.path.append('.')
sys.path.append('..')

from sailfish import lbm, geo, backend_dummy

class DummyOptions(object):
    boundary = 'fullbb'
    force = False
    incompressible = False

def update_options(target, src):
    for x in filter(lambda x: x[0] != '_', dir(src)):
        setattr(target, x, getattr(src, x))

class TestGeo3D(geo.LBMGeo3D):
    def define_nodes(self):
        # Create a small box of wall nodes for use in testing of direction detection.
        self.set_geo((5, 5, 5), self.NODE_WALL)
        self.fill_geo((5, 5, 5), (slice(5, 8), slice(5, 8), slice(5, 8)))

        # Wall node plate at the boundary of the simulation domain.
        self.set_geo((10, 10, 63), self.NODE_WALL)
        self.set_geo((10, 10, 0), self.NODE_WALL)
        self.fill_geo((10, 10, 0), (slice(10, 15), slice(10, 15), 0))
        self.fill_geo((10, 10, 63), (slice(10, 15), slice(10, 15), 63))

        # Create a box of wall nodes.
        self.set_geo((14, 15, 16), self.NODE_WALL)
        self.fill_geo((14, 15, 16), (slice(14, 22), slice(15, 23), slice(16, 24)))

        # Create two flat plates of velocity nodes.
        self.set_geo((22, 15, 16), self.NODE_VELOCITY, (0.1, 0.2, 0.3))
        self.set_geo((23, 15, 16), self.NODE_VELOCITY, (0.1, 0.2, 0.0))
        self.fill_geo((22, 15, 16), (22, slice(15, 23), slice(16, 24)))
        self.fill_geo((23, 15, 16), (23, slice(15, 23), slice(16, 24)))

        # Create a line of pressure nodes.
        self.set_geo((24, 15, 16), self.NODE_PRESSURE, 3.0)
        self.fill_geo((24, 15, 16), (24, slice(15, 23), 16))

        # Create a square plate for force calculation.
        self.set_geo((100, 10, 10), self.NODE_WALL)
        self.fill_geo((100, 10, 10), (100, slice(10, 14), slice(10, 14)))

        if self.options.force:
            self.add_force_object('plate', (99, 9, 9), (3, 6, 6))

    def init_dist(self, dist):
        self.velocity_to_dist((99, 11, 11), (0.1, 0.0, 0.0), dist)
        self.velocity_to_dist((99, 12, 12), (0.05, 0.0, 0.0), dist)
        self.velocity_to_dist((99, 13, 13), (0.025, 0.0, 0.0), dist)
        self.velocity_to_dist((101, 12, 12), (-0.075, 0.0, 0.0), dist)

class Test3DForce(unittest.TestCase):
    shape = (128, 64, 64)

    def setUp(self):
        backend = backend_dummy.DummyBackend()
        options = DummyOptions()
        options.force = True
        self.sim = lbm.FluidLBMSim(TestGeo3D, defaults={'grid': 'D3Q19', 'quiet': True,
            'lat_nx': self.shape[0], 'lat_ny': self.shape[1], 'lat_nz': self.shape[2]})
        self.sim._init_shape()
        update_options(self.sim.options, options)
        self.sim._init_geo()
        self.geo = TestGeo3D(self.shape, options, float=numpy.float32, backend=backend,
                sim=self.sim)

    def testForceCalculation(self):
        shape = list(self.geo.map.shape)
        dist = numpy.zeros([self.sim.grid.Q] + shape, dtype=numpy.float32)
        self.geo.init_dist(dist)
        force = self.geo.force('plate', dist)

        for i, x in enumerate((0.23559029, -0.02574653, -0.02574653)):
            self.assertAlmostEqual(force[i], x, places=4)

    def testForceObject(self):
        a = [(numpy.array([], dtype=numpy.int64),
              numpy.array([], dtype=numpy.int64),
              numpy.array([], dtype=numpy.int64)),
            (numpy.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]),
             numpy.array([10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13]),
             numpy.array([99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99])),
            (numpy.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]),
             numpy.array([10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13]),
             numpy.array([101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101])),
            (numpy.array([10, 11, 12, 13]), numpy.array([9, 9, 9, 9]), numpy.array([100, 100, 100, 100])),
            (numpy.array([10, 11, 12, 13]), numpy.array([14, 14, 14, 14]), numpy.array([100, 100, 100, 100])),
            (numpy.array([9, 9, 9, 9]), numpy.array([10, 11, 12, 13]), numpy.array([100, 100, 100, 100])),
            (numpy.array([14, 14, 14, 14]), numpy.array([10, 11, 12, 13]), numpy.array([100, 100, 100, 100])),
            (numpy.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]),
             numpy.array([ 9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12]),
             numpy.array([99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99])),
            (numpy.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]),
             numpy.array([ 9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12,  9, 10, 11, 12]),
             numpy.array([101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101])),
            (numpy.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]),
             numpy.array([11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14]),
             numpy.array([99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99])),
            (numpy.array([10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13]),
             numpy.array([11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14]),
             numpy.array([101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101])),
            (numpy.array([ 9,  9,  9,  9, 10, 11, 12]),
             numpy.array([ 9, 10, 11, 12,  9,  9,  9]),
             numpy.array([100, 100, 100, 100, 100, 100, 100])),
            (numpy.array([ 9,  9,  9,  9, 10, 11, 12]),
             numpy.array([11, 12, 13, 14, 14, 14, 14]),
             numpy.array([100, 100, 100, 100, 100, 100, 100])),
            (numpy.array([11, 12, 13, 14, 14, 14, 14]),
             numpy.array([ 9,  9,  9,  9, 10, 11, 12]),
             numpy.array([100, 100, 100, 100, 100, 100, 100])),
            (numpy.array([11, 12, 13, 14, 14, 14, 14]),
             numpy.array([14, 14, 14, 11, 12, 13, 14]),
             numpy.array([100, 100, 100, 100, 100, 100, 100])),
            (numpy.array([ 9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12]),
             numpy.array([10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13]),
             numpy.array([99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99])),
            (numpy.array([ 9,  9,  9,  9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12]),
             numpy.array([10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13]),
             numpy.array([101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101])),
            (numpy.array([11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14]),
             numpy.array([10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13]),
             numpy.array([99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99])),
            (numpy.array([11, 11, 11, 11, 12, 12, 12, 12, 13, 13, 13, 13, 14, 14, 14, 14]),
             numpy.array([10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13]),
             numpy.array([101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101]))]

        for i in range(1, self.sim.grid.Q):
            for j in range(0, 3):
                self.assertTrue(numpy.all(self.geo._force_nodes['plate'][i][0][j] == a[i][j]))

class Test3DNodeProcessing(unittest.TestCase):
    shape = (128, 64, 64)

    def setUp(self):
        backend = backend_dummy.DummyBackend()
        self.sim = lbm.FluidLBMSim(TestGeo3D, defaults={'grid': 'D3Q19', 'quiet': True,
            'lat_nx': self.shape[0], 'lat_ny': self.shape[1], 'lat_nz': self.shape[2], 'periodic_z': True})
        self.sim._init_shape()
        update_options(self.sim.options, DummyOptions())
        self.sim._init_geo()
        self.geo = TestGeo3D(self.shape, options=DummyOptions(), float=numpy.float32,
                backend=backend, sim=self.sim)

    def testPostprocess(self):
        self.geo._clear_state()
        self.geo.define_nodes()
        self.geo._prep_params()
        self.geo._postprocess_nodes()

        # Primary direction detection on a small box.
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((6, 6, 5))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((0,0,-1)), 0),
                    type=self.geo.NODE_WALL))
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((6, 6, 7))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((0,0,1)), 0),
                    type=self.geo.NODE_WALL))
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((5, 6, 6))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((-1,0,0)), 0),
                    type=self.geo.NODE_WALL))
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((7, 6, 6))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((1,0,0)), 0),
                    type=self.geo.NODE_WALL))
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((6, 5, 6))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((0,-1,0)), 0),
                    type=self.geo.NODE_WALL))
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((6, 7, 6))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((0,1,0)), 0),
                    type=self.geo.NODE_WALL))

        # Orientation detection at domain boundaries.
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((11, 11, 0))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((0,0,1)), 0),
                    type=self.geo.NODE_WALL))
        self.assertEqual(
                self.geo._decode_node(self.geo._get_map((11, 11, 63))),
                geo.NodeInfo(misc=self.geo._encode_orientation_and_param(
                        self.sim.grid.vec_to_dir((0,0,-1)), 0),
                    type=self.geo.NODE_WALL))

    def testVelocityNodes(self):
        self.geo._clear_state()
        self.geo.define_nodes()
        self.geo._prep_params()

        self.assertAlmostEqual(self.geo.params[0], 0.1)
        self.assertAlmostEqual(self.geo.params[1], 0.2)
        self.assertAlmostEqual(self.geo.params[2], 0.0)
        self.assertEqual(self.geo._get_map((23, 15, 16)), self.geo.NODE_VELOCITY)
        self.assertEqual(self.geo._get_map((23, 22, 23)), self.geo.NODE_VELOCITY)

        self.assertAlmostEqual(self.geo.params[3], 0.1)
        self.assertAlmostEqual(self.geo.params[4], 0.2)
        self.assertAlmostEqual(self.geo.params[5], 0.3)
        self.assertEqual(self.geo._get_map((22, 15, 16)), self.geo.NODE_VELOCITY)
        self.assertEqual(self.geo._get_map((22, 22, 23)), self.geo.NODE_VELOCITY)


    def testPressureNodes(self):
        self.geo._clear_state()
        self.geo.define_nodes()
        self.geo._prep_params()

        self.assertAlmostEqual(self.geo.params[6], 3.0)
        self.assertEqual(self.geo._get_map((24, 15, 16)), self.geo.NODE_PRESSURE)
        self.assertEqual(self.geo._get_map((24, 22, 16)), self.geo.NODE_PRESSURE)

if __name__ == '__main__':
    unittest.main()

#!/usr/bin/python

import sys
import unittest
import numpy

sys.path.append('.')
sys.path.append('..')

import geo
import sym
import backend_dummy

class DummyOptions(object):
	boundary = 'fullbb'
	force = False

sym.use_grid(sym.D3Q19)

class TestGeo3D(geo.LBMGeo3D):
	def _define_nodes(self):
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
		self.geo = TestGeo3D(self.shape, options, float=numpy.float32, backend=backend,
				save_cache=False, use_cache=False)

	def testForceCalculation(self):
		shape = list(self.geo.map.shape)
		dist = numpy.zeros([sym.GRID.Q] + shape, dtype=numpy.float32)
		self.geo.init_dist(dist)
		force = self.geo.force('plate', dist, dist)

		for i, x in enumerate((-0.56284726, 0.05565972, 0.05565972)):
			self.assertAlmostEqual(force[i], x)

	def testForceObject(self):
		b = set(self.geo._force_nodes['plate'])
		a = set([((99, 9, 10), 7), ((99, 9, 11), 7), ((99, 9, 12), 7),
			((99, 9, 13), 7), ((99, 10, 9), 11), ((99, 10, 10), 1),
			((99, 10, 10), 7), ((99, 10, 10), 11), ((99, 10, 11), 1),
			((99, 10, 11), 7), ((99, 10, 11), 11), ((99, 10, 11), 13),
			((99, 10, 12), 1), ((99, 10, 12), 7), ((99, 10, 12), 11),
			((99, 10, 12), 13), ((99, 10, 13), 1), ((99, 10, 13), 7),
			((99, 10, 13), 13), ((99, 10, 14), 13), ((99, 11, 9), 11),
			((99, 11, 10), 1), ((99, 11, 10), 7), ((99, 11, 10), 9),
			((99, 11, 10), 11), ((99, 11, 11), 1), ((99, 11, 11), 7),
			((99, 11, 11), 9), ((99, 11, 11), 11), ((99, 11, 11), 13),
			((99, 11, 12), 1), ((99, 11, 12), 7), ((99, 11, 12), 9),
			((99, 11, 12), 11), ((99, 11, 12), 13), ((99, 11, 13), 1),
			((99, 11, 13), 7), ((99, 11, 13), 9), ((99, 11, 13), 13),
			((99, 11, 14), 13), ((99, 12, 9), 11), ((99, 12, 10), 1),
			((99, 12, 10), 7), ((99, 12, 10), 9), ((99, 12, 10), 11),
			((99, 12, 11), 1), ((99, 12, 11), 7), ((99, 12, 11), 9),
			((99, 12, 11), 11), ((99, 12, 11), 13), ((99, 12, 12), 1),
			((99, 12, 12), 7), ((99, 12, 12), 9), ((99, 12, 12), 11),
			((99, 12, 12), 13), ((99, 12, 13), 1), ((99, 12, 13), 7),
			((99, 12, 13), 9), ((99, 12, 13), 13), ((99, 12, 14), 13),
			((99, 13, 9), 11), ((99, 13, 10), 1), ((99, 13, 10), 9),
			((99, 13, 10), 11), ((99, 13, 11), 1), ((99, 13, 11), 9),
			((99, 13, 11), 11), ((99, 13, 11), 13), ((99, 13, 12), 1),
			((99, 13, 12), 9), ((99, 13, 12), 11), ((99, 13, 12), 13),
			((99, 13, 13), 1), ((99, 13, 13), 9), ((99, 13, 13), 13),
			((99, 13, 14), 13), ((99, 14, 10), 9), ((99, 14, 11), 9),
			((99, 14, 12), 9), ((99, 14, 13), 9), ((100, 9, 9), 15),
			((100, 9, 10), 3), ((100, 9, 10), 15), ((100, 9, 11), 3),
			((100, 9, 11), 15), ((100, 9, 11), 17), ((100, 9, 12), 3),
			((100, 9, 12), 15), ((100, 9, 12), 17), ((100, 9, 13), 3),
			((100, 9, 13), 17), ((100, 9, 14), 17), ((100, 10, 9), 5),
			((100, 10, 9), 15), ((100, 10, 14), 6), ((100, 10, 14), 17),
			((100, 11, 9), 5), ((100, 11, 9), 15), ((100, 11, 9), 16),
			((100, 11, 14), 6), ((100, 11, 14), 17), ((100, 11, 14), 18),
			((100, 12, 9), 5), ((100, 12, 9), 15), ((100, 12, 9), 16),
			((100, 12, 14), 6), ((100, 12, 14), 17), ((100, 12, 14), 18),
			((100, 13, 9), 5), ((100, 13, 9), 16), ((100, 13, 14), 6),
			((100, 13, 14), 18), ((100, 14, 9), 16), ((100, 14, 10), 4),
			((100, 14, 10), 16), ((100, 14, 11), 4), ((100, 14, 11), 16),
			((100, 14, 11), 18), ((100, 14, 12), 4), ((100, 14, 12), 16),
			((100, 14, 12), 18), ((100, 14, 13), 4), ((100, 14, 13), 18),
			((100, 14, 14), 18), ((101, 9, 10), 8), ((101, 9, 11), 8),
			((101, 9, 12), 8), ((101, 9, 13), 8), ((101, 10, 9), 12),
			((101, 10, 10), 2), ((101, 10, 10), 8), ((101, 10, 10), 12),
			((101, 10, 11), 2), ((101, 10, 11), 8), ((101, 10, 11), 12),
			((101, 10, 11), 14), ((101, 10, 12), 2), ((101, 10, 12), 8),
			((101, 10, 12), 12), ((101, 10, 12), 14), ((101, 10, 13), 2),
			((101, 10, 13), 8), ((101, 10, 13), 14), ((101, 10, 14), 14),
			((101, 11, 9), 12), ((101, 11, 10), 2), ((101, 11, 10), 8),
			((101, 11, 10), 10), ((101, 11, 10), 12), ((101, 11, 11), 2),
			((101, 11, 11), 8), ((101, 11, 11), 10), ((101, 11, 11), 12),
			((101, 11, 11), 14), ((101, 11, 12), 2), ((101, 11, 12), 8),
			((101, 11, 12), 10), ((101, 11, 12), 12), ((101, 11, 12), 14),
			((101, 11, 13), 2), ((101, 11, 13), 8), ((101, 11, 13), 10),
			((101, 11, 13), 14), ((101, 11, 14), 14), ((101, 12, 9), 12),
			((101, 12, 10), 2), ((101, 12, 10), 8), ((101, 12, 10), 10),
			((101, 12, 10), 12), ((101, 12, 11), 2), ((101, 12, 11), 8),
			((101, 12, 11), 10), ((101, 12, 11), 12), ((101, 12, 11), 14),
			((101, 12, 12), 2), ((101, 12, 12), 8), ((101, 12, 12), 10),
			((101, 12, 12), 12), ((101, 12, 12), 14), ((101, 12, 13), 2),
			((101, 12, 13), 8), ((101, 12, 13), 10), ((101, 12, 13), 14),
			((101, 12, 14), 14), ((101, 13, 9), 12), ((101, 13, 10), 2),
			((101, 13, 10), 10), ((101, 13, 10), 12), ((101, 13, 11), 2),
			((101, 13, 11), 10), ((101, 13, 11), 12), ((101, 13, 11), 14),
			((101, 13, 12), 2), ((101, 13, 12), 10), ((101, 13, 12), 12),
			((101, 13, 12), 14), ((101, 13, 13), 2), ((101, 13, 13), 10),
			((101, 13, 13), 14), ((101, 13, 14), 14), ((101, 14, 10), 10),
			((101, 14, 11), 10), ((101, 14, 12), 10), ((101, 14, 13), 10)])
		self.assertEqual(b, a)

class Test3DNodeProcessing(unittest.TestCase):
	shape = (128, 64, 64)

	def setUp(self):
		backend = backend_dummy.DummyBackend()
		self.geo = TestGeo3D(self.shape, options=DummyOptions(), float=numpy.float32, backend=backend,
				save_cache=False, use_cache=False)

	def testPostprocess(self):
		self.geo._clear_state()
		self.geo._define_nodes()
		self.geo._postprocess_nodes()
		self.assertEqual(
				self.geo._decode_node(self.geo._get_map((14, 15, 16))),
				(self.geo.NODE_DIR_OTHER, self.geo.NODE_WALL))
		self.assertEqual(
				self.geo._decode_node(self.geo._get_map((21, 22, 23))),
				(self.geo.NODE_DIR_OTHER, self.geo.NODE_WALL))

	def testVelocityNodes(self):
		self.geo._clear_state()
		self.geo._define_nodes()

		self.assertAlmostEqual(self.geo.params[0], 0.1)
		self.assertAlmostEqual(self.geo.params[1], 0.2)
		self.assertAlmostEqual(self.geo.params[2], 0.0)
		self.assertEqual(self.geo._get_map((23, 15, 16)), self.geo.NODE_VELOCITY)
		self.assertEqual(self.geo._get_map((23, 22, 23)), self.geo.NODE_VELOCITY)

		self.assertAlmostEqual(self.geo.params[3], 0.1)
		self.assertAlmostEqual(self.geo.params[4], 0.2)
		self.assertAlmostEqual(self.geo.params[5], 0.3)
		self.assertEqual(self.geo._get_map((22, 15, 16)), self.geo.NODE_VELOCITY+1)
		self.assertEqual(self.geo._get_map((22, 22, 23)), self.geo.NODE_VELOCITY+1)

	def testPressureNodes(self):
		self.geo._clear_state()
		self.geo._define_nodes()

		self.assertAlmostEqual(self.geo.params[6], 3.0)
		self.assertEqual(self.geo._get_map((24, 15, 16)), self.geo.NODE_PRESSURE+1)
		self.assertEqual(self.geo._get_map((24, 22, 16)), self.geo.NODE_PRESSURE+1)

if __name__ == '__main__':
    unittest.main()

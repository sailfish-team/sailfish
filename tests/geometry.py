#!/usr/bin/python

import sys
import unittest
import numpy

sys.path.append('.')
sys.path.append('..')

import geo
import backend_dummy

class DummyOptions(object):
	boundary = 'fullbb'

class TestGeo3D(geo.LBMGeo3D):
	def _define_nodes(self):
		self.set_geo((14, 15, 16), self.NODE_WALL)
		self.fill_geo((14, 15, 16), (slice(14, 22), slice(15, 23), slice(16, 24)))

		self.set_geo((22, 15, 16), self.NODE_VELOCITY, (0.1, 0.2, 0.3))
		self.set_geo((23, 15, 16), self.NODE_VELOCITY, (0.1, 0.2, 0.0))
		self.fill_geo((22, 15, 16), (22, slice(15, 23), slice(16, 24)))
		self.fill_geo((23, 15, 16), (23, slice(15, 23), slice(16, 24)))

		self.set_geo((24, 15, 16), self.NODE_PRESSURE, 3.0)

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

	def testForceObject(self):
		# TODO
		pass

if __name__ == '__main__':
    unittest.main()

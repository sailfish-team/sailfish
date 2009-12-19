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

class Test3DNodeProcessing(unittest.TestCase):
	shape = (512, 128, 128)

	def setUp(self):
		backend = backend_dummy.DummyBackend()
		self.geo = geo.LBMGeo3D(self.shape, options=DummyOptions(), float=numpy.float32, backend=backend,
				save_cache=False, use_cache=False)

		for x in range(14, 22):
			for y in range(15, 23):
				for z in range(16, 24):
					self.geo.set_geo((x, y, z), self.geo.NODE_WALL)

	def testPostprocess(self):
		self.geo._postprocess_nodes()

		self.assertEqual(
				self.geo._decode_node(self.geo._get_map((14, 15, 16))),
				(self.geo.NODE_DIR_OTHER, self.geo.NODE_WALL))

	def testForceObject(self):
		pass

if __name__ == '__main__':
    unittest.main()

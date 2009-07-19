#!/usr/bin/python

import numpy
import lbm

from sim import *

class LBMGeoLDC(lbm.LBMGeo):
	"""Lid-driven cavity geometry."""

	def reset(self):
		"""Initialize the simulation for the lid-driven cavity geometry."""
		self.map = numpy.zeros((self.lat_h, self.lat_w), numpy.int32)
		# bottom/top
		for i in range(0, self.lat_w):
			self.map[0][i] = numpy.int32(GEO_WALL)
			self.map[self.lat_h-1][i] = numpy.int32(GEO_INFLOW)
		# left/right
		for i in range(0, self.lat_h):
			self.map[i][0] = self.map[i][self.lat_w-1] = numpy.int32(GEO_WALL)
		self.update_map()

	def init_dist(self, dist):
		for x in range(0, self.lat_w):
			for y in range(0, self.lat_h):
				dist[0][y][x] = numpy.float32(4.0/9.0)
				dist[1][y][x] = dist[2][y][x] = dist[3][y][x] = dist[4][y][x] = numpy.float32(1.0/9.0)
				dist[5][y][x] = dist[6][y][x] = dist[7][y][x] = dist[8][y][x] = numpy.float32(1.0/36.0)


sim = lbm.LBMSim(LBMGeoLDC)
sim.run()

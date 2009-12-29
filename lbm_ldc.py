#!/usr/bin/python

import numpy
import lbm
import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError


class LBMGeoLDC(geo.LBMGeo2D):
	"""Lid-driven cavity geometry."""

	max_v = 0.1

	def define_nodes(self):
		"""Initialize the simulation for the lid-driven cavity geometry."""
		# bottom/top
		for i in range(0, self.lat_w):
			self.set_geo((i, 0), self.NODE_WALL)
			self.set_geo((i, self.lat_h-1), self.NODE_VELOCITY, (self.max_v, 0.0))
		# left/right
		for i in range(0, self.lat_h):
			self.set_geo((0, i), self.NODE_WALL)
			self.set_geo((self.lat_w-1, i), self.NODE_WALL)

	def init_dist(self, dist):
		self.velocity_to_dist((0,0), (0.0, 0.0), dist)
		self.fill_dist((0,0), dist)

		for i in range(0, self.lat_w):
			self.velocity_to_dist((i, self.lat_h-1), (self.max_v, 0.0), dist)

	def get_reynolds(self, viscosity):
		return int((self.lat_w-1) * self.max_v/viscosity)

class LDCSim(lbm.LBMSim):

	filename = 'ldc'

	def __init__(self, geo_class):
		opts = []
		opts.append(optparse.make_option('--test_re100', dest='test_re100', action='store_true', default=False, help='generate test data for Re=100'))
		opts.append(optparse.make_option('--test_re1000', dest='test_re1000', action='store_true', default=False, help='generate test data for Re=1000'))

		lbm.LBMSim.__init__(self, geo_class, options=opts)

		if self.options.test_re100:
			self.options.batch = True
			self.options.max_iters = 50000
			self.options.lat_w = 128
			self.options.lat_h = 128
			self.options.visc = 0.127
		elif self.options.test_re1000:
			self.options.batch = True
			self.options.max_iters = 50000
			self.options.lat_w = 128
			self.options.lat_h = 128
			self.options.visc = 0.0127


		self.add_iter_hook(49999, self.output_profile)

	def output_profile(self):
		print '# Re = %d' % self.geo.get_reynolds(self.options.visc)

		for i, (x, y) in enumerate(zip(self.vx[:,int(self.options.lat_w/2)] / self.geo_class.max_v,
										self.vy[int(self.options.lat_h/2),:] / self.geo_class.max_v)):
			print float(i) / self.options.lat_h, x, y


sim = LDCSim(LBMGeoLDC)
sim.run()

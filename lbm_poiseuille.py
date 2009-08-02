#!/usr/bin/python -u

import numpy
import lbm
import geo2d

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError


class LBMGeoPoiseuille(geo2d.LBMGeo):
	"""2D Poiseuille geometry."""

	maxv = 0.1

	def _reset(self):
		self.map = numpy.zeros((self.lat_h, self.lat_w), numpy.int32)
		for i in range(0, self.lat_h):
			self.set_geo(0, i, geo2d.LBMGeo.NODE_WALL)
			self.set_geo(self.lat_w-1, i, geo2d.LBMGeo.NODE_WALL)

	def init_dist(self, dist):
		for x in range(0, self.lat_w):
			for y in range(0, self.lat_h):
				self.velocity_to_dist(0.0, 0.0, dist, x, y)

	def get_reynolds(self, viscosity):
		return int((self.lat_w-1) * self.maxv/viscosity)

class LPoiSim(lbm.LBMSim):

	def __init__(self, geo_class):
		opts = []
		opts.append(optparse.make_option('--test_re100', dest='test_re100', action='store_true', default=False, help='generate test data for Re=100'))

		lbm.LBMSim.__init__(self, geo_class, misc_options=opts)

		if self.options.test_re100:
			self.options.periodic_y = True
			self.options.batch = True
			self.options.max_iters = 500000
			self.options.lat_w = 64
			self.options.lat_h = 512
			self.options.visc = 0.01
			self.options.accel_y = -geo_class.maxv * (8.0 * self.options.visc) / ((self.options.lat_w-1)**2)

		self.add_iter_hook(1000, self.output_pars, every=True)
		self.add_iter_hook(499999, self.output_profile)

	def output_pars(self):
		print numpy.max(self.geo.mask_array_by_fluid(self.vx)),	numpy.min(self.geo.mask_array_by_fluid(self.vy)), numpy.average(self.geo.mask_array_by_fluid(self.rho))

	def output_profile(self):
		print '# Re = %d' % self.geo.get_reynolds(self.options.visc)

		for i, (x, y, z) in enumerate(zip(
				self.vx[int(self.options.lat_h/2),:],
				self.vy[int(self.options.lat_h/2),:],
				self.rho[int(self.options.lat_h/2),:],
				)):
			print i, x, y, z


sim = LPoiSim(LBMGeoPoiseuille)
sim.run()

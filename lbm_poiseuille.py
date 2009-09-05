#!/usr/bin/python -u

import numpy
import lbm
import geo2d

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError


class LBMGeoPoiseuille(geo2d.LBMGeo):
	"""2D Poiseuille geometry."""

	maxv = 0.02

	def _define_nodes(self):
		self.map = numpy.zeros((self.lat_h, self.lat_w), numpy.int32)
		if self.options.horizontal:
			for i in range(0, self.lat_w):
				self.set_geo(i, 0, geo2d.LBMGeo.NODE_WALL)
				self.set_geo(i, self.lat_h-1, geo2d.LBMGeo.NODE_WALL)
		else:
			for i in range(0, self.lat_h):
				self.set_geo(0, i, geo2d.LBMGeo.NODE_WALL)
				self.set_geo(self.lat_w-1, i, geo2d.LBMGeo.NODE_WALL)

	def init_dist(self, dist):
		for x in range(0, self.lat_w):
			for y in range(0, self.lat_h):
				self.velocity_to_dist(0.0, 0.0, dist, x, y)

	def get_reynolds(self, viscosity):
		if self.options.horizontal:
			x = self.lat_h-1
		else:
			x = self.lat_w-1
		return int(x * self.maxv/viscosity)

class LPoiSim(lbm.LBMSim):

	filename = 'poiseuille'

	def __init__(self, geo_class):
		opts = []
		opts.append(optparse.make_option('--test_re100', dest='test_re100', action='store_true', default=False, help='generate test data for Re=100'))
		opts.append(optparse.make_option('--horizontal', dest='horizontal', action='store_true', default=False, help='use horizontal channel'))

		lbm.LBMSim.__init__(self, geo_class, misc_options=opts)

		if self.options.test_re100:
			self.options.periodic_y = not self.options.horizontal
			self.options.periodic_x = self.options.horizontal
			self.options.batch = True
			self.options.max_iters = 500000
			self.options.visc = 0.1
			if self.options.horizontal:
				self.options.lat_w = 64
				self.options.lat_h = 64
				self.options.accel_x = geo_class.maxv * (8.0 * self.options.visc) / ((self.options.lat_h-1)**2)
				self.add_iter_hook(499999, self.output_profile_horiz)
			else:
				self.options.lat_w = 64
				self.options.lat_h = 64
				self.options.accel_y = geo_class.maxv * (8.0 * self.options.visc) / ((self.options.lat_w-1)**2)
				self.add_iter_hook(499999, self.output_profile_vert)

		self.add_iter_hook(1000, self.output_pars, every=True)

	def output_pars(self):
		print numpy.max(self.geo.mask_array_by_fluid(self.vx)),	numpy.max(self.geo.mask_array_by_fluid(self.vy)) / 0.02, numpy.average(self.geo.mask_array_by_fluid(self.rho))

	def output_profile_vert(self):
		print '# Re = %d' % self.geo.get_reynolds(self.options.visc)

		for i, (x, y, z) in enumerate(zip(
				self.vx[int(self.options.lat_h/2),:],
				self.vy[int(self.options.lat_h/2),:],
				self.rho[int(self.options.lat_h/2),:],
				)):
			print i, x, y, z

	def output_profile_horiz(self):
		print '# Re = %d' % self.geo.get_reynolds(self.options.visc)

		for i, (x, y, z) in enumerate(zip(
				self.vx[:,int(self.options.lat_w/2)],
				self.vy[:,int(self.options.lat_w/2)],
				self.rho[:,int(self.options.lat_w/2)],
				)):
			print i, x, y, z


sim = LPoiSim(LBMGeoPoiseuille)
sim.run()

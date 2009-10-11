#!/usr/bin/python -u

import sys

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
		if self.options.static:
			profile = self.get_velocity_profile()

			if self.options.horizontal:
				for y in range(0, self.lat_h):
					self.velocity_to_dist(profile[y], 0.0, dist, 0, y)
				for x in range(1, self.lat_w):
					dist[:,:,x:] = dist[:,:,0]
			else:
				for x in range(0, self.lat_w):
					self.velocity_to_dist(0.0, profile[x], dist, x, 0)
				for y in range(1, self.lat_h):
					dist[:,y,:] = dist[:,0,:]
		else:
			for x in range(0, self.lat_w):
				for y in range(0, self.lat_h):
					self.velocity_to_dist(0.0, 0.0, dist, x, y)

	def get_velocity_profile(self):
		width = self.get_chan_width()
		lat_width = self.get_width()
		ret = []
		h = 0

		bc = self.get_bc()
		if bc.midgrid:
			h = -0.5

		for x in range(0, lat_width):
			tx = x+h
			ret.append(4.0*self.maxv/width**2 * tx * (width-tx))

		return ret

	def get_chan_width(self):
		width = self.get_width() - 1
		bc = self.get_bc()
		if bc.midgrid:
			return width - 1
		else:
			return width

	def get_width(self):
		if self.options.horizontal:
			return self.lat_h
		else:
			return self.lat_w

	def get_reynolds(self, viscosity):
		return int(self.get_width() * self.maxv/viscosity)

class LPoiSim(lbm.LBMSim):

	filename = 'poiseuille'

	def __init__(self, geo_class, args=sys.argv[1:]):
		opts = []
		opts.append(optparse.make_option('--test', dest='test', action='store_true', default=False, help='generate test data'))
		opts.append(optparse.make_option('--horizontal', dest='horizontal', action='store_true', default=False, help='use horizontal channel'))
		opts.append(optparse.make_option('--static', dest='static', action='store_true', default=False, help='start with the correct velocity profile in the whole simulation domain'))

		lbm.LBMSim.__init__(self, geo_class, misc_options=opts, args=args)

		defaults = {'batch': True, 'max_iters': 500000, 'visc': 0.1, 'lat_w': 64, 'lat_h': 64}

		if self.options.test:
			self.options.periodic_y = not self.options.horizontal
			self.options.periodic_x = self.options.horizontal

			for k, v in defaults.iteritems():
				if k not in self.options.specified:
					setattr(self.options, k, v)

			self._init_geo()

			if self.options.horizontal:
				self.options.accel_x = geo_class.maxv * (8.0 * self.options.visc) / (self.geo.get_chan_width()**2)
				self.add_iter_hook(self.options.max_iters-1, self.output_profile_horiz)
			else:
				self.options.accel_y = geo_class.maxv * (8.0 * self.options.visc) / (self.geo.get_chan_width()**2)
				self.add_iter_hook(self.options.max_iters-1, self.output_profile_vert)

		self.add_iter_hook(1000, self.output_pars, every=True)

	def get_profile(self):
		if self.options.horizontal:
			return self.vx[:,int(self.options.lat_w/2)]
		else:
			return self.vy[int(self.options.lat_h/2),:]

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

if __name__ == '__main__':
	sim = LPoiSim(LBMGeoPoiseuille)
	sim.run()

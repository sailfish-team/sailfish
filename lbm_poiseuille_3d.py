#!/usr/bin/python -u

import sys

import math
import numpy
import lbm
import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

import sym

sym.use_grid(sym.D3Q13)

class LBMGeoPoiseuille(geo.LBMGeo3D):
	"""3D Poiseuille geometry."""

	maxv = 0.02

	def _define_nodes(self):
		radiussq = (self.get_width() / 2 - 1)**2

		if self.options.along_z:
			for x in range(0, self.lat_w):
				for y in range(0, self.lat_h):
					if (x-(self.lat_w/2-0.5))**2 + (y-(self.lat_h/2-0.5))**2 >= radiussq:
							self.set_geo((x,y,0), self.NODE_WALL)
			self.fill_geo((slice(None), slice(None), 0))

		elif self.options.along_y:
			for z in range(0, self.lat_d):
				for x in range(0, self.lat_w):
					if (x-(self.lat_w/2-0.5))**2 + (z-(self.lat_d/2-0.5))**2 >= radiussq:
						self.set_geo((x,0,z), self.NODE_WALL)
			self.fill_geo((slice(None), 0, slice(None)))
		else:
			for z in range(0, self.lat_d):
				for y in range(0, self.lat_h):
					if (y-(self.lat_h/2-0.5))**2 + (z-(self.lat_d/2-0.5))**2 >= radiussq:
						self.set_geo((0,y,z), self.NODE_WALL)
			self.fill_geo((0, slice(None), slice(None)))

	def init_dist(self, dist):
		if self.options.static:

			radius = self.get_width() / 2.0
			if self.options.along_z:
				for x in range(0, self.lat_w):
					for y in range(0, self.lat_h):
						rc = math.sqrt((x-self.lat_w/2)**2 + (y-self.lat_h/2)**2)
						if rc > radius:
							self.velocity_to_dist((x, y, 0), (0.0, 0.0, 0.0), dist)
						else:
							self.velocity_to_dist((x, y, 0), (0.0, 0.0, self.get_velocity_profile(rc)), dist)
				self.fill_dist((slice(None), slice(None), 0), dist)
			elif self.options.along_y:
				for x in range(0, self.lat_w):
					for z in range(0, self.lat_d):
						rc = math.sqrt((x-self.lat_w/2)**2 + (z-self.lat_d/2)**2)
						if rc > radius:
							self.velocity_to_dist((x, 0, z), (0.0, 0.0, 0.0), dist)
						else:
							self.velocity_to_dist((x, 0, z), (0.0, self.get_velocity_profile(rc), 0.0), dist)
				self.fill_dist((slice(None), 0, slice(None)), dist)
			else:
				for z in range(0, self.lat_d):
					for y in range(0, self.lat_h):
						rc = math.sqrt((z-self.lat_d/2)**2 + (y-self.lat_h/2)**2)
						if rc > radius:
							self.velocity_to_dist((0, y, z), (0.0, 0.0, 0.0), dist)
						else:
							self.velocity_to_dist((0, y, z), (self.get_velocity_profile(rc), 0.0, 0.0), dist)
				self.fill_dist((0, slice(None), slice(None)), dist)
		else:
			self.velocity_to_dist((0, 0, 0), (0.0, 0.0, 0.0), dist)
			self.fill_dist((0, 0, 0), dist)

	def get_velocity_profile(self, r):
		width = self.get_chan_width()
		lat_width = self.get_width()
		h = 0

		bc = geo.get_bc(self.options.bc_wall)
		if bc.midgrid:
			h = -0.5

		tx = r+h
		return self.maxv/(width/2.0)**2 * ((width/2.0)**2 - tx**2)

	def get_chan_width(self):
		width = self.get_width() - 1
		bc = geo.get_bc(self.options.bc_wall)
		if bc.midgrid:
			return width - 1
		else:
			return width

	def get_width(self):
		if self.options.along_z:
			return min(self.lat_w, self.lat_h)
		elif self.options.along_y:
			return min(self.lat_w, self.lat_d)
		else:
			return min(self.lat_h, self.lat_d)

	def get_reynolds(self, viscosity):
		return int(self.get_width() * self.maxv/viscosity)

class LPoiSim(lbm.LBMSim):

	filename = 'poiseuille'

	def __init__(self, geo_class, args=sys.argv[1:], defaults=None):
		opts = []
		opts.append(optparse.make_option('--test', dest='test', action='store_true', default=False, help='generate test data'))
		opts.append(optparse.make_option('--along_y', dest='along_y', action='store_true', default=False, help='flow along the Y direction'))
		opts.append(optparse.make_option('--along_z', dest='along_z', action='store_true', default=False, help='flow along the Z direction'))
		opts.append(optparse.make_option('--static', dest='static', action='store_true', default=False, help='start with the correct velocity profile in the whole simulation domain'))

		if defaults is not None:
			defaults_ = defaults
		else:
			defaults_ = {'max_iters': 500000, 'visc': 0.1, 'lat_w': 64, 'lat_h': 64, 'lat_d': 64}

		lbm.LBMSim.__init__(self, geo_class, misc_options=opts, args=args, defaults=defaults_)

		if self.options.test or self.options.benchmark:
			self._init_geo()
			accel = geo_class.maxv * (16.0 * self.options.visc) / (self.geo.get_chan_width()**2)

			if self.options.along_z:
				self.options.periodic_z = True
				self.options.accel_z = accel
			elif self.options.along_y:
				self.options.periodic_y = True
				self.options.accel_y = accel
			else:
				self.options.periodic_x = True
				self.options.accel_x = accel


if __name__ == '__main__':
	sim = LPoiSim(LBMGeoPoiseuille)
	sim.run()

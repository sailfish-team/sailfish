#!/usr/bin/python -u

import sys
import geo
import math
import numpy

import lbm
import sym

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

sym.use_grid(sym.D3Q13)

def cd_theoretical(diam_ratio, re):
	"""Return the theoretical value of drag coefficient.

	The form of the drag coefficient is taken from:
		Toelke, J. and Krafczyk, M. (2008) 'TeraFLOP computing on a
		desktop PC with GPUs for 3D CFD', International Journal of Computational Fluid
		Dynamics, 22:7, 443 - 456.

	Args:
	  diam_ratio: ratio of sphere diameter to pipe diameter
	  re: Reynolds number
	"""
	cd = 24.0 / re * (1.0 + 0.15 * re**0.687)

	if re <= 50 and diam_ratio <= 0.6:
		K = (1.0 - 0.75857 * diam_ratio**5) / (1.0 - 2.1050*diam_ratio + 2.00865*diam_ratio**3 - 1.7068*diam_ratio**5 + 0.72603*diam_ratio**6)
		return cd + 24.0 / re * (K - 1.0)

	if re >= 100 and re <= 800:
		return cd * 1.0 / (1.0 - 1.6*diam_ratio**1.6)

	return None

def sphere_diam(width, bc):
	"""The actual sphere diameter in lattice units."""
	w = width-1
	if bc.midgrid:
		w -= 1

#	w -= (w%4)

	return w/2.0

class LBMGeoSphere(geo.LBMGeo3D):
	"""3D pipe with a spherical obstacle in the middle.  Flow in the X direction."""
	maxv = 0.01

	def _define_nodes(self):
		radiussq = ((self.chan_diam)/2)**2
		diam = sphere_diam(self.width, geo.get_bc(self.options.bc_velocity))
		x0 = int(2.4*diam)
		y0 = (self.lat_h - 1) / 2.0
		z0 = (self.lat_d - 1) / 2.0
		h = 0.0

		bc = geo.get_bc(self.options.bc_velocity)
		if bc.midgrid:
			h = -0.5
			z0 -= 0.5
			y0 -= 0.5

		for z in range(0, self.lat_d):
			for y in range(0, self.lat_h):
				if (y0 - (y + h))**2 + (z0 - (z + h))**2 >= radiussq:
#					self.set_geo((0,y,z), self.NODE_WALL)
					self.set_geo((0,y,z), self.NODE_VELOCITY, (self.maxv, 0.0, 0.0))
		self.fill_geo((0, slice(None), slice(None)))

		# Velocity BC at the inlet.
		self.set_geo((0,0,0), self.NODE_VELOCITY, (self.maxv, 0.0, 0.0))
		self.fill_geo((0,0,0), (0, slice(None), slice(None)))

		ix0 = int(x0)
		iy0 = int(y0)
		iz0 = int(z0)

		rsq = diam**2 / 4.0

		miny = iy0
		maxy = iy0

		radius = int(diam / 2)

		for x in range(ix0-radius-2, ix0+radius+3):
			for y in range(iy0-radius-2, iy0+radius+3):
				for z in range(iz0-radius-2, iz0+radius+3):
					if (x0 - (x+h))**2 + (y0 - (y+h))**2 + (z0 - (z+h))**2 <= rsq:
						self.set_geo((x, y, z), self.NODE_WALL)

		for x in range(ix0-2, ix0+3):
			for z in range(iz0-2, iz0+3):
				for y in range(iy0-radius-2, iy0-radius+3):
					if self._get_map((x, y, z)) == self.NODE_WALL:
						miny = min(miny, y)
				for y in range(iy0+radius, iy0+radius+3):
					if self._get_map((x, y, z)) == self.NODE_WALL:
						maxy = max(maxy, y)

		self.sphere_diam = float(maxy - miny)

		bc = geo.get_bc(self.options.bc_wall)
		if bc.midgrid:
			self.sphere_diam += 1

		diam = int(diam)
		self.add_force_object('sphere', (ix0-diam/2-3, iy0-diam/2-3, iz0-diam/2-3),
						(diam+6, diam+6, diam+6))

	def init_dist(self, dist):
		self.velocity_to_dist((0, 0, 0), (self.maxv, 0.0, 0.0), dist)
		self.fill_dist((0, 0, 0), dist)

	def get_reynolds(self, visc):
		re = round(self.sphere_diam * self.maxv/visc)
		if re == 0:
			return 1
		else:
			return re

	@property
	def width(self):
		return min(self.lat_h, self.lat_d)

	@property
	def chan_diam(self):
		"""The actual channel diameter in lattice units."""
		w = self.width-1
		bc = geo.get_bc(self.options.bc_velocity)

		if bc.midgrid:
			 w -= 1

#		w -= (w%4)

		return w

class LSphereSim(lbm.LBMSim):
	filename = 'sphere3d'

	def __init__(self, geo_class, defaults={}, args=sys.argv[1:]):
		opts = []
		opts.append(optparse.make_option('--re', dest='re', type='int', help='Reynolds number', default=100))
		defaults_ = {'lat_d': 128,
			'lat_h': 128,
			'lat_w': 512,
			'max_iters': 320000,
			'model': 'mrt',
			'every': 1000}
		defaults_.update(defaults)

		lbm.LBMSim.__init__(self, geo_class, misc_options=opts, args=args, defaults=defaults_)

		diam = sphere_diam(self.options.lat_h, geo.get_bc(self.options.bc_velocity))

		# If the diameter here is odd, the channel width is even and we will end up
		# with a sphere of an even diameter so that the system can be symmetric.
		if diam % 2:
			diam -= 1.0

		# maxv / visc
		ratio = self.options.re / diam

		visc = 0.12
		maxv = ratio * visc

		# Try to keep the viscosity as high as possible to make sure the computed force
		# is correct.  For single precision calculations, flows with low viscosity and
		# low flow speed can yield incorrect values.  There are no such problems for
		# double precision simulations.
		if maxv > 0.1:
			geo_class.maxv = 0.1
			self.options.visc = geo_class.maxv / ratio
		else:
			geo_class.maxv = maxv
			self.options.visc = visc

		if self.options.verbose:
			self._timed_print('# maxv = %s' % geo_class.maxv)

		self.add_iter_hook(self.options.every, self.print_force, every=True)

		if self.options.verbose:
			self.add_iter_hook(5, self.print_theoretical_drag)

		self.coeffs = []

	def drag_coeff(self, force):
		return math.sqrt(force[0]*force[0]) * 8.0 / (math.pi *
				self.geo.maxv**2 * self.geo.sphere_diam**2)

	def drag_theo(self):
		return cd_theoretical(self.geo.sphere_diam / self.geo.chan_diam,
				self.geo.get_reynolds(self.options.visc))

	def print_theoretical_drag(self):
		self._timed_print('# sphere diam / chan diam: %s / %s' % (self.geo.sphere_diam, self.geo.chan_diam))
		self._timed_print('# drag coeff th: %s' % self.drag_theo())
		self._timed_print('# re = %s' % (self.geo.sphere_diam * self.geo.maxv/self.options.visc))

	def print_force(self):
		if self._iter & 1:
			self.backend.from_buf(self.gpu_dist2)
			curr = self.dist2
		else:
			self.backend.from_buf(self.gpu_dist1)
			curr = self.dist1

		self.force = self.geo.force('sphere', curr)
		coeff = self.drag_coeff(self.force)

		# Keep the last ten drag coefficients (for averaging in the regtest).
		self.coeffs.append(coeff)
		self.coeffs = self.coeffs[-10:]

		if self.options.verbose:
			self._timed_print(coeff)

if __name__ == '__main__':
	sim = LSphereSim(LBMGeoSphere)
	sim.run()

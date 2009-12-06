#!/usr/bin/python -u

import sys
import geo

from lbm_poiseuille_3d import LBMGeoPoiseuille, LPoiSim

class LBMGeoSphere(LBMGeoPoiseuille):
	"""2D tunnel with a cylinder."""

	maxv = 0.09375

	def _define_nodes(self):
		LBMGeoPoiseuille._define_nodes(self)

		diam = self.get_width() / 3

		if self.options.along_z:
			z0 = 2*diam
			x0 = self.lat_w / 2
			y0 = self.lat_h / 2

		elif self.options.along_y:
			y0 = 2*diam
			x0 = self.lat_w / 2
			z0 = self.lat_d / 2
		else:
			x0 = 2*diam
			y0 = self.lat_h / 2
			z0 = self.lat_d / 2

		for z in range(-diam/2, diam/2+1):
			for x in range(-diam/2, diam/2+1):
				for y in range(-diam/2, diam/2+1):
					if z**2 + x**2 + y**2 <= (diam**2)/4:
						self.set_geo((x + x0, y + y0, z + z0), self.NODE_WALL)

	def get_reynolds(self, visc):
		return int((self.get_width() / 3) * self.maxv/visc)

class LSphereSim(LPoiSim):
	filename = 'cylinder'

	def __init__(self, geo_class, args=sys.argv[1:]):
		LPoiSim.__init__(self, geo_class, args, defaults={'lat_d': 48, 'lat_h': 48, 'lat_w': 256, 'test': True, 'visc': 0.005})
		self.clear_hooks()

if __name__ == '__main__':
	sim = LSphereSim(LBMGeoSphere)
	sim.run()

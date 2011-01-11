#!/usr/bin/python -u

import sys
import numpy as np
from lbm_poiseuille_3d import LBMGeoPoiseuille, LPoiSim

class LBMGeoSphere(LBMGeoPoiseuille):
    """2D tunnel with a cylinder."""

    maxv = 0.09375

    def define_nodes(self):
        LBMGeoPoiseuille.define_nodes(self)

        diam = self.get_width() / 3

        if self.options.along_z:
            z0 = 2*diam
            x0 = self.lat_nx / 2
            y0 = self.lat_ny / 2

        elif self.options.along_y:
            y0 = 2*diam
            x0 = self.lat_nx / 2
            z0 = self.lat_nz / 2
        else:
            x0 = 2*diam
            y0 = self.lat_ny / 2
            z0 = self.lat_nz / 2

        hz, hy, hx = np.mgrid[0:self.lat_nz, 0:self.lat_ny, 0:self.lat_nx]
        node_map = (hx - x0)**2 + (hy - y0)**2 + (hz - z0)**2 < (diam/2.0)**2
        self.set_geo(node_map, self.NODE_WALL)

    def get_reynolds(self, visc):
        return int((self.get_width() / 3) * self.maxv/visc)

class LSphereSim(LPoiSim):
    filename = 'cylinder'

    def __init__(self, geo_class, args=sys.argv[1:]):
        LPoiSim.__init__(self, geo_class, args, defaults={'lat_nz': 48,
            'lat_ny': 48, 'lat_nx': 256, 'test': True, 'visc': 0.005, 'verbose':
            True})
        self.clear_hooks()

if __name__ == '__main__':
    sim = LSphereSim(LBMGeoSphere)
    sim.run()

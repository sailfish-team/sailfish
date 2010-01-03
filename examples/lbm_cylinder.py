#!/usr/bin/python -u

import sys

from sailfish import geo
from lbm_poiseuille import LBMGeoPoiseuille, LPoiSim

class LBMGeoCylinder(LBMGeoPoiseuille):
    """2D tunnel with a cylinder."""

    maxv = 0.1

    def define_nodes(self):
        LBMGeoPoiseuille.define_nodes(self)

        if self.options.horizontal:
            diam = self.lat_h / 3
            x0 = 2*diam
            y0 = self.lat_h / 2
        else:
            diam = self.lat_w / 3
            x0 = self.lat_w / 2
            y0 = self.lat_h - 2*diam

        for x in range(-diam/2, diam/2+1):
            for y in range(-diam/2, diam/2+1):
                if x**2 + y**2 <= (diam**2)/4:
                    self.set_geo((x + x0, y + y0), self.NODE_WALL)

class LCylinderSim(LPoiSim):
    filename = 'cylinder'

    def __init__(self, geo_class, args=sys.argv[1:]):
        LPoiSim.__init__(self, geo_class, args, defaults={'lat_w': 192, 'lat_h': 48, 'test': True, 'visc': 0.001, 'horizontal': True})
        self.clear_hooks()

if __name__ == '__main__':
    sim = LCylinderSim(LBMGeoCylinder)
    sim.run()

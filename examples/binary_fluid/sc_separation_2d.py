#!/usr/bin/python

import numpy
from sailfish import geo, lb_binary


class GeoSC(geo.LBMGeo2D):
    def init_fields(self):
        self.sim.rho[:] = numpy.random.rand(*self.sim.rho.shape) / 1000
        self.sim.phi[:] = numpy.random.rand(*self.sim.phi.shape) / 1000
        self.sim.rho[:] += 1.0
        self.sim.phi[:] += 1.0

class SCSim(lb_binary.ShanChenBinary):
    filename = 'sc_separation_2d'

    def __init__(self, geo_class, defaults={}):
        settings = {'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 256,
                    'lat_ny': 256, 'grid': 'D2Q9', 'G': -1.2,
                    'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'every': 5,
                    'scr_scale': 1}
        settings.update(defaults)
        lb_binary.ShanChenBinary.__init__(self, geo_class, options=[], defaults=settings)
        self.options.tau_phi = self.get_tau()

if __name__ == '__main__':
    sim = SCSim(GeoSC)
    sim.run()

#!/usr/bin/python

import numpy
from sailfish import geo, lb_single

class GeoSC(geo.LBMGeo2D):
    def init_fields(self):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = 0.2
        self.sim.rho[(self.lat_nx/2 - hx)**2 + (self.lat_ny/2 - hy)**2 <= 40**2] = 1.8

#        self.sim.rho[:] = numpy.random.rand(*self.sim.rho.shape) / 100
#        self.sim.rho[:] += 0.693


class SCSim(lb_single.ShanChenSingle):
    filename = 'sc_phase_2d'

    def __init__(self, geo_class):
        lb_single.ShanChenSingle.__init__(self, geo_class, options=[],
                                defaults={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 256,
                                'lat_ny': 256, 'grid': 'D2Q9', 'G': 5.0,
                                'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'every': 20,
                                'scr_scale': 1})

sim = SCSim(GeoSC)
sim.run()

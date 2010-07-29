#!/usr/bin/python

import random
import numpy
from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class GeoFE(geo.LBMGeo2D):

    def init_fields(self):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = 1.0
        self.sim.phi[:] = numpy.random.rand(*self.sim.phi.shape) / 100.0

class FESim(lbm.BinaryFluidFreeEnergy):
    filename = 'fe_seperation_2d'

    def __init__(self, geo_class, defaults={}):
        settings = {'verbose': True, 'lat_nx': 256,
                    'lat_ny': 256, 'grid': 'D2Q9',
                    'kappa': 2e-4, 'Gamma': 25.0, 'A': 1e-4,
                    'scr_scale': 2,
                    'tau_a': 4.5, 'tau_b': 0.8, 'tau_phi': 1.0,
                    'periodic_x': True, 'periodic_y': True}
        settings.update(defaults)

        lbm.BinaryFluidFreeEnergy.__init__(self, geo_class, options=[], defaults=settings)

if __name__ == '__main__':
    sim = FESim(GeoFE)
    sim.run()

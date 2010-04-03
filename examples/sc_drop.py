#!/usr/bin/python

import random
import numpy
from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class GeoSC(geo.LBMGeo2D):

    def define_nodes(self):
        pass

    def init_dist(self, dist):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = 0.2
        self.sim.rho[(self.lat_nx/2 - hx)**2 + (self.lat_ny/2 - hy)**2 <= 40**2] = 1.8

#        self.sim.rho[:] = numpy.random.rand(*self.sim.rho.shape) / 100
#        self.sim.rho[:] += 0.693

        self.sim.ic_fields = True

class SCSim(lbm.ShanChenSingle):
    filename = 'sc_phase_2d'

    def __init__(self, geo_class):
        lbm.ShanChenSingle.__init__(self, geo_class, options=[],
                              defaults={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 256,
                                'lat_ny': 256, 'grid': 'D2Q9', 'G': 5.0,
                                'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'every': 20,
                                'scr_scale': 1})

sim = SCSim(GeoSC)
sim.run()

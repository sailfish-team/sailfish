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

        self.sim.rho[:] = numpy.random.rand(*self.sim.rho.shape) / 1000
        self.sim.phi[:] = numpy.random.rand(*self.sim.phi.shape) / 1000
        self.sim.rho[:] += 1.0
        self.sim.phi[:] += 1.0

        self.sim.ic_fields = True

class SCSim(lbm.ShanChen):
    filename = 'sc_separation_2d'

    def __init__(self, geo_class):
        lbm.ShanChen.__init__(self, geo_class, options=[],
                              defaults={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 256,
                                'lat_ny': 256, 'grid': 'D2Q9', 'G': -1.2,
                                'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'every': 5,
                                'scr_scale': 1})
        self.options.tau_phi = self.get_tau()


sim = SCSim(GeoSC)
sim.run()

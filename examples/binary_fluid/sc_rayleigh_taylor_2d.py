#!/usr/bin/python

import random
import numpy
from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class GeoSC(geo.LBMGeo2D):

    def define_nodes(self):
        self.set_geo((0,0), self.NODE_WALL)
        self.fill_geo((0,0), (slice(None), 0))
        self.fill_geo((0,0), (slice(None), self.lat_ny-1))

    def init_dist(self, dist):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = numpy.random.rand(*self.sim.rho.shape) / 100.0
        self.sim.phi[:] = numpy.random.rand(*self.sim.phi.shape) / 100.0

        self.sim.rho[(hy <= self.lat_ny/2)] += 1.0
        self.sim.phi[(hy <= self.lat_ny/2)] = 1e-4

        self.sim.rho[(hy > self.lat_ny/2)] = 1e-4
        self.sim.phi[(hy > self.lat_ny/2)] += 1.0

        self.sim.ic_fields = True

class SCSim(lbm.ShanChen):
    filename = 'sc_instability_2d'

    def __init__(self, geo_class):
        lbm.ShanChen.__init__(self, geo_class, options=[],
                              defaults={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 640, #1280,
                                'lat_ny': 400, 'grid': 'D2Q9', 'G': -1.2,
                                'visc': 0.166666666666, 'periodic_x': True, 'scr_scale': 1})
        self.options.tau_phi = self.get_tau()
        self.add_body_force((0.0, -0.15 / self.options.lat_ny), grid=1)

sim = SCSim(GeoSC)
sim.run()

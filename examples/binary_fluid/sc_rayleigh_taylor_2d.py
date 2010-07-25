#!/usr/bin/python

import random
import numpy as np
from sailfish import geo, lbm

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class GeoSC(geo.LBMGeo2D):

    def define_nodes(self):
        hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]
        self.set_geo(np.logical_or(hy == 0, hy == self.lat_ny-1), self.NODE_WALL)

    def init_dist(self, dist):
        hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = np.random.rand(*self.sim.rho.shape) / 100.0
        self.sim.phi[:] = np.random.rand(*self.sim.phi.shape) / 100.0

        self.sim.rho[(hy <= self.lat_ny/2)] += 1.0
        self.sim.phi[(hy <= self.lat_ny/2)] = 1e-4

        self.sim.rho[(hy > self.lat_ny/2)] = 1e-4
        self.sim.phi[(hy > self.lat_ny/2)] += 1.0

        self.sim.ic_fields = True

class SCSim(lbm.ShanChenBinary):
    filename = 'sc_instability_2d'

    def __init__(self, geo_class):
        lbm.ShanChenBinary.__init__(self, geo_class, options=[],
                              defaults={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 640, #1280,
                                'lat_ny': 400, 'grid': 'D2Q9', 'G': -1.2,
                                'visc': 0.166666666666, 'periodic_x': True, 'scr_scale': 1})
        self.options.tau_phi = self.get_tau()
        self.add_body_force((0.0, -0.15 / self.options.lat_ny), grid=1)

sim = SCSim(GeoSC)
sim.run()

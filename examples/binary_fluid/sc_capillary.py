#!/usr/bin/python

#
# A low Reynolds number flow of a drop through a capillary channel.
#

import random
import numpy
from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class GeoSC(geo.LBMGeo2D):

    maxv = 0.005

    def define_nodes(self):
        chan_diam = 32 * self.lat_ny / 200.0
        chan_len = 200 * self.lat_ny / 200.0

        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        geometry = numpy.zeros(list(reversed(self.shape)), dtype=numpy.bool) 
        rem_y = (self.lat_ny - chan_diam) / 2

        geometry[0,:] = True
        geometry[self.lat_ny-1,:] = True

        geometry[numpy.logical_and(
                    hy < rem_y,
                    hy < rem_y - (numpy.abs((hx - self.lat_nx/2)) - chan_len/2)
                )] = True

        geometry[numpy.logical_and(
                    (self.lat_ny - hy) < rem_y,
                    (self.lat_ny - hy) < rem_y - (numpy.abs((hx - self.lat_nx/2)) - chan_len/2)
                )] = True
 
        self.set_geo_from_bool_array(geometry)
    
    def init_dist(self, dist):

        drop_diam = 30 * self.lat_ny / 200.0

        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = 1.0
        self.sim.phi[:] = 0.124
        self.sim.rho[(hx - drop_diam * 2) ** 2 + (hy - self.lat_ny / 2.0)**2 < drop_diam**2] = 0.124
        self.sim.phi[(hx - drop_diam * 2) ** 2 + (hy - self.lat_ny / 2.0)**2 < drop_diam**2] = 1.0

        self.sim.ic_fields = True

    def get_reynolds(self, viscosity):
        return int(self.lat_ny * self.maxv/viscosity)


class SCSim(lbm.ShanChenBinary):
    filename = 'sc_instability_2d'

    def __init__(self, geo_class):
        lbm.ShanChenBinary.__init__(self, geo_class, options=[],
                              defaults={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 640,
                                'lat_ny': 200, 'grid': 'D2Q9', 'G': -1.2,
                                'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'scr_scale': 1})

        self.options.tau_phi = self.get_tau()

        f1 = self.geo_class.maxv * (8.0 * self.options.visc) / self.options.lat_ny

        self.add_body_force((f1, 0.0), grid=0)
        self.add_body_force((f1, 0.0), grid=1)

        self.add_iter_hook(100, self.average_dens, every=True)

    def average_dens(self):
        print self.iter_, numpy.average(self.rho), numpy.average(self.phi) 

sim = SCSim(GeoSC)
sim.run()

#!/usr/bin/python

import random
import numpy
import math
from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class GeoSC(geo.LBMGeo2D):

    def define_nodes(self):
        pass

    def init_dist(self, dist):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = numpy.random.rand(*self.sim.rho.shape) / 100
        self.sim.rho[:] += 0.693

        self.sim.ic_fields = True

class SCSim(lbm.ShanChenSingle):
    filename = 'sc_phase_2d'

    def __init__(self, geo_class, defaults={}):
        settings={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 256,
                                'lat_ny': 256, 'grid': 'D2Q9', 'G': 5.0,
                                'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'every': 20,
                                'scr_scale': 1}
        settings.update(defaults)

        lbm.ShanChenSingle.__init__(self, geo_class, options=[], defaults=settings)
        self.add_iter_hook(1000, self.stats, every=True)

    def stats(self):
        avg = numpy.average(self.rho)
        order = math.sqrt(numpy.average(numpy.square(self.rho - avg))) / avg
        self._stats = numpy.min(self.rho), numpy.max(self.rho), order

if __name__ == '__main__':
    sim = SCSim(GeoSC)
    sim.run()

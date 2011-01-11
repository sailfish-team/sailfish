#!/usr/bin/python

import numpy
import math
from sailfish import geo, lb_single


class GeoSC(geo.LBMGeo2D):

    def init_fields(self):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = numpy.random.rand(*self.sim.rho.shape) / 100
        self.sim.rho[:] += 0.693

class SCSim(lb_single.ShanChenSingle):
    filename = 'sc_phase_2d'

    def __init__(self, geo_class, defaults={}):
        settings={'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 256,
                                'lat_ny': 256, 'grid': 'D2Q9', 'G': 5.0,
                                'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'every': 20,
                                'scr_scale': 1}
        settings.update(defaults)

        lb_single.ShanChenSingle.__init__(self, geo_class, options=[], defaults=settings)
        self.add_iter_hook(1000, self.stats, every=True)

    def stats(self):
        avg = numpy.average(self.rho)
        order = math.sqrt(numpy.average(numpy.square(self.rho - avg))) / avg
        self._stats = numpy.min(self.rho), numpy.max(self.rho), order

if __name__ == '__main__':
    sim = SCSim(GeoSC)
    sim.run()

#!/usr/bin/python

import numpy as np
from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError


class LBMGeoLDC(geo.LBMGeo2D):
    """Lid-driven cavity geometry."""

    max_v = 0.1

    def define_nodes(self):
        """Initialize the simulation for the lid-driven cavity geometry."""
        hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]

        wall_map = np.logical_or(
                np.logical_or(hx == self.lat_nx-1, hx == 0), hy == 0)

        # velocity BC at the top of the box
        self.set_geo(hy == self.lat_ny-1, self.NODE_VELOCITY, (self.max_v, 0.0))

        # walls
        self.set_geo(wall_map, self.NODE_WALL)

    def init_fields(self):
        hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]
        self.sim.rho[:] = 1.0
        self.sim.vx[hy == self.lat_ny-1] = self.max_v

    def get_reynolds(self, viscosity):
        return int((self.lat_nx-1) * self.max_v/viscosity)

class LDCSim(lbm.FluidLBMSim):

    filename = 'ldc'

    def __init__(self, geo_class, defaults={}):
        opts = []
        opts.append(optparse.make_option('--test_re100', dest='test_re100', action='store_true', default=False, help='generate test data for Re=100'))
        opts.append(optparse.make_option('--test_re1000', dest='test_re1000', action='store_true', default=False, help='generate test data for Re=1000'))

        settings = {'bc_velocity': 'equilibrium', 'verbose': True}
        settings.update(defaults)

        lbm.FluidLBMSim.__init__(self, geo_class, options=opts, defaults=settings)

        if self.options.test_re100:
            self.options.batch = True
            self.options.max_iters = 50000
            self.options.lat_nx = 128
            self.options.lat_ny = 128
            self.options.visc = 0.127
        elif self.options.test_re1000:
            self.options.batch = True
            self.options.max_iters = 50000
            self.options.lat_nx = 128
            self.options.lat_ny = 128
            self.options.visc = 0.0127

        self.add_iter_hook(49999, self.output_profile)

    def output_profile(self):
        print '# Re = %d' % self.geo.get_reynolds(self.options.visc)

        for i, (x, y) in enumerate(zip(self.vx[:,int(self.options.lat_nx/2)] / self.geo_class.max_v,
                                        self.vy[int(self.options.lat_ny/2),:] / self.geo_class.max_v)):
            print float(i) / self.options.lat_ny, x, y

if __name__ == '__main__':
    sim = LDCSim(LBMGeoLDC)
    sim.run()

#!/usr/bin/python -u

import sys
import numpy

from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError


class LBMGeoPoiseuille(geo.LBMGeo2D):
    """2D Poiseuille geometry."""

    maxv = 0.02

    def define_nodes(self):
        hy, hx = numpy.mgrid[0:self.lat_ny, 0:self.lat_nx]

        if self.options.horizontal:
            self.set_geo(hy == 0, self.NODE_WALL)
            self.set_geo(hy == self.lat_ny-1, self.NODE_WALL)
        else:
            self.set_geo(hx == 0, self.NODE_WALL)
            self.set_geo(hx == self.lat_nx-1, self.NODE_WALL)

        # If the flow is driven by a pressure difference, add pressure boundary conditions
        # at the ends of the pipe.
        if self.options.drive == 'pressure':
            if self.options.horizontal:
                pressure = self.maxv * (8.0 * self.options.visc) / (self.get_chan_width()**2) * self.lat_nx

                for i in range(1, self.lat_ny-1):
                    self.set_geo((0, i), self.NODE_PRESSURE, (1.0/3.0) + pressure/2.0)
                    self.set_geo((self.lat_nx-1, i), self.NODE_PRESSURE,
                            (1.0/3.0) - pressure/2.0)
            else:
                pressure = self.maxv * (8.0 * self.options.visc) / (self.get_chan_width()**2) * self.lat_ny

                for i in range(1, self.lat_nx-1):
                    self.set_geo((i, 0), self.NODE_PRESSURE, (1.0/3.0) + pressure/2.0)
                    self.set_geo((i, self.lat_ny-1), self.NODE_PRESSURE, (1.0/3.0) - pressure/2.0)

    def init_dist(self, dist):
        if self.options.stationary:
            if self.options.drive == 'pressure':
                # Start with correct pressure profile.
                pressure = self.maxv * (8.0 * self.options.visc) / (self.get_chan_width()**2)

                if self.options.horizontal:
                    for x in range(0, self.lat_nx):
                        self.velocity_to_dist((x, 0), (0.0, 0.0), dist, rho=(1.0 + 3.0 * pressure * (self.lat_nx/2.0 - x)))
                        self.fill_dist((x, 0), dist, (x, slice(None)))
                else:
                    for y in range(0, self.lat_ny):
                        self.velocity_to_dist((0, y), (0.0, 0.0), dist, rho=(1.0 + 3.0 * pressure * (self.lat_ny/2.0 - y)))
                        self.fill_dist((0, y), dist, (slice(None), y))
            else:
                # Start with correct velocity profile.
                profile = self.get_velocity_profile()

                if self.options.horizontal:
                    for y in range(0, self.lat_ny):
                        self.velocity_to_dist((0, y), (profile[y], 0.0), dist)
                    self.fill_dist((0, slice(None)), dist)
                else:
                    for x in range(0, self.lat_nx):
                        self.velocity_to_dist((x, 0), (0.0, profile[x]), dist)
                    self.fill_dist((slice(None), 0), dist)
        else:
            # Start with fluid at rest everywhere and no pressure gradients.
            self.velocity_to_dist((0, 0), (0.0, 0.0), dist)
            self.fill_dist((0, 0), dist)

    def get_velocity_profile(self, fluid_only=False):
        bc = geo.get_bc(self.options.bc_wall)
        width = self.get_chan_width()
        lat_width = self.get_width()
        ret = []
        h = -bc.location

        for x in range(0, lat_width):
            tx = x+h
            ret.append(4.0*self.maxv/width**2 * tx * (width-tx))

        # Remove data corresponding to non-fluid nodes if necessary.
        if fluid_only and not bc.wet_nodes:
            return ret[1:-1]

        return ret

    def get_chan_width(self):
        bc = geo.get_bc(self.options.bc_wall)
        return self.get_width() - 1 - 2 * bc.location

    def get_width(self):
        if self.options.horizontal:
            return self.lat_ny
        else:
            return self.lat_nx

    def get_reynolds(self, viscosity):
        return int(self.get_width() * self.maxv/viscosity)

class LPoiSim(lbm.FluidLBMSim):

    filename = 'poiseuille'

    def __init__(self, geo_class, args=sys.argv[1:], defaults=None):
        opts = []
        opts.append(optparse.make_option('--horizontal', dest='horizontal', action='store_true', default=False, help='use horizontal channel'))
        opts.append(optparse.make_option('--stationary', dest='stationary', action='store_true', default=False, help='start with the correct velocity profile in the whole simulation domain'))
        opts.append(optparse.make_option('--drive', dest='drive', type='choice', choices=['force', 'pressure'], default='force'))

        if defaults is not None:
            defaults_ = defaults
        else:
            defaults_ = {'max_iters': 500000, 'visc': 0.1, 'lat_nx': 64,
                    'lat_ny': 64, 'verbose': True}

        lbm.FluidLBMSim.__init__(self, geo_class, options=opts, args=args, defaults=defaults_)
        self.add_iter_hook(100, self.status, every=True)

    def status(self):
        self.res_maxv = numpy.max(self.geo.mask_array_by_fluid(self.vy))
        self.th_maxv = max(self.geo.get_velocity_profile())
        print self.res_maxv, self.th_maxv

    def _init_post_geo(self):
        if self.options.drive == 'force':
            self.options.periodic_y = not self.options.horizontal
            self.options.periodic_x = self.options.horizontal
            accel = self.geo.maxv * (8.0 * self.options.visc) / (self.geo.get_chan_width()**2)
            if self.options.horizontal:
                self.add_body_force((accel, 0.0))
            else:
                self.add_body_force((0.0, accel))

    def get_profile(self):
        if geo.get_bc(self.options.bc_wall).wet_nodes:
            if self.options.horizontal:
                return self.vx[:,int(self.options.lat_nx/2)]
            else:
                return self.vy[int(self.options.lat_ny/2),:]
        else:
            if self.options.horizontal:
                return self.vx[1:-1,int(self.options.lat_nx/2)]
            else:
                return self.vy[int(self.options.lat_ny/2),1:-1]

if __name__ == '__main__':
    sim = LPoiSim(LBMGeoPoiseuille)
    sim.run()

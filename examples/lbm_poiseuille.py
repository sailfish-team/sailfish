#!/usr/bin/python -u

import sys
import numpy as np

from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError


class LBMGeoPoiseuille(geo.LBMGeo2D):
    """2D Poiseuille geometry."""

    maxv = 0.02

    def define_nodes(self):
        hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]

        # If the flow is driven by a pressure difference, add pressure boundary conditions
        # at the ends of the pipe.
        if self.options.drive == 'pressure':
            if self.options.horizontal:
                pressure = self.maxv * (8.0 * self.options.visc) / (self.get_chan_width()**2) * self.lat_nx

                self.set_geo(hx == 0, self.NODE_PRESSURE, (1.0/3.0) + pressure/2.0)
                self.set_geo(hx == self.lat_nx-1, self.NODE_PRESSURE, (1.0/3.0) - pressure/2.0)
            else:
                pressure = self.maxv * (8.0 * self.options.visc) / (self.get_chan_width()**2) * self.lat_ny

                self.set_geo(hy == 0, self.NODE_PRESSURE, (1.0/3.0) + pressure/2.0)
                self.set_geo(hy == self.lat_ny-1, self.NODE_PRESSURE, (1.0/3.0) - pressure/2.0)

        if self.options.horizontal:
            self.set_geo(hy == 0, self.NODE_WALL)
            self.set_geo(hy == self.lat_ny-1, self.NODE_WALL)
        else:
            self.set_geo(hx == 0, self.NODE_WALL)
            self.set_geo(hx == self.lat_nx-1, self.NODE_WALL)

    def init_dist(self, dist):
        hy, hx = np.mgrid[0:self.lat_ny, 0:self.lat_nx]

        self.sim.ic_fields = True
        self.sim.rho[:] = 1.0

        if self.options.stationary:
            if self.options.drive == 'pressure':
                # Start with correct pressure profile.
                pressure = self.maxv * (8.0 * self.options.visc) / (self.get_chan_width()**2)

                if self.options.horizontal:
                    self.sim.rho[:] = 1.0 + 3.0 * pressure * (self.lat_nx/2.0 - hx)
                else:
                    self.sim.rho[:] = 1.0 + 3.0 * pressure * (self.lat_ny/2.0 - hy)
            else:
                # Start with correct velocity profile.
                profile = self.get_velocity_profile()

                if self.options.horizontal:
                    self.sim.vx[:] = self._velocity_profile(hy)
                else:
                    self.sim.vy[:] = self._velocity_profile(hx)

    def _velocity_profile(self, hi):
        bc = geo.get_bc(self.options.bc_wall)
        width = self.get_chan_width()
        lat_width = self.get_width()
        h = -bc.location

        return (4.0 * self.maxv/width**2 * (hi + h) * (width - hi - h))

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
        self.res_maxv = np.max(self.geo.mask_array_by_fluid(self.vy))
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

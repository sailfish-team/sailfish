#!/usr/bin/python -u

import sys
import math
import numpy as np

from sailfish import geo, lbm

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

def cd_theoretical(diam_ratio, re):
    """Return the theoretical value of drag coefficient.

    The form of the drag coefficient is taken from:
        Toelke, J. and Krafczyk, M. (2008) 'TeraFLOP computing on a
        desktop PC with GPUs for 3D CFD', International Journal of Computational Fluid
        Dynamics, 22:7, 443 - 456.

    Args:
      diam_ratio: ratio of sphere diameter to pipe diameter
      re: Reynolds number
    """
    cd = 24.0 / re * (1.0 + 0.15 * re**0.687)

    if re <= 50 and diam_ratio <= 0.6:
        K = (1.0 - 0.75857 * diam_ratio**5) / (1.0 - 2.1050*diam_ratio + 2.00865*diam_ratio**3 - 1.7068*diam_ratio**5 + 0.72603*diam_ratio**6)
        return cd + 24.0 / re * (K - 1.0)

    if re >= 100 and re <= 800:
        return cd * 1.0 / (1.0 - 1.6*diam_ratio**1.6)

    return None

def sphere_diam(width, bc):
    """The actual sphere diameter in lattice units."""
    return (width - 1 - 2 * bc.location) / 2.0

class LBMGeoSphere(geo.LBMGeo3D):
    """3D pipe with a spherical obstacle in the middle.  Flow in the X direction."""
    maxv = 0.01

    def define_nodes(self):
        radiussq = ((self.chan_diam)/2)**2
        diam = sphere_diam(self.width, geo.get_bc(self.options.bc_velocity))
        x0 = int(2.4*diam)
        y0 = (self.lat_ny - 1) / 2.0
        z0 = (self.lat_nz - 1) / 2.0
        h = 0.0

        bc = geo.get_bc(self.options.bc_velocity)
        h = -bc.location
        z0 -= bc.location
        y0 -= bc.location

        hz, hy, hx = np.mgrid[0:self.lat_nz, 0:self.lat_ny, 0:self.lat_nx]

        # Channel walls.
        node_map = (y0 - (hy + h))**2 + (z0 - (hz + h))**2 >= radiussq
        self.set_geo(node_map, self.NODE_VELOCITY, (self.maxv, 0.0, 0.0))

        # Inlet.
        self.set_geo(hx == 0, self.NODE_VELOCITY, (self.maxv, 0.0, 0.0))

        ix0 = int(x0)
        iy0 = int(y0)
        iz0 = int(z0)

        rsq = diam**2 / 4.0

        miny = iy0
        maxy = iy0

        radius = int(diam / 2)

        wall_map = (x0 - (hx+h))**2 + (y0 - (hy+h))**2 + (z0 - (hz+h))**2 <= rsq
        self.set_geo(wall_map, self.NODE_WALL)
        maxy = np.max(hy[wall_map])
        miny = np.min(hy[wall_map])

        bc = geo.get_bc(self.options.bc_wall)
        self.sphere_diam = float(maxy - miny) + 2.0 * bc.location

        diam = int(diam)
        self.add_force_object('sphere', (ix0-diam/2-3, iy0-diam/2-3, iz0-diam/2-3),
                        (diam+6, diam+6, diam+6))

    def init_dist(self, dist):
        self.sim.ic_fields = True
        self.sim.rho[:] = 1.0
        self.sim.vx[:] = self.maxv

    def get_reynolds(self, visc):
        re = round(self.sphere_diam * self.maxv/visc)
        if re == 0:
            return 1
        else:
            return re

    @property
    def width(self):
        return min(self.lat_ny, self.lat_nz)

    @property
    def chan_diam(self):
        """The actual channel diameter in lattice units."""
        bc = geo.get_bc(self.options.bc_velocity)
        return self.width - 1 - 2.0 * bc.location

class LSphereSim(lbm.FluidLBMSim):
    filename = 'sphere3d'

    def __init__(self, geo_class, defaults={}, args=sys.argv[1:]):
        opts = []
        opts.append(optparse.make_option('--re', dest='re', type='int', help='Reynolds number', default=100))
        defaults_ = {'lat_nz': 128,
            'lat_ny': 128,
            'lat_nx': 512,
            'max_iters': 320000,
            'model': 'mrt',
            'every': 100,
            'incompressible': True,
            'grid': 'D3Q13',
            'bc_velocity': 'fullbb',
            'verbose': True}
        defaults_.update(defaults)

        lbm.FluidLBMSim.__init__(self, geo_class, options=opts, args=args, defaults=defaults_)

        if self.options.batch and not ('every' in self.options.specified):
            self.options.every = 1000

        diam = sphere_diam(self.options.lat_ny, geo.get_bc(self.options.bc_velocity))

        # If the diameter here is odd, the channel width is even and we will end up
        # with a sphere of an even diameter so that the system can be symmetric.
        if diam % 2:
            diam -= 1.0

        # maxv / visc
        ratio = self.options.re / diam

        visc = 0.12
        maxv = ratio * visc

        # Try to keep the viscosity as high as possible to make sure the computed force
        # is correct.  For single precision calculations, flows with low viscosity and
        # low flow speed can yield incorrect values.  There are no such problems for
        # double precision simulations.
        if maxv > 0.1:
            geo_class.maxv = 0.1
            self.options.visc = geo_class.maxv / ratio
        else:
            geo_class.maxv = maxv
            self.options.visc = visc

        if self.options.verbose:
            self._timed_print('# maxv = %s' % geo_class.maxv)

        self.add_iter_hook(self.options.every, self.print_force, every=True)

        if self.options.verbose:
            self.add_iter_hook(5, self.print_theoretical_drag)

        self.coeffs = []

    def drag_coeff(self, force):
        return math.sqrt(force[0]*force[0]) * 8.0 / (math.pi *
                self.geo.maxv**2 * self.geo.sphere_diam**2)

    def drag_theo(self):
        return cd_theoretical(self.geo.sphere_diam / self.geo.chan_diam,
                self.geo.get_reynolds(self.options.visc))

    def print_theoretical_drag(self):
        self._timed_print('# sphere diam / chan diam: %s / %s' % (self.geo.sphere_diam, self.geo.chan_diam))
        self._timed_print('# drag coeff th: %s' % self.drag_theo())
        self._timed_print('# re = %s' % (self.geo.sphere_diam * self.geo.maxv/self.options.visc))

    def print_force(self):
        self.hostsync_dist()
        self.force = self.geo.force('sphere', self.dist)
        coeff = self.drag_coeff(self.force)

        self.coeffs.append(coeff)

        if self.options.verbose:
            self._timed_print(coeff)

if __name__ == '__main__':
    sim = LSphereSim(LBMGeoSphere)
    sim.run()

#!/usr/bin/python

import numpy as np
from sailfish import lbm
from sailfish import geo

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class LBMGeoLDC(geo.LBMGeo3D):
    """Lid-driven cavity geometry."""

    max_v = 0.1

    def define_nodes(self):
        """Initialize the simulation for the lid-driven cavity geometry."""
        hz, hy, hx = np.mgrid[0:self.lat_nz, 0:self.lat_ny, 0:self.lat_nx]

        wall_map = np.logical_or(hz == 0, np.logical_or(
                np.logical_or(hx == self.lat_nx-1, hx == 0),
                np.logical_or(hy == self.lat_ny-1, hy == 0)))

        self.set_geo(hz == self.lat_nz-1, self.NODE_VELOCITY, (self.max_v, 0.0, 0.0))
        self.set_geo(wall_map, self.NODE_WALL)

    def init_fields(self):
        hz, hy, hx = np.mgrid[0:self.lat_nz, 0:self.lat_ny, 0:self.lat_nx]

        self.sim.rho[:] = 1.0
        self.sim.vx[hz == self.lat_nz-1] = self.max_v

    # FIXME
    def get_reynolds(self, viscosity):
        return int((self.lat_nx-1) * self.max_v/viscosity)

class LDCSim(lbm.FluidLBMSim):

    filename = 'ldc_3d'

    def __init__(self, geo_class, defaults={}):
        opts = []
        settings={'lat_nz': 64, 'lat_ny': 64, 'lat_nx': 64,
                'grid': 'D3Q19', 'bc_velocity': 'equilibrium', 'verbose': True}
        settings.update(defaults)
        lbm.FluidLBMSim.__init__(self, geo_class, options=opts, defaults=settings)

if __name__ == '__main__':
    sim = LDCSim(LBMGeoLDC)
    sim.run()

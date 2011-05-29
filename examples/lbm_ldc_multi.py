#!/usr/bin/python

import numpy as np
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D, GeoBlock2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class LDCBlock(GeoBlock2D):
    """Lid-driven cavity geometry."""

    max_v = 0.1

    def _define_nodes(self, hx, hy):
        wall_map = np.logical_or(
                np.logical_or(hx == self.gx-1, hx == 0), hy == 0)

        self.set_geo(hy == self.gy-1, self.NODE_VELOCITY, (self.max_v, 0.0))
        self.set_geo(wall_map, self.NODE_WALL)

    def _init_fields(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[hy == self.gy-1] = self.max_v


class LDCSim(LBFluidSim, LBForcedSim):
    geo = LDCBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256})

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--blocks', type=int, default=1, help='number of blocks to use')


if __name__ == '__main__':
    ctrl = LBSimulationController(LDCSim)
    ctrl.run()

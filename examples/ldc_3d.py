#!/usr/bin/env python
"""3D lid-driven cavity."""

import numpy as np
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall, NTRegularizedVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim

class LDCBlock(Subdomain3D):
    """3D Lid-driven geometry."""

    max_v = 0.05

    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall
        velocity_bc = NTRegularizedVelocity

        wall_map = ((hz == 0) | (hx == self.gx - 1) | (hx == 0) | (hy == 0) |
                (hy == self.gy - 1))
        self.set_node(wall_map, wall_bc)
        self.set_node((hz == self.gz - 1) & np.logical_not(wall_map),
                velocity_bc((self.max_v, 0.0, 0.0)))

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vx[hz == self.gz - 1] = self.max_v


class LDCSim(LBFluidSim):
    subdomain = LDCBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 64,
            'lat_ny': 64,
            'lat_nz': 64,
            'grid': 'D3Q19'})

if __name__ == '__main__':
    ctrl = LBSimulationController(LDCSim)
    ctrl.run()

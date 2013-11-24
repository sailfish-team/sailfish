#!/usr/bin/env python
"""2D lid-driven cavity."""

import math
import numpy as np
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall, NTRegularizedVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim


class LDCBlock(Subdomain2D):
    """2D Lid-driven cavity geometry."""

    max_v = 0.1

    def boundary_conditions(self, hx, hy):
        wall_bc = NTFullBBWall
        velocity_bc = NTRegularizedVelocity

        wall_map = (hx == self.gx-1) | (hx == 0) | (hy == 0)
        self.set_node((hy == self.gy-1) & (hx > 0) & (hx < self.gx-1), velocity_bc((self.max_v, 0.0)))
        self.set_node(wall_map, wall_bc)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[hy == self.gy-1] = self.max_v


class LDCSim(LBFluidSim):
    subdomain = LDCBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256})

if __name__ == '__main__':
    ctrl = LBSimulationController(LDCSim)
    ctrl.run()

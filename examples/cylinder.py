#!/usr/bin/env python -u

import numpy as np
from sailfish.geo import EqualSubdomainsGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall, NTHalfBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim


class CylinderBlock(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        wall_bc = NTHalfBBWall
        if self.config.vertical:
            diam = self.gx / 3
            x0 = self.gx / 2
            y0 = 2 * diam

            self.set_node(hx == 0, wall_bc)
            self.set_node(hx == self.gx - 1, wall_bc)
        else:
            diam = self.gy / 3
            x0 = 2 * diam
            y0 = self.gy / 2

            self.set_node(hy == 0, wall_bc)
            self.set_node(hy == self.gy - 1, wall_bc)

        cylinder_map = (np.abs(hx - x0) < diam / 2) & (np.abs(hy - y0) < diam / 2)
        self.set_node(cylinder_map, wall_bc)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vy[:] = 0.0
        sim.vx[:] = 0.0


class CylinderSimulation(LBFluidSim, LBForcedSim):
    subdomain = CylinderBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'visc': 0.1})


    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--vertical', action='store_true')

    @classmethod
    def modify_config(cls, config):
        if config.vertical:
            config.periodic_y = True
        else:
            config.periodic_x = True

    def __init__(self, config):
        super(CylinderSimulation, self).__init__(config)

        if config.vertical:
            self.add_body_force((0.0, 1e-5))
        else:
            self.add_body_force((1e-5, 0.0))


if __name__ == '__main__':
    ctrl = LBSimulationController(CylinderSimulation, EqualSubdomainsGeometry2D)
    ctrl.run()

#!/usr/bin/python -u

import numpy as np
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import SubdomainSpec2D, Subdomain2D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim

class CylinderGeometry(LBGeometry2D):
    def blocks(self, n=None):
        blocks = []
        if self.config.vertical:
            q = self.gy / self.config.blocks
            diff = self.gy % self.config.blocks
        else:
            q = self.gx / self.config.blocks
            diff = self.gx % self.config.blocks

        for i in range(0, self.config.blocks):
            size = q
            if i == self.config.blocks-1:
                size += diff

            if self.config.vertical:
                blocks.append(SubdomainSpec2D((0, i * q), (self.gx, size)))
            else:
                blocks.append(SubdomainSpec2D((i * q, 0), (size, self.gy)))
        return blocks


class CylinderBlock(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        if self.config.vertical:
            diam = self.gx / 3
            x0 = self.gx / 2
            y0 = 2 * diam

            self.set_node(hx == 0, NTFullBBWall)
            self.set_node(hx == self.gx - 1, NTFullBBWall)
        else:
            diam = self.gy / 3
            x0 = 2 * diam
            y0 = self.gy / 2

            self.set_node(hy == 0, NTFullBBWall)
            self.set_node(hy == self.gy - 1, NTFullBBWall)

        cylinder_map = np.square(hx - x0) + np.square(hy - y0) < diam**2 / 4.0
        self.set_node(cylinder_map, NTFullBBWall)

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

        group.add_argument('--blocks', type=int, default=1, help='number of blocks to use')
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
    ctrl = LBSimulationController(CylinderSimulation, CylinderGeometry)
    ctrl.run()

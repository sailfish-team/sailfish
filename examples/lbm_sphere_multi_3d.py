#!/usr/bin/python -u

import numpy as np
from sailfish.geo import LBGeometry3D
from sailfish.geo_block import SubdomainSpec3D, Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class SphereGeometry(LBGeometry3D):
    def blocks(self, n=None):
        blocks = []
        q = self.gx / self.config.blocks
        diff = self.gx % self.config.blocks

        for i in range(0, self.config.blocks):
            size = q
            if i == self.config.blocks - 1:
                size += diff

            blocks.append(
                    SubdomainSpec3D((i * q, 0, 0), (size, self.gy, self.gz)))

        return blocks

class SphereBlock(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        diam = self.gy / 3.0
        z0 = self.gz / 2.0
        y0 = self.gy / 2.0
        x0 = 2.0 * diam

        wall_map = np.logical_or(
                        np.logical_or(hy == 0, hy == self.gy-1),
                        np.logical_or(hz == 0, hz == self.gz-1))
        self.set_node(wall_map, self.NODE_WALL)

        sphere_map = (np.square(hx - x0) +
                      np.square(hy - y0) +
                      np.square(hz - z0)) <= np.square(diam / 2.0)
        self.set_node(sphere_map, self.NODE_WALL)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vy[:] = 0.0
        sim.vx[:] = 0.0
        sim.vz[:] = 0.0


class SphereSimulation(LBFluidSim, LBForcedSim):
    subdomain = SphereBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 128,
            'lat_ny': 128,
            'lat_nz': 128,
            'visc': 0.01,
            'grid': 'D3Q19'})

    @classmethod
    def modify_config(cls, config):
        config.periodic_x = True

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--blocks', type=int, default=1, help='number of blocks to use')

    def __init__(self, config):
        super(SphereSimulation, self).__init__(config)
        self.add_body_force((1e-5, 0.0, 0.0))


if __name__ == '__main__':
    ctrl = LBSimulationController(SphereSimulation, SphereGeometry)
    ctrl.run()

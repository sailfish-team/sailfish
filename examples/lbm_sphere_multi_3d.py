#!/usr/bin/python -u

import numpy as np
from sailfish.geo import LBGeometry3D
from sailfish.geo_block import LBBlock3D, GeoBlock3D
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
                    LBBlock3D((i * q, 0, 0), (size, self.gy, self.gz)))

        return blocks

class SphereBlock(GeoBlock3D):
    def _define_nodes(self, hx, hy, hz):
        diam = self.gy / 3.0
        x0 = self.gx / 2.0
        y0 = self.gy / 2.0
        z0 = 2.0 * diam

        sphere_map = (np.square(hx - x0) +
                      np.square(hy - y0) +
                      np.square(hz - z0)) <= np.square(diam / 2.0)
        self.set_geo(sphere_map, self.NODE_WALL)

    def _init_fields(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vy[:] = 0.0
        sim.vx[:] = 0.0
        sim.vz[:] = 0.0


class SphereSimulation(LBFluidSim, LBForcedSim):
    geo = SphereBlock

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

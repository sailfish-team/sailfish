#!/usr/bin/python -u

import numpy as np
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.geo_block import Subdomain3D, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class SphereBlock(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall
        diam = self.gy / 3.0
        z0 = self.gz / 2.0
        y0 = self.gy / 2.0
        x0 = 2.0 * diam

        wall_map = np.logical_or(
                        np.logical_or(hy == 0, hy == self.gy-1),
                        np.logical_or(hz == 0, hz == self.gz-1))
        self.set_node(wall_map, wall_bc)

        sphere_map = (np.square(hx - x0) +
                      np.square(hy - y0) +
                      np.square(hz - z0)) <= np.square(diam / 2.0)
        self.set_node(sphere_map, wall_bc)

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
            'lat_ny': 64,
            'lat_nz': 64,
            'visc': 0.01,
            'grid': 'D3Q19'})

    @classmethod
    def modify_config(cls, config):
        config.periodic_x = True

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

    def __init__(self, config):
        super(SphereSimulation, self).__init__(config)
        self.add_body_force((1e-5, 0.0, 0.0))


if __name__ == '__main__':
    ctrl = LBSimulationController(SphereSimulation, EqualSubdomainsGeometry3D)
    ctrl.run()

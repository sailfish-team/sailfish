#!/usr/bin/python -u
"""Demonstrates how to load geometry from a boolean numpy array."""

import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class BoolSubdomain(Subdomain3D):

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall
        if hasattr(self.config, 'wall_map'):
            x0 = np.min(hx)
            x1 = np.max(hx)
            y0 = np.min(hy)
            y1 = np.max(hy)
            z0 = np.min(hz)
            z1 = np.max(hz)

            partial_wall_map = self.config.wall_map[z0:z1+1, y0:y1+1, x0:x1+1]
            self.set_node(partial_wall_map, wall_bc)


class BoolSimulation(LBFluidSim, LBForcedSim):
    subdomain = BoolSubdomain

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--geometry', type=str,
                help='file defining the geometry')

    @classmethod
    def modify_config(cls, config):
        if not config.geometry:
            return

        # Override lattice size based on the geometry file.
        wall_map = np.load(config.geometry)
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape
        config._wall_map = wall_map


if __name__ == '__main__':
    LBSimulationController(BoolSimulation, LBGeometry3D).run()

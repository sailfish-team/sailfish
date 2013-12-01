#!/usr/bin/env python -u
"""Demonstrates how to load geometry from an external file.

The external file is a Boolean numpy array in the .npy format.
Nodes marked as True indicate walls.

In order to generate a .npy file from an STL geometry, check
out utils/voxelizer.

The sample file pipe.npy was generated using:
  a = np.zeros((128, 41, 41), dtype=np.bool)
  hz, hy, hx = np.mgrid[0:41, 0:41, 0:128]
  a[(hz - 20)**2 + (hy - 20)**2 >
    (19.3 * (0.8 + 0.2 * np.sin(2 * pi * hx / 128.0)))**2] = True
"""

import os
import numpy as np

from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class ExternalSubdomain(Subdomain3D):
    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

    def boundary_conditions(self, hx, hy, hz):
        if hasattr(self.config, '_wall_map'):
            partial_wall_map = self.select_subdomain(
                self.config._wall_map, hx, hy, hz)
            self.set_node(partial_wall_map, NTFullBBWall)

    # Only used with node_addressing = 'indirect'.
    def load_active_node_map(self, hx, hy, hz):
        partial_wall_map = self.select_subdomain(
            self.config._wall_map, hx, hy, hz)
        self.set_active_node_map_from_wall_map(partial_wall_map)


class ExternalSimulation(LBFluidSim, LBForcedSim):
    subdomain = ExternalSubdomain

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)
        group.add_argument('--geometry', type=str, default='pipe.npy',
                           help='file defining the geometry')

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'visc': 0.01,
            'grid': 'D3Q19',
            'periodic_x': True})

    @classmethod
    def modify_config(cls, config):
        if not config.geometry:
            return

        if config.geometry == 'pipe.npy':
            # Look for the geometry file in the same directory where
            # external_geometry.py is located.
            config.geometry = os.path.join(os.path.dirname(
                os.path.realpath(__file__)), config.geometry)

        # Override lattice size based on the geometry file.
        wall_map = np.load(config.geometry)
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape
        # Add nodes corresponding to ghosts. Assumes an envelope size of 1.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map

    def __init__(self, config):
        super(ExternalSimulation, self).__init__(config)
        self.add_body_force((1e-5, 0.0, 0.0))


if __name__ == '__main__':
    LBSimulationController(ExternalSimulation).run()

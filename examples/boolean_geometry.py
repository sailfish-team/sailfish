#!/usr/bin/python -u
"""Demonstrates how to load geometry from a boolean numpy array."""

import sys
import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.geo_block import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class BoolSubdomain(Subdomain3D):
    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

    def boundary_conditions(self, hx, hy, hz):
        if hasattr(self.config, 'wall_map'):
            x0 = np.min(hx)
            x1 = np.max(hx)
            y0 = np.min(hy)
            y1 = np.max(hy)
            z0 = np.min(hz)
            z1 = np.max(hz)

            partial_wall_map = self.config.wall_map[z0:z1+1, y0:y1+1, x0:x1+1]
            self.set_node(partial_wall_map, self.NODE_WALL)


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

        wall_map = np.logical_not(np.load(config.geometry))
        orig_shape = wall_map.shape
        shape = list(wall_map.shape)

        # Add walls around the whole simulation domain.
        shape[0] += 2
        shape[1] += 2
        shape[2] += 2

        # Perform funny gymnastics to extend the array to the new shape.  numpy's
        # resize can only be used on the 1st axis if the position of the data in
        # the array is not to be changed.
        wall_map = np.resize(wall_map, (shape[0], orig_shape[1], orig_shape[2]))
        wall_map = np.rollaxis(wall_map, 1, 0)
        wall_map = np.resize(wall_map, (shape[1], shape[0], orig_shape[2]))
        wall_map = np.rollaxis(wall_map, 0, 2)
        wall_map = np.rollaxis(wall_map, 2, 0)
        wall_map = np.resize(wall_map, (shape[2], shape[0], shape[1]))
        wall_map = np.rollaxis(wall_map, 0, 3)

        wall_map[:,:,-2:] = True
        wall_map[:,-2:,:] = True
        wall_map[-2:,:,:] = True

        # Make sure the walls are _around_ the simulation domain.
        wall_map = np.roll(wall_map, 1, 0)
        wall_map = np.roll(wall_map, 1, 1)
        wall_map = np.roll(wall_map, 1, 2)

        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape
        config._wall_map = wall_map


if __name__ == '__main__':
    LBSimulationController(BoolSimulation, LBGeometry3D).run()

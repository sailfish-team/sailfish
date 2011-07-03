#!/usr/bin/python

import math
import numpy as np
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D, GeoBlock2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim

class LDCGeometry(LBGeometry2D):
    def blocks(self, n=None):
        blocks = []
        bps = int(math.sqrt(self.config.blocks))

        # Special case.
        if self.config.blocks == 3:
            w1 = self.gx / 2
            w2 = self.gx - w1
            h1 = self.gy / 2
            h2 = self.gy - h1

            blocks.append(LBBlock2D((0, 0), (w1, h1)))
            blocks.append(LBBlock2D((0, h1), (w1, h2)))
            blocks.append(LBBlock2D((w1, 0), (w2, self.gy)))
            return blocks

        if bps**2 != self.config.blocks:
            print ('Only configurations with '
                    'square-of-interger numbers of blocks are supported. '
                    'Falling back to {} x {} blocks.'.format(bps, bps))

        yq = self.gy / bps
        ydiff = self.gy % bps
        xq = self.gx / bps
        xdiff = self.gx % bps

        for i in range(0, bps):
            xsize = xq
            if i == bps:
                xsize += xdiff

            for j in range(0, bps):
                ysize = yq
                if j == bps:
                    ysize += ydiff

                blocks.append(LBBlock2D((i * xq, j * yq), (xsize, ysize)))

        return blocks


class LDCBlock(GeoBlock2D):
    """2D Lid-driven cavity geometry."""

    max_v = 0.1

    def _define_nodes(self, hx, hy):
        wall_map = np.logical_or(
                np.logical_or(hx == self.gx-1, hx == 0), hy == 0)

        self.set_geo(hy == self.gy-1, self.NODE_VELOCITY, (self.max_v, 0.0))
        self.set_geo(wall_map, self.NODE_WALL)

    def _init_fields(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[hy == self.gy-1] = self.max_v


class LDCSim(LBFluidSim, LBForcedSim):
    geo = LDCBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256})

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--blocks', type=int, default=1, help='number of blocks to use')


if __name__ == '__main__':
    ctrl = LBSimulationController(LDCSim, LDCGeometry)
    ctrl.run()

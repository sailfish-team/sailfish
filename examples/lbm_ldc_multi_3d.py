#!/usr/bin/python

import numpy as np
from sailfish.geo import LBGeometry3D
from sailfish.geo_block import LBBlock3D, GeoBlock3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class LDCGeometry(LBGeometry3D):
    def blocks(self, n=None):
        blocks = []
        bps = int(self.config.blocks**(1.0/3))

        if bps**3 != self.config.blocks:
            print ('Only configurations with '
                    'a third power of an integer number of blocks are '
                    'supported.  Falling back to {0} x {0} blocks.'.
                    format(bps))

        xq = self.gx / bps
        xd = self.gx % bps
        yq = self.gy / bps
        yd = self.gy % bps
        zq = self.gz / bps
        zd = self.gz % bps

        for i in range(0, bps):
            xsize = xq
            if i == bps:
                xsize += xd
            for j in range(0, bps):
                ysize = yq
                if j == bps:
                    ysize += yd
                for k in range(0, bps):
                    zsize = zq
                    if k == bps:
                        zsize += zd
                    blocks.append(LBBlock3D((i * xq, j * yq, k * zq),
                                (xsize, ysize, zsize)))
        return blocks


class LDCBlock(GeoBlock3D):
    """3D Lid-driven geometry."""

    max_v = 0.05

    def _define_nodes(self, hx, hy, hz):
        wall_map = np.logical_or(hz == 0, np.logical_or(
                np.logical_or(hx == self.gx-1, hx == 0),
                np.logical_or(hy == self.gy-1, hy == 0)))

        self.set_geo(wall_map, self.NODE_WALL)
        self.set_geo(hz == self.gz-1, self.NODE_VELOCITY,
                (self.max_v, 0.0, 0.0))

    def _init_fields(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vx[hz == self.gz-1] = self.max_v


class LDCSim(LBFluidSim, LBForcedSim):
    geo = LDCBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 128,
            'lat_ny': 128,
            'lat_nz': 128,
            'grid': 'D3Q19'})

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--blocks', type=int, default=1, help='number of blocks to use')


if __name__ == '__main__':
    ctrl = LBSimulationController(LDCSim, LDCGeometry)
    ctrl.run()

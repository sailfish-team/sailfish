#!/usr/bin/python

import numpy as np
from sailfish.geo import LBGeometry3D
from sailfish.geo_block import SubdomainSpec3D, Subdomain3D
from sailfish.geo_block import NTFullBBWall, NTEquilibriumVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim


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
            if i == bps - 1:
                xsize += xd
            for j in range(0, bps):
                ysize = yq
                if j == bps - 1:
                    ysize += yd
                for k in range(0, bps):
                    zsize = zq
                    if k == bps - 1:
                        zsize += zd
                    blocks.append(SubdomainSpec3D((i * xq, j * yq, k * zq),
                                (xsize, ysize, zsize)))
        return blocks


class LDCBlock(Subdomain3D):
    """3D Lid-driven geometry."""

    max_v = 0.05

    def boundary_conditions(self, hx, hy, hz):
        wall_bc = NTFullBBWall
        velocity_bc = NTEquilibriumVelocity

        lor = np.logical_or
        land = np.logical_and
        lnot = np.logical_not

        wall_map = land(lor(hz == 0, lor(lor(hx == self.gx - 1, hx == 0),
                        lor(hy == self.gy - 1, hy == 0))), lnot(hz == self.gz - 1))

        self.set_node(wall_map, wall_bc)
        self.set_node(hz == self.gz - 1, velocity_bc((self.max_v, 0.0, 0.0)))

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vx[hz == self.gz - 1] = self.max_v


class LDCSim(LBFluidSim):
    subdomain = LDCBlock

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

        group.add_argument('--blocks', type=int, default=1, help='number of blocks to use')


if __name__ == '__main__':
    ctrl = LBSimulationController(LDCSim, LDCGeometry)
    ctrl.run()

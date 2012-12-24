#!/usr/bin/env python

import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.node_type import NTFullBBWall
from examples.binary_fluid.sc_separation_3d import SeparationDomain, SeparationSCSim

class SeparationDomainWithWalls(SeparationDomain):
    def boundary_conditions(self, hx, hy, hz):
        self.set_node(
            (hx == 0) | (hy == 0) | (hz == 0) |
            (hx == self.gx - 1) | (hy == self.gy - 1) | (hz == self.gz - 1),
            NTFullBBWall)


class SeparationSCSim2(SeparationSCSim):
    subdomain = SeparationDomainWithWalls

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 192,
            'lat_ny': 192,
            'lat_nz': 192,
            'grid': 'D3Q19',
            'G': 1.2,
            'visc': 1.0/6.0,
            'periodic_x': False,
            'periodic_y': False,
            'periodic_z': False})


if __name__ == '__main__':
    ctrl = LBSimulationController(SeparationSCSim2, LBGeometry3D)
    ctrl.run()

#!/usr/bin/python

import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.geo_block import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.lb_single import LBForcedSim

class SeparationDomain(Subdomain3D):
    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0 + np.random.rand(*sim.rho.shape) / 1000.0
        sim.phi[:] = 1.0 + np.random.rand(*sim.phi.shape) / 1000.0

    def boundary_conditions(self, hx, hy, hz):
        pass


class SeparationSCSim(LBBinaryFluidShanChen, LBForcedSim):
    subdomain = SeparationDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 192,
            'lat_ny': 192,
            'lat_nz': 192,
            'grid': 'D3Q19',
            'G': 1.2,
            'visc': 1.0/6.0,
            'periodic_x': True,
            'periodic_y': True,
            'periodic_z': True})


if __name__ == '__main__':
    ctrl = LBSimulationController(SeparationSCSim, LBGeometry3D)
    ctrl.run()

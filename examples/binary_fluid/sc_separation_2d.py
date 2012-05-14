#!/usr/bin/python

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.lb_single import LBForcedSim

class SeparationDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0 + np.random.rand(*sim.rho.shape) / 1000.0
        sim.phi[:] = 1.0 + np.random.rand(*sim.phi.shape) / 1000.0

    def boundary_conditions(self, hx, hy):
        pass

class SeparationSCSim(LBBinaryFluidShanChen, LBForcedSim):
    subdomain = SeparationDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'grid': 'D2Q9',
            'G': 1.2,
            'visc': 1.0/6.0,
            'periodic_x': True,
            'periodic_y': True})


if __name__ == '__main__':
    ctrl = LBSimulationController(SeparationSCSim, LBGeometry2D)
    ctrl.run()

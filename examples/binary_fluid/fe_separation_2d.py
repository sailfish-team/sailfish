#!/usr/bin/python

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.geo_block import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidFreeEnergy
from sailfish.lb_single import LBForcedSim


class SeparationDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        self.rho[:] = 1.0
        self.phi[:] = np.random.rand(*self.phi.shape) / 100.0

    def boundary_conditions(self, hx, hy):
        pass


class SeparationFESim(LBBinaryFluidFreeEnergy, LBForcedSim):
    subdomain = SeparationDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'grid': 'D2Q9',
            'kappa': 2e-4,
            'Gamma': 25.0,
            'A': 1e-4,
            'tau_a': 4.5,
            'tau_b': 0.8,
            'tau_phi': 1.0,
            'periodic_x': True,
            'periodic_y': True})


if __name__ == '__main__':
    ctrl = LBSimulationController(SeparationFESim, LBGeometry2D)
    ctrl.run()

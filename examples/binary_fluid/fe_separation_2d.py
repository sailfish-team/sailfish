#!/usr/bin/env python

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidFreeEnergy


class SeparationDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.phi[:] = np.random.rand(*sim.phi.shape) / 100.0

    def boundary_conditions(self, hx, hy):
        pass


class SeparationFESim(LBBinaryFluidFreeEnergy):
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

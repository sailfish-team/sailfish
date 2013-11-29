#!/usr/bin/env python

import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidFreeEnergy
from sailfish.node_type import NTFullBBWall, _NTUnused


class SeparationDomain(Subdomain3D):
    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.phi[:] = np.random.rand(*sim.phi.shape) / 100.0

    def boundary_conditions(self, hx, hy, hz):
        pass

class SeparationFESim(LBBinaryFluidFreeEnergy):
    subdomain = SeparationDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 32,
            'lat_ny': 32,
            'lat_nz': 32,
            'grid': 'D3Q19',
            'kappa': 2e-4,
            'Gamma': 25.0,
            'A': 1e-4,
            'tau_a': 4.5,
            'tau_b': 0.8,
            'tau_phi': 1.0,
            'periodic_x': True,
            'periodic_z': True,
            'periodic_y': True})


if __name__ == '__main__':
    ctrl = LBSimulationController(SeparationFESim, LBGeometry3D)
    ctrl.run()

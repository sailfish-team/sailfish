#!/usr/bin/python

import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidFreeEnergy
from sailfish.lb_base import LBForcedSim


class FingeringDomain(Subdomain3D):
    def initial_conditions(self, sim, hx, hy, hz):
        a = 50.0 - 8.0 * np.cos(2.0 * np.pi * hy / self.gy)
        b = 100.0 - 8.0 * np.cos(2.0 * np.pi * hy / self.gy)

        sim.rho[:] = 1.0
        sim.phi[:] = 1.0
        sim.phi[np.logical_or(hx <= a, hx >= b)] = -1.0

    def boundary_conditions(self, hx, hy, hz):
        self.set_node(np.logical_or(hz == 0, hz == self.gz - 1), NTFullBBWall)


class FingeringFESim(LBBinaryFluidFreeEnergy, LBForcedSim):
    subdomain = FingeringDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 320,
            'lat_ny': 101,
            'lat_nz': 37,
            'grid': 'D3Q19',
            'tau_a': 4.5,
            'tau_b': 0.6,
            'tau_phi': 1.0,
            'kappa': 9.18e-5,
            'Gamma': 25.0,
            'A': 1.41e-4,
            'model': 'mrt',
            'periodic_x': True,
            'periodic_y': True,
            'periodic_z': True})

    def __init__(self, config):
        super(FingeringFESim, self).__init__(config)

        self.add_body_force((3.0e-5, 0.0, 0.0), grid=0, accel=False)

        # Use the fluid velocity in the relaxation of the order parameter field,
        # and the molecular velocity in the relaxation of the density field.
        self.use_force_for_equilibrium(None, target_grid=0)
        self.use_force_for_equilibrium(0, target_grid=1)


if __name__ == '__main__':
    ctrl = LBSimulationController(FingeringFESim, LBGeometry3D)
    ctrl.run()

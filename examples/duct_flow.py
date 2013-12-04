#!/usr/bin/env python
# coding=utf-8
"""
Velocity-driven flow through a duct of rectangular cross-section.

Geometry of the flow is the same as:

S. S. Chikatamarla1, S. Ansumali and I. V. Karlin
Grad’s approximation for missing data in lattice Boltzmann simulations
Europhys. Lett., 74 (2), pp. 215–221 (2006)

and the simulation output can be used for the inflow profile of a
backward-facing step simulation.

TODO: Compare the results with the analytical solution for rectangular
duct flow from F. M. White, "Viscous Fluid Flow", 2nd, p. 120, Eq. 3.48:

    -a <= y <= a
    -b <= z <= b

    u(y, z) = \frac{16 a^2}{\mu \pi^3} \left(- \frac{dp}{dx} \right) \sum_{i =
        1,3,5,...}^{\infty} (-1)^{(i-2)/2} \left(1 - \frac{\cosh(i \pi z / 2a)}{\cosh({i
        \pi b / 2a)}} \right) \frac{\cos(i \pi y/2a)}{i^3}
"""

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity, NTGradFreeflow

class DuctSubdomain(Subdomain3D):
    max_v = 0.02

    def boundary_conditions(self, hx, hy, hz):
        wall_map = (hz == 0) | (hz == self.gz - 1) | (hx == 0) | (hx == self.gx - 1)
        inflow_map = np.logical_not(wall_map) & (hy == 0)
        outflow_map = np.logical_not(wall_map) & (hy == self.gy - 1)

        self.set_node(wall_map, NTFullBBWall)
        self.set_node(inflow_map, NTEquilibriumVelocity((0.0, self.max_v, 0.0)))
        self.set_node(outflow_map, NTGradFreeflow)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vy[:] = self.max_v

class DuctSim(LBFluidSim):
    subdomain = DuctSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        S = 10
        defaults.update({
            'lat_nx': 36 * S + 2,
            'lat_ny': 15 * S,
            'lat_nz': 1 * S + 2,
            'grid': 'D3Q15',
            'visc': 0.01,
            'max_iters': 100000,
            'every': 10000,
            'output': 'duct',
            'checkpoint_file': 'duct.cpoint',
            'final_checkpoint': True,
            'checkpoint_every': 1000000000})

if __name__ == '__main__':
    ctrl = LBSimulationController(DuctSim, EqualSubdomainsGeometry3D)
    ctrl.run()

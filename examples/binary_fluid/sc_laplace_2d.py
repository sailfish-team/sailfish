#!/usr/bin/env python
"""
Simulation of a spherical drop. This can be used to measure surface tension /
verify Laplace's law:

  \Delta p = \sigma / r
"""
import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTFullBBWall, _NTUnused
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.sym import relaxation_time

H = 128
R = 20
visc1 = 1.0 / 3.0
visc2 = visc1
G = 3.5


class PoiseuilleDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        drop = ((hx - H / 2)**2 + (hy - H / 2)**2) < R**2
        not_drop = np.logical_not(drop)

        sim.rho[drop] = 1.0
        sim.phi[drop] = 1e-4

        sim.rho[not_drop] = 1e-4
        sim.phi[not_drop] = 1.0

    def boundary_conditions(self, hx, hy):
        pass

class PoiseuilleSim(LBBinaryFluidShanChen):
    subdomain = PoiseuilleDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': H,
            'lat_ny': H,
            'grid': 'D2Q9',
            'visc': visc1,
            'tau_phi': relaxation_time(visc2),
            'force_implementation': 'edm',
            'G': G,
            'periodic_y': True,
            'periodic_x': True,
        })

    def after_step(self, runner):
        every = 100

        if self.iteration % every == every - 1:
            self.need_sync_flag = True

        if self.iteration % every == 0:
            phi1 = runner._sim.phi[H/2, H/2]
            rho1 = runner._sim.rho[H/2, H/2]

            phi2 = runner._sim.phi[10, 10]
            rho2 = runner._sim.rho[10, 10]

            p1 = (phi1 + rho1) + G * phi1 * rho1
            p2 = (phi2 + rho2) + G * phi2 * rho2

            print p1 - p2, p1, np.nansum(runner._sim.phi), np.nansum(runner._sim.rho)

if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim).run()

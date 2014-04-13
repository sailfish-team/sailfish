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
from sailfish.lb_multi import LBMultiFluidShanChen
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

        sim.g0m0[drop] = 1.0
        sim.g1m0[drop] = 1e-4

        sim.g0m0[not_drop] = 1e-4
        sim.g1m0[not_drop] = 1.0

    def boundary_conditions(self, hx, hy):
        pass

class PoiseuilleSim(LBMultiFluidShanChen):
    subdomain = PoiseuilleDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 2,
            'lat_nx': H,
            'lat_ny': H,
            'grid': 'D2Q9',
            'tau0': relaxation_time(visc1),
            'tau1': relaxation_time(visc2),
            'force_implementation': 'edm',
            'G01': G,
            'periodic_y': True,
            'periodic_x': True,
        })

    def after_step(self, runner):
        every = 100

        if self.iteration % every == every - 1:
            self.need_sync_flag = True

        if self.iteration % every == 0:
            g1m0a = runner._sim.g1m0[H/2, H/2]
            g0m0a = runner._sim.g0m0[H/2, H/2]

            g1m0b = runner._sim.g1m0[10, 10]
            g0m0b = runner._sim.g0m0[10, 10]

            p1 = (g1m0a + g0m0a) + G * g1m0a * g0m0a
            p2 = (g1m0b + g0m0b) + G * g1m0b * g0m0b

            print p1 - p2, p1, np.nansum(runner._sim.g1m0), np.nansum(runner._sim.g0m0)

if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim).run()

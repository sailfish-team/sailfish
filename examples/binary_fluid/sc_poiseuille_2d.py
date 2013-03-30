#!/usr/bin/env python

"""Poiseuille flow with two fluid species in the channel."""

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTFullBBWall, _NTUnused
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.sym import relaxation_time

H = 256
max_v = 0.05
visc2 = 0.16666666666666666
visc1 = visc2 / 5.0

class PoiseuilleDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        core = (hx > H / 4) & (hx <= 3 * H / 4)
        sim.rho[core] = 1.0
        sim.phi[core] = 1e-6

        boundary = np.logical_not(core)
        sim.rho[boundary] = 1e-6
        sim.phi[boundary] = 1.0

    def boundary_conditions(self, hx, hy):
        self.set_node((hx == 1) | (hx == self.gx - 2), NTHalfBBWall)
        self.set_node((hx == 0) | (hx == self.gx - 1), _NTUnused)


class PoiseuilleSim(LBBinaryFluidShanChen):
    subdomain = PoiseuilleDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': H + 2,
            'lat_ny': H / 4,
            'grid': 'D2Q9',
            'visc': visc1,
            'tau_phi': relaxation_time(visc2),
            'force_implementation': 'edm',
            'G': 1.2,
            'periodic_y': True,
        })

    def __init__(self, config):
        super(PoiseuilleSim, self).__init__(config)

        accel = max_v * 32.0 / H**2 / (3.0 / visc2 + 1.0 / visc1)
        self.add_body_force((0.0, accel))
        self.add_body_force((0.0, accel), grid=1)

if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim).run()

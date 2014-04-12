#!/usr/bin/env python

"""Poiseuille flow with two fluid species in the channel."""

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTFullBBWall, _NTUnused
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen
from sailfish.sym import relaxation_time

H = 256
max_v = 0.05
visc1 = 0.16666666666666666
visc0 = visc1 / 5.0

class PoiseuilleDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        core = (hx > H / 4) & (hx <= 3 * H / 4)
        sim.g0m0[core] = 1.0
        sim.g1m0[core] = 1e-6

        boundary = np.logical_not(core)
        sim.g0m0[boundary] = 1e-6
        sim.g1m0[boundary] = 1.0

    def boundary_conditions(self, hx, hy):
        self.set_node((hx == 1) | (hx == self.gx - 2), NTHalfBBWall)
        self.set_node((hx == 0) | (hx == self.gx - 1), _NTUnused)


class PoiseuilleSim(LBMultiFluidShanChen):
    subdomain = PoiseuilleDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 2,
            'lat_nx': H + 2,
            'lat_ny': H / 4,
            'grid': 'D2Q9',
            'visc0': visc0,
            'visc1': visc1,
            'force_implementation': 'edm',
            'G01': 1.2,
            'periodic_y': True,
        })

    def __init__(self, config):
        super(PoiseuilleSim, self).__init__(config)

        accel = max_v * 32.0 / H**2 / (3.0 / visc1 + 1.0 / visc0)
        self.add_body_force((0.0, accel))
        self.add_body_force((0.0, accel), grid=1)

if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim).run()

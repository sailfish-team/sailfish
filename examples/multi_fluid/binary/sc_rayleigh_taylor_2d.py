#!/usr/bin/env python

import time
import numpy as np

from sailfish import sym
from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen
from sailfish.lb_base import LBForcedSim

class RayleighTaylorDomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        self.set_node(np.logical_or(hy == 0, hy == self.gy - 1),
                NTFullBBWall)

    def initial_conditions(self, sim, hx, hy):
        sim.g0m0[:] = np.random.rand(*sim.g0m0.shape) / 100.0
        sim.g1m0[:] = np.random.rand(*sim.g1m0.shape) / 100.0

        sim.g0m0[(hy <= self.gy / 2)] += 1.0
        sim.g1m0[(hy <= self.gy / 2)] = 1e-4

        sim.g0m0[(hy > self.gy / 2)] = 1e-4
        sim.g1m0[(hy > self.gy / 2)] += 1.0


class RayleighTaylorSCSim(LBMultiFluidShanChen, LBForcedSim):
    subdomain = RayleighTaylorDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 2,
            'lat_nx': 640,
            'lat_ny': 400,
            'grid': 'D2Q9',
            'G01': 1.2,
            'visc0': 1.0/6.0,
            'visc1': 1.0/6.0,
            'periodic_x': True})

    def __init__(self, config):
        super(RayleighTaylorSCSim, self).__init__(config)
        self.add_body_force((0.0, -0.15 / config.lat_ny), grid=1)


if __name__ == '__main__':
    ctrl = LBSimulationController(RayleighTaylorSCSim, LBGeometry2D)
    ctrl.run()


#!/usr/bin/python

import numpy as np

from sailfish import sym
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.lb_single import LBForcedSim

class RayleighTaylorDomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        self.set_node(np.logical_or(hy == 0, hy == self.gy - 1),
                self.NODE_WALL)

    def initial_conditions(self, sim, hx, hy):
        self.rho[:] = np.random.rand(*self.rho.shape) / 100.0
        self.phi[:] = np.random.rand(*self.phi.shape) / 100.0

        self.rho[(hy <= self.gy / 2)] += 1.0
        self.phi[(hy <= self.gy / 2)] = 1e-4

        self.rho[(hy > self.gy / 2)] = 1e-4
        self.phi[(hy > self.gy / 2)] += 1.0


class RayleighTaylorSCSim(LBBinaryFluidShanChen, LBForcedSim):
    subdomain = RayleighTaylorDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 640,
            'lat_ny': 400,
            'grid': 'D2Q9',
            'G': 1.2,
            'visc': 1.0 / 6.0,
            'periodic_x': False})

    @classmethod
    def modify_config(cls, config):
        config.tau_phi = sym.relaxation_time(config.visc)

    def __init__(self, config):
        super(RayleighTaylorSCSim, self).__init__(config)
        self.add_body_force((0.0, -0.15 / config.lat_ny), grid=1)


if __name__ == '__main__':
    ctrl = LBSimulationController(RayleighTaylorSCSim, LBGeometry2D)
    ctrl.run()


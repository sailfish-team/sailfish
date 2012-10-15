#!/usr/bin/python
# TODO: verify time evolution by comparing to the known solution (factor out
# code from intial_conditiosn)

import math
import numpy as np

from sailfish.geo import EqualSubdomainsGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim


class TaylorGreenSubdomain(Subdomain2D):
    max_v = 0.05

    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        Ma = self.max_v / math.sqrt(sim.grid.cssq)

        kx = np.pi * 2.0 / (self.config.lambda_x * self.gx)
        ky = np.pi * 2.0 / (self.config.lambda_y * self.gy)

        ksq = np.square(kx) + np.square(ky)

        t = 0
        f = np.exp(-self.config.visc * ksq * t)

        sim.vx[:] = -self.max_v * ky / np.sqrt(ksq) * f * np.sin(ky * hy) * np.cos(kx * hx)
        sim.vy[:] =  self.max_v * kx / np.sqrt(ksq) * f * np.sin(kx * hx) * np.cos(ky * hy)
        sim.rho[:] = 1.0 - Ma**2 / (ksq * 2.0) * (
                ky**2 * np.cos(2 * kx * hx) +
                kx**2 * np.cos(2 * ky * hy))


class TaylorGreenSim(LBFluidSim):
    subdomain = TaylorGreenSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'periodic_x': True,
            'periodic_y': True,
            'lat_nx': 256,
            'lat_ny': 256,
            'visc': 0.001
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)

        group.add_argument('--lambda_x', type=int, default=1)
        group.add_argument('--lambda_y', type=int, default=1)


if __name__ == '__main__':
    ctrl = LBSimulationController(TaylorGreenSim, EqualSubdomainsGeometry2D)
    ctrl.run()

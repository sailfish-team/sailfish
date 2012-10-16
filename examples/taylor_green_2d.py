#!/usr/bin/python

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
        t = 0
        Ma = self.max_v / math.sqrt(sim.grid.cssq)
        rho, vx, vy = self.solution(self.config, hx, hy, self.gx, self.gy, t, Ma)
        sim.rho[:] = rho
        sim.vx[:] = vx
        sim.vy[:] = vy

    @classmethod
    def solution(cls, config, hx, hy, gx, gy, t, Ma):
        """Returns the analytical solution of the 2D Taylor Green flow for time t."""
        kx = np.pi * 2.0 / (config.lambda_x * gx)
        ky = np.pi * 2.0 / (config.lambda_y * gy)

        ksq = np.square(kx) + np.square(ky)
        f = np.exp(-config.visc * ksq * t)

        vx = -cls.max_v * ky / np.sqrt(ksq) * f * np.sin(ky * hy) * np.cos(kx * hx)
        vy =  cls.max_v * kx / np.sqrt(ksq) * f * np.sin(kx * hx) * np.cos(ky * hy)
        rho = 1.0 - Ma**2 / (ksq * 2.0) * (
            ky**2 * np.cos(2 * kx * hx) +
            kx**2 * np.cos(2 * ky * hy))
        return rho, vx, vy


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

    def after_step(self, runner):
        every = 200

        if self.iteration % (every - 1) == 0:
            self.need_sync_flag = True
        elif self.iteration % every == 0:
            u = np.sqrt(runner._sim.vx**2 + runner._sim.vy**2)
            ny, nx = runner._sim.vx.shape
            hy, hx = np.mgrid[0:ny, 0:nx]

            Ma = self.subdomain.max_v / math.sqrt(runner._sim.grid.cssq)
            _, ref_ux, ref_uy = self.subdomain.solution(self.config, hx, hy,
                                                        nx, ny,
                                                        self.iteration, Ma)
            ref_u = np.sqrt(ref_ux**2 + ref_uy**2)

            du_norm = np.linalg.norm(u - ref_u) / u.size
            print du_norm


if __name__ == '__main__':
    ctrl = LBSimulationController(TaylorGreenSim, EqualSubdomainsGeometry2D)
    ctrl.run()

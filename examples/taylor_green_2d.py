#!/usr/bin/env/python

import math
import numpy as np
from sailfish.geo import EqualSubdomainsGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim


class TaylorGreenSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        t = 0
        Ma = self.config.max_v / math.sqrt(sim.grid.cssq)
        rho, vx, vy = self.solution(self.config, hx, hy, self.gx, self.gy, t, Ma)
        sim.rho[:] = rho
        sim.vx[:] = vx
        sim.vy[:] = vy

    @classmethod
    def get_k(cls, config, gx, gy):
        kx = np.pi * 2.0 / (config.lambda_x * gx)
        ky = np.pi * 2.0 / (config.lambda_y * gy)
        ksq = np.square(kx) + np.square(ky)
        return kx, ky, ksq

    @classmethod
    def solution(cls, config, hx, hy, gx, gy, t, Ma):
        """Returns the analytical solution of the 2D Taylor Green flow for time t."""
        kx, ky, ksq = cls.get_k(config, gx, gy)
        f = np.exp(-config.visc * ksq * t)

        vx = -config.max_v * ky / np.sqrt(ksq) * f * np.sin(ky * hy) * np.cos(kx * hx)
        vy =  config.max_v * kx / np.sqrt(ksq) * f * np.sin(kx * hx) * np.cos(ky * hy)
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

        group.add_argument('--max_v', type=float, default=0.01)
        group.add_argument('--lambda_x', type=int, default=1)
        group.add_argument('--lambda_y', type=int, default=1)
        group.add_argument('--err_every', type=int, default=100)

    # This is factored out into a separate method so that it can be easily
    # overridden in four_rolls_mill.
    def reference_solution(self, hx, hy, nx, ny, iteration, Ma):
        return self.subdomain.solution(self.config, hx, hy, nx, ny,
                                       iteration, Ma)

    def after_step(self, runner):
        if self.iteration % self.config.err_every == self.config.err_every - 1:
            self.need_sync_flag = True
        elif self.iteration % self.config.err_every == 0:
            vx = runner._sim.vx.astype(np.float64)
            vy = runner._sim.vy.astype(np.float64)

            ny, nx = runner._sim.vx.shape
            hy, hx = np.mgrid[0:ny, 0:nx]

            Ma = self.config.max_v / math.sqrt(runner._sim.grid.cssq)
            _, ref_ux, ref_uy = self.reference_solution(hx, hy, nx, ny,
                                                        self.iteration, Ma)
            top_v = np.max(np.sqrt(np.square(vx) + np.square(vy)))
            top_ref_v = np.max(np.sqrt(np.square(ref_ux) + np.square(ref_uy)))
            dx = ref_ux - vx
            dy = ref_uy - vy
            err = (np.linalg.norm(np.dstack([dx, dy])) /
                   np.linalg.norm(np.dstack([ref_ux, ref_uy])))
            print self.iteration, err, top_v, top_ref_v


if __name__ == '__main__':
    ctrl = LBSimulationController(TaylorGreenSim, EqualSubdomainsGeometry2D)
    ctrl.run()

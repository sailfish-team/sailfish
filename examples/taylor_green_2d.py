#!/usr/bin/env python
"""
2D Taylor-Green vortex flow.

This is a well-known decaying vortex flow with an exact closed form solution:

    u_x = -u_0 cos(x) sin(y) exp(-2 nu t)
    u_y =  u_0 sin(x) cos(y) exp(-2 nu t)
    p = rho / 4 u_0^2 (cos(2x) + cos(2y)) * exp(-4 nu t)

and can be used to test accuracy of LB models e.g. as a function of viscosity
or maximum velocity in the simulation (Mach number).

Compared to the formulas above, the code introduces additional factors kx, ky
which result from the LB -> physical parameter scaling.

While running, the simulation prints to stdout:
    iteration, relative velocity error (simulation result / reference solution),
    max velocity in the simulation, max velocity in the reference solution

To verify that this is a solution of the incompressible Navier-Stokes equations:

    from sympy import *
    x, y, nu, t, u0, rho, p0 = symbols('x y nu t u0 rho p0 kx ky')
    u = -u0 * cos(x) * sin(y) * exp(-2 * nu * t)
    v =  u0 * sin(x) * cos(y) * exp(-2 * nu * t)
    P = p0 - rho / 4 * u0**2 * (cos(2 * x) + cos(2 * y)) * exp(-4 * nu * t)

    dpdx = simplify((diff(u, x, x) + diff(u, y, y)) * nu -
                    (diff(u, t) + u * diff(u, x) + v * diff(u, y)))  # dp/dx / rho
    dpdy = simplify((diff(v, x, x) + diff(v, y, y)) * nu -
                    (diff(v, t) + u * diff(v, x) + v * diff(v, y)))  # dp/dy / rho

    s.expand_trig(s.diff(P, x) / rho) == dpdx
    s.expand_trig(s.diff(P, y) / rho) == dpdy

For the LB solution, we take p0 = 1.0 and use the P = rho/c_s^2 equation of
state to get the form in solution()."""

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
        """Returns scaling factors that transform LB coordinates into
        coordinates from the range [0:2*pi]."""
        kx = np.pi * 2.0 * config.lambda_x / gx
        ky = np.pi * 2.0 * config.lambda_y / gy
        ksq = kx**2 + ky**2
        k = math.sqrt(ksq)
        return kx, ky, ksq, k

    @classmethod
    def solution(cls, config, hx, hy, gx, gy, t, Ma):
        """Returns the analytical solution of the 2D Taylor Green flow for time t."""
        kx, ky, ksq, k = cls.get_k(config, gx, gy)
        f = np.exp(-config.visc * ksq * t)

        vx = -config.max_v * ky / k * f * np.sin(ky * hy) * np.cos(kx * hx)
        vy =  config.max_v * kx / k * f * np.sin(kx * hx) * np.cos(ky * hy)
        rho = 1.0 - Ma**2 / (ksq * 4.0) * (
            ky**2 * np.cos(2 * kx * hx) +
            kx**2 * np.cos(2 * ky * hy)) * f * f
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

        group.add_argument('--max_v', type=float, default=0.01,
                           help='Maximum velocity in LB units.')
        group.add_argument('--lambda_x', type=int, default=1)
        group.add_argument('--lambda_y', type=int, default=1)
        group.add_argument('--err_every', type=int, default=100,
                           help='How often to evaluate the error.')

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
            rho = runner._sim.rho.astype(np.float64)

            ny, nx = runner._sim.vx.shape
            hy, hx = np.mgrid[0:ny, 0:nx]

            Ma = self.config.max_v / math.sqrt(runner._sim.grid.cssq)
            ref_rho, ref_ux, ref_uy = self.reference_solution(hx, hy, nx, ny,
                                                        self.iteration, Ma)
            top_v = np.max(np.sqrt(np.square(vx) + np.square(vy)))
            top_ref_v = np.max(np.sqrt(np.square(ref_ux) + np.square(ref_uy)))
            dx = ref_ux - vx
            dy = ref_uy - vy
            err = (np.linalg.norm(np.dstack([dx, dy])) /
                   np.linalg.norm(np.dstack([ref_ux, ref_uy])))
            rho_err = np.linalg.norm(ref_rho - rho) / np.linalg.norm(ref_rho)
            print self.iteration, err, top_v, top_ref_v, np.max(rho), np.max(ref_rho), rho_err


if __name__ == '__main__':
    ctrl = LBSimulationController(TaylorGreenSim, EqualSubdomainsGeometry2D)
    ctrl.run()

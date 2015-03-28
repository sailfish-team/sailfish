#!/usr/bin/env python
# coding=utf-8
"""
Velocity-driven flow through a duct of rectangular cross-section.

This case has an analytical solution:
(from F. M. White, "Viscous Fluid Flow", 2nd, p. 120, Eq. 3.48):

    -a <= y <= a
    -b <= z <= b

    u(y, z) = \\frac{16 a^2}{\mu \pi^3} \left(- \\frac{dp}{dx} \\right) \sum_{i =
        1,3,5,...}^{\infty} (-1)^{(i-1)/2} \left(1 - \\frac{\cosh(i \pi z / 2a)}{\cosh({i
        \pi b / 2a)}} \\right) \\frac{\cos(i \pi y/2a)}{i^3}
"""

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_base import LBForcedSim
from sailfish.lb_single import LBFluidSim
from sailfish.node_type import NTFullBBWall, NTHalfBBWall

class DuctSubdomain(Subdomain3D):
    max_v = 0.02
    wall_bc = NTHalfBBWall

    def boundary_conditions(self, hx, hy, hz):
        wall_map = (hx == 0) | (hx == self.gx - 1) | (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, self.wall_bc)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vz[:] = self.analytical(hx, hy)

    @classmethod
    def width(cls, config):
        return config.lat_ny - 1 - 2 * cls.wall_bc.location

    @classmethod
    def accel(cls, config):
        # The maximum velocity is attained at the center of the channel,
        # i.e. x = y = 0.
        ii = np.arange(1, 100, 2)
        ssum = np.sum((-1)**((ii - 1)/2.0) * (1 - np.cosh(0) / np.cosh(ii * np.pi / 2)) *
                      np.cos(0) / ii**3)
        a = cls.width(config) / 2.0
        prefactor = 16 * a**2 / (config.visc * np.pi**3)
        return cls.max_v / (prefactor * ssum)

    def analytical(self, hx, hy):
        a = self.width(self.config) / 2.0
        hy = hy - self.wall_bc.location
        hx = hx - self.wall_bc.location
        ry = np.abs(a - hy)
        rx = np.abs(a - hx)

        prefactor = 16 * a**2 / (self.config.visc * np.pi**3)
        ii = np.arange(1, 100, 2)
        ret = np.zeros_like(hy)
        for i in ii:
            ret += ((-1)**((i - 1)/2.0) *
                (1 - np.cosh(i * np.pi * rx / (2.0 * a)) /
                     np.cosh(i * np.pi / 2)) *
                np.cos(i * np.pi * ry / (2.0 * a)) / i**3)

        return self.accel(self.config) * prefactor * ret


class DuctSim(LBFluidSim, LBForcedSim):
    subdomain = DuctSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 64,
            'lat_ny': 64,
            'lat_nz': 32,
            'grid': 'D3Q19',
            'visc': 0.0064,
            'max_iters': 500000,
            'every': 50000,
            'periodic_z': True,
            'block_size': 64,
            # Double precision is required for convergence studies.
            'precision': 'double'
        })

    def __init__(self, config):
        super(DuctSim, self).__init__(config)
        # The flow is driven by a body force.
        self.add_body_force((0.0, 0.0, DuctSubdomain.accel(config)))

        a = DuctSubdomain.width(config) / 2.0
        self.config.logger.info('Re = %.2f, width = %d' % (a * DuctSubdomain.max_v / config.visc, a))

    _ref = None
    _prev_l2 = 0.0
    def after_step(self, runner):
        every = 1000
        mod = self.iteration % every

        if mod == every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            # Cache the reference solution.
            if self._ref is None:
                hx, hy, hz = runner._subdomain._get_mgrid()
                self._ref = runner._subdomain.analytical(hx, hy)

            # Output data useful for monitoring the state of the simulation.
            m = runner._output._fluid_map
            l2 = (np.linalg.norm(self._ref[m] - runner._sim.vz[m]) /
                  np.linalg.norm(self._ref[m]))

            self.config.logger.info('%d %e %e' % (self.iteration, l2, np.nanmax(runner._sim.vz)))

            if (np.abs(self._prev_l2 - l2) / l2 < 1e-6):
                runner._quit_event.set()

            self._prev_l2 = l2

if __name__ == '__main__':
    ctrl = LBSimulationController(DuctSim, EqualSubdomainsGeometry3D)
    ctrl.run()

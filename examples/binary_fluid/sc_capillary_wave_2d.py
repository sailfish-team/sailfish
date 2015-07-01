#!/usr/bin/env python

"""Capillary wave.

Two components are initially are separated by an interface perturbed with a
sinusoidal wave. The code simulates free relaxation of the interface.
"""

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTFullBBWall, _NTUnused
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen
from sailfish.sym import relaxation_time

import scipy.ndimage.filters
import scipy.optimize
import scipy.interpolate

W = 512
H = 512
visc2 = 1.0 / 18.0
visc1 = visc2 #/ 4.0
n_waves = 16
A = 10

class CapillaryWaveDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        wave = H / 2 + A * np.sin(2.0 * np.pi * hx * n_waves / W)

        # The values here are chosen based on iterative numerical optimization
        # to find an equilibrium between the two components.
        sim.phi[hy < wave] = 0.00341573786772
        sim.rho[hy < wave] = 1.00011520663

        sim.phi[hy >= wave] = 1.00011141574
        sim.rho[hy >= wave] = 0.00341763840659

        import scipy.ndimage.filters
        sim.rho[:] = scipy.ndimage.filters.gaussian_filter(sim.rho, 3)
        sim.phi[:] = scipy.ndimage.filters.gaussian_filter(sim.phi, 3)

    def boundary_conditions(self, hx, hy):
        self.set_node((hy == 1) | (hy == self.gy - 2), NTHalfBBWall)
        self.set_node((hy == 0) | (hy == self.gy - 1), _NTUnused)


class CapillaryWaveSim(LBBinaryFluidShanChen):
    subdomain = CapillaryWaveDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': W,
            'lat_ny': H + 2,
            'grid': 'D2Q9',
            'visc': visc1,
            'tau_phi': relaxation_time(visc2),
            'force_implementation': 'edm',
            'G12': 0.9,
            'periodic_x': True,
            'max_iters': 100000,
        })

    def after_step(self, runner):
        every = 5

        if self.iteration % every == every - 1:
            self.need_sync_flag = True

        if self.iteration % every == 0:
            x = W / n_waves / 4
            m1 = H / 2 - 2 * A
            m2 = H / 2 + 2 * A
            phi = runner._sim.phi[m1:m2, x]
            rho = runner._sim.rho[m1:m2, x]

            phi1 = phi[0]
            phi2 = phi[-1]
            phi_m = (phi1 + phi2) / 2.0

            rho1 = rho[0]
            rho2 = rho[-1]
            rho_m = (rho1 + rho2) / 2.0

            x = np.linspace(0, phi.shape[0] - 1, phi.shape[0])
            f = scipy.interpolate.interp1d(x, phi)
            f_rho = scipy.interpolate.interp1d(x, rho)

            interface1 = m1 + scipy.optimize.brentq(lambda x: f(x) - phi_m, 0, phi.shape[0] - 1)
            interface2 = m1 + scipy.optimize.brentq(lambda x: f_rho(x) - rho_m, 0, phi.shape[0] - 1)

            self.config.logger.info('iface %e %e' % (interface1, interface2))


if __name__ == '__main__':
    LBSimulationController(CapillaryWaveSim).run()

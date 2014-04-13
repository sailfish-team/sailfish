#!/usr/bin/env python

"""Capillary wave.

Two components are initially separated by an interface perturbed with a
sinusoidal wave. The code simulates free relaxation of the interface.
"""

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTFullBBWall, _NTUnused
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen
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
        sim.g1m0[hy < wave] = 0.00341573786772
        sim.g0m0[hy < wave] = 1.00011520663

        sim.g1m0[hy >= wave] = 1.00011141574
        sim.g0m0[hy >= wave] = 0.00341763840659

        import scipy.ndimage.filters
        sim.g0m0[:] = scipy.ndimage.filters.gaussian_filter(sim.g0m0, 3)
        sim.g1m0[:] = scipy.ndimage.filters.gaussian_filter(sim.g1m0, 3)

    def boundary_conditions(self, hx, hy):
        self.set_node((hy == 1) | (hy == self.gy - 2), NTHalfBBWall)
        self.set_node((hy == 0) | (hy == self.gy - 1), _NTUnused)


class CapillaryWaveSim(LBMultiFluidShanChen):
    subdomain = CapillaryWaveDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 2,
            'lat_nx': W,
            'lat_ny': H + 2,
            'grid': 'D2Q9',
            'visc0': visc1,
            'visc1': visc2,
            'force_implementation': 'edm',
            'G01': 0.9,
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
            g1m0 = runner._sim.g1m0[m1:m2, x]
            g0m0 = runner._sim.g0m0[m1:m2, x]

            g1m0a = g1m0[0]
            g1m0b = g1m0[-1]
            g1m0_m = (g1m0a + g1m0b) / 2.0

            g0m0a = g0m0[0]
            g0m0b = g0m0[-1]
            g0m0_m = (g0m0a + g0m0b) / 2.0

            x = np.linspace(0, g1m0.shape[0] - 1, g1m0.shape[0])
            f = scipy.interpolate.interp1d(x, g1m0)
            f_g0m0 = scipy.interpolate.interp1d(x, g0m0)

            interface1 = m1 + scipy.optimize.brentq(lambda x: f(x) - g1m0_m, 0, g1m0.shape[0] - 1)
            interface2 = m1 + scipy.optimize.brentq(lambda x: f_g0m0(x) - g0m0_m, 0, g1m0.shape[0] - 1)

            print interface1, interface2

if __name__ == '__main__':
    LBSimulationController(CapillaryWaveSim).run()

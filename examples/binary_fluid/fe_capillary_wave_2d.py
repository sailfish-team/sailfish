#!/usr/bin/env python

"""Poiseuille flow with two fluid species in the channel.

Interface thickness = 2\sqrt{2} \sqrt{\kappa/A} should be kept wide to avoid
aliasing effects when measuring the interface positiion.

The envelope of the measured position should fit y = \exp(C - \gamma t).
Potential flow solution is:

    \gamma = 2 \\nu k^2
    \omega = \sqrt{\\frac{sigma k^3}{2 \\rho}}

with:

    k = 2 \pi / \lambda

and \\lambda being the wavelength. The surface tension in the FE model
is \sigma = \sqrt{8 \kappa A / 9}.
"""

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTFullBBWall, _NTUnused
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidFreeEnergy
from sailfish.sym import relaxation_time

import scipy.ndimage.filters
import scipy.optimize
import scipy.interpolate

H = 256
visc2 = 1.0 / 18.0
visc1 = visc2
n_waves = 1
A = 10

class PoiseuilleDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        wave = H / 2 + A * np.sin(2.0 * np.pi * hx * n_waves / H)

        sim.rho[:] = 1.0
        sim.phi[hy < wave] = 1.0
        sim.phi[hy >= wave] = -1.0

    def boundary_conditions(self, hx, hy):
        self.set_node((hy == 1) | (hy == self.gy - 2), NTHalfBBWall)
        self.set_node((hy == 0) | (hy == self.gy - 1), _NTUnused)


class PoiseuilleSim(LBBinaryFluidFreeEnergy):
    subdomain = PoiseuilleDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': H,
            'lat_ny': H + 2,
            'grid': 'D2Q9',
            'tau_a': relaxation_time(visc1),
            'tau_b': relaxation_time(visc2),
            'tau_phi': 1.0,
            # A wider interface helps with aliasing effects when decting the
            # position of the interface.
            'kappa': 0.04,
            'A': 0.02,
            # Keep this low to make the simulation reasonably fast.
            'Gamma': 0.8,
            'periodic_x': True,
            'bc_wall_grad_order': 1,
            'use_link_tags': False,
        })

    phi_log = []

    def after_step(self, runner):
        every = 50

        if self.iteration % every == every - 1:
            self.need_sync_flag = True

        if self.iteration % every == 0:
            x = H / n_waves / 4
            m1 = H / 2 - 2 * A
            m2 = H / 2 + 2 * A
            phi = runner._sim.phi[m1:m2, x]

            x = np.linspace(0, phi.shape[0] - 1, phi.shape[0])
            f = scipy.interpolate.interp1d(x, phi)
            interface = m1 + scipy.optimize.brentq(f, 0, phi.shape[0] - 1)
            print interface

        if self.iteration == 99999:
            np.savez('phi.npz', np.array(self.phi_log))


if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim).run()

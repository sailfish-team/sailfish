#!/usr/bin/env python -u
"""Womersley flow in 3D.

The Womersley flow is an incompressible fluid flow in a straight pipe with
circular cross-section, driven by periodic oscillations in the pressure
gradient betweeen the inlet and the outlet.

This simulation inherits from the 3D Poiseuille flow example in order to
avoid duplicating some code (e.g. setting the pipe geometry).
"""

from math import sqrt
import numpy as np
from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.node_type import NTFullBBWall, NTEquilibriumDensity, DynamicValue
from poiseuille_3d import PoiseuilleSubdomain, PoiseuilleSim
from sailfish.sym import S
from sympy import sin

omega = 0.00005
visc = 0.001

class WomersleySubdomain(PoiseuilleSubdomain):
    max_v = 0.004

    def _set_pressure_bc(self, hx, hy, hz, wall_map):
        pressure_bc = NTEquilibriumDensity
        not_wall = np.logical_not(wall_map)

        if self.config.flow_direction == 'z':
            inlet_map = (hz == 0) & not_wall
            outlet_map = (hz == self.gz - 1) & not_wall
        elif self.config.flow_direction == 'y':
            inlet_map = (hy == 0) & not_wall
            outlet_map = (hy == self.gy - 1) & not_wall
        else:
            inlet_map = (hx == 0) & not_wall
            outlet_map = (hx == self.gx - 1) & not_wall

        pressure = self.pressure_delta * sin(S.time * omega)
        self.set_node(inlet_map, pressure_bc(
            DynamicValue(1.0 + 3.0 * pressure / 2.0)))
        self.set_node(outlet_map, pressure_bc(
            DynamicValue(1.0 - 3.0 * pressure / 2.0)))

        print 'Re = %.2f' % (self.max_v * self.channel_width(self.config) / 2.0 / visc)
        print 'Wo = %.2f' % (self.channel_width(self.config) / 2.0 * sqrt(omega / visc))
        print 'dP = %.8e' % self.pressure_delta

        # The oscillation period (in lattice time units) should be significantly longer
        # than the length of the pipe (in lattice length units) in order for the
        # compressibility effects of LBM to be minimized.
        print 'T = %.2f' % (2 * np.pi / omega)

    def womersley_profile(self, r, t, alpha, omega):
        """Returns the analytical velocity profile for a flow driven by pressure
        gradient oscillation of the form dP = A sin(omega t).

        :param r: normalized radial coordinate
        :param t: time
        :param alpha: Womersley number
        :param omega: oscillation frequency
        :param A: pressure oscillation amplitude
        """

        if r is None:
            D = self.width(self.config)
            # D - 2 is the actual width of the channel; -2 is correct for
            # full-way BB.
            r = np.abs((D - 1.0) / 2.0 - np.linspace(0, D - 1, D)) / (D - 2)

        from scipy.special import jv
        L = self.channel_length(self.config)
        return -np.real((1 - jv(0, 1j**1.5 * alpha * r) / jv(0, 1j**1.5 * alpha)) *
                        exp(1j * omega * t) * A / L / omega)

    @classmethod
    def channel_length(cls, config):
        if config.flow_direction == 'x':
            return config.lat_nx
        elif config.flow_direction == 'y':
            return config.lat_ny
        else:
            return config.lat_nz


class WomersleySim(PoiseuilleSim):
    subdomain = WomersleySubdomain

    @classmethod
    def update_defaults(cls, defaults):
        PoiseuilleSim.update_defaults(defaults)
        defaults.update({
            'drive': 'pressure',
            'grid': 'D3Q19',
            'lat_nx': 256,
            'visc': visc
            })

    def after_step(self, runner):
        every = 100
        mod = self.iteration % every

        if mod == every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            nz, ny, nx = runner._sim.rho.shape
            # Dump data from the middle of the channel to stdout so that the
            # ~pi/2 phase lag between pressure and velocity oscillations can be
            # easily observed.
            print '%d %.8e %.8e %.8e %.8e' % (self.iteration,
                                              runner._sim.vx[nz/2,ny/2,nx/2],
                                              runner._sim.vy[nz/2,ny/2,nx/2],
                                              runner._sim.vz[nz/2,ny/2,nx/2],
                                              runner._sim.rho[nz/2,ny/2,nx/2] - 1.0)

if __name__ == '__main__':
    LBSimulationController(WomersleySim, EqualSubdomainsGeometry3D).run()

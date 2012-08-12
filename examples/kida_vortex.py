#!/usr/bin/python

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim

class KidaSubdomain(Subdomain3D):
    max_v = 0.05

    def boundary_conditions(self, hx, hy, hz):
        pass

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

        sin = np.sin
        cos = np.cos

        x = (hx + self.config.shift_x) * np.pi * 2.0 / self.gx
        y = (hy + self.config.shift_y) * np.pi * 2.0 / self.gy
        z = (hz + self.config.shift_z) * np.pi * 2.0 / self.gz

        sim.vx[:] = self.max_v * sin(x) * (cos(3 * y) * cos(z) - cos(y) * cos(3 * z))
        sim.vy[:] = self.max_v * sin(y) * (cos(3 * z) * cos(x) - cos(z) * cos(3 * x))
        sim.vz[:] = self.max_v * sin(z) * (cos(3 * x) * cos(y) - cos(x) * cos(3 * y))


class KidaSim(LBFluidSim):
    subdomain = KidaSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'periodic_x': True,
            'periodic_y': True,
            'periodic_z': True,
            'lat_nx': 110,
            'lat_ny': 110,
            'lat_nz': 110,
            'grid': 'D3Q15',
            'visc': 0.01,
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)

        # Parameters used to verify phase shift invariance.
        group.add_argument('--shift_x', type=int, default=0)
        group.add_argument('--shift_y', type=int, default=0)
        group.add_argument('--shift_z', type=int, default=0)

    @classmethod
    def modify_config(cls, config):
        print 'Re = {0}'.format(config.lat_nx *
                cls.subdomain.max_v / config.visc)


if __name__ == '__main__':
    ctrl = LBSimulationController(KidaSim, EqualSubdomainsGeometry3D)
    ctrl.run()

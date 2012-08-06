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

        x = hx * np.pi * 2.0 / (self.gx - 1)
        y = hy * np.pi * 2.0 / (self.gy - 1)
        z = hz * np.pi * 2.0 / (self.gz - 1)

        sim.vx[:] = self.max_v * sin(x) * (cos(3 * y) * cos(z) - cos(y) * cos(3 * z))
        sim.vy[:] = self.max_v * sin(y) * (cos(3 * z) * cos(x) - cos(z) * cos(3 * x))
        sim.vz[:] = self.max_v * sin(z) * (cos(3 * x) * cos(y) - cos(x) * cos(3 * y))


class KidaSim(LBFluidSim):
    subdomain = KidaSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 110,
            'lat_ny': 110,
            'lat_nz': 110,
            'grid': 'D3Q15',
            'visc': 0.01,
            })

    def __init__(self, config):
        super(KidaSim, self).__init__(config)
        print 'Re = {0}'.format(config.lat_nx *
                self.subdomain.max_v / config.visc)


if __name__ == '__main__':
    ctrl = LBSimulationController(KidaSim, EqualSubdomainsGeometry3D)
    ctrl.run()

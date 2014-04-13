#!/usr/bin/env python

"""Stationary droplet in ternary system with multiple self-interactions"""

import numpy as np
import matplotlib.pyplot as plt

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen

class DropSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        radius = 32

        drop_map1 = (hx-self.gx/4)**2 + (hy-self.gy/4)**2 <= radius**2
        drop_map2 = (hx-3*self.gx/4)**2 + (hy-3*self.gy/4)**2 <= radius**2

        sim.g0m0[:] = 2.0
        sim.g1m0[:] = 0.02
        sim.g2m0[:] = 0.02

        sim.g0m0[drop_map1] = 0.02
        sim.g1m0[drop_map1] = 0.5
        sim.g2m0[drop_map1] = 0.02

        sim.g0m0[drop_map2] = 0.02
        sim.g1m0[drop_map2] = 0.02
        sim.g2m0[drop_map2] = 2.0

class SCSim(LBMultiFluidShanChen):
    subdomain = DropSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 3,
            'lat_nx': 256,
            'lat_ny': 256,
            'G00': -4.8,
            'G22': -4.8,
            'periodic_x': True,
            'periodic_y': True,
            'sc_potential': 'classic'
            })

    def after_step(self, runner):
        every_n = 10000

        if self.iteration % every_n == every_n - 1:
            self.need_sync_flag = True
        elif self.iteration % every_n == 0:
            rho0 = runner._sim.g0m0.astype(np.float64)
            rho1 = runner._sim.g1m0.astype(np.float64)
            rho2 = runner._sim.g2m0.astype(np.float64)

            ny, nx = runner._sim.g0m0.shape
            hy, hx = np.mgrid[0:ny, 0:nx]

            fig1 = plt.figure(1); fig1.clf();
            plt.plot( hx[hx==hy], rho0[hx==hy], 'r',
                      hx[hx==hy], rho1[hx==hy], 'g',
                      hx[hx==hy], rho2[hx==hy], 'b', )
            plt.legend(['rho0','rho1','rho2'])
            fig1.savefig('profiles.png')

if __name__ == '__main__':
    LBSimulationController(SCSim, LBGeometry2D).run()

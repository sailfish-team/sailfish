#!/usr/bin/env python

"""Stationary droplet in ternary system with multiple self-interactions"""

import numpy as np
import matplotlib.pyplot as plt

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_ternary import LBTernaryFluidShanChen

class DropSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        radius = 32
        
        drop_map1 = (hx-self.gx/4)**2 + (hy-self.gy/4)**2 <= radius**2
        drop_map2 = (hx-3*self.gx/4)**2 + (hy-3*self.gy/4)**2 <= radius**2

        sim.rho[:] = 2.0
        sim.phi[:] = 0.02
        sim.theta[:] = 0.02

        sim.rho[drop_map1] = 0.02
        sim.phi[drop_map1] = 0.5
        sim.theta[drop_map1] = 0.02

        sim.rho[drop_map2] = 0.02
        sim.phi[drop_map2] = 0.02
        sim.theta[drop_map2] = 2.0

class SCSim(LBTernaryFluidShanChen):
    subdomain = DropSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'G11': -4.8,
            'G33': -4.8,
            'visc': 1.0 / 6.0,
            'periodic_x': True,
            'periodic_y': True,
            'sc_potential': 'classic'
            })

    def after_step(self, runner):
        every_n = 10000
        
        if self.iteration % every_n == every_n - 1:
            self.need_sync_flag = True
        elif self.iteration % every_n == 0:
            rho = runner._sim.rho.astype(np.float64)
            phi = runner._sim.phi.astype(np.float64)
            theta = runner._sim.theta.astype(np.float64)

            ny, nx = runner._sim.rho.shape
            hy, hx = np.mgrid[0:ny, 0:nx]

            fig1 = plt.figure(1); fig1.clf();
            plt.plot( hx[hx==hy], rho[hx==hy], 'r',
                      hx[hx==hy], phi[hx==hy], 'g',
                      hx[hx==hy], theta[hx==hy], 'b', )
            plt.legend(['rho','phi','theta'])
            fig1.savefig('profiles.png')

if __name__ == '__main__':
    LBSimulationController(SCSim, LBGeometry2D).run()

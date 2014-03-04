#!/usr/bin/env python

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_ternary import LBTernaryFluidShanChen

class DropSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        drop_map1 = (self.gx/4-hx)**2 + (self.gy/4-hy)**2 <= 32**2
        drop_map2 = (3*self.gx/4-hx)**2 + (3*self.gy/4-hy)**2 <= 32**2
        sim.rho[:] = 2.15
        sim.phi[:] = 0.02
        sim.theta[:] = 0.02
        sim.rho[drop_map1 & drop_map2] = 0.02
        sim.phi[drop_map1] = 0.15
        sim.theta[drop_map2] = 0.15

class SCSim(LBTernaryFluidShanChen):
    subdomain = DropSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'G12': -4.8,
            'G13': -4.8,
            'G23': 1.6,
            'visc': 1.0 / 6.0,
            'periodic_x': True,
            'periodic_y': True,
            'sc_potential': 'classic',
            'every': 20,
            })


if __name__ == '__main__':
    LBSimulationController(SCSim, LBGeometry2D).run()

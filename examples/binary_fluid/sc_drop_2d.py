#!/usr/bin/env python

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen

class DropSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        drop_map = (self.gx/2-hx)**2 + (self.gy/2-hy)**2 <= 64**2
        sim.rho[:] = 2.0
        sim.phi[:] = 0.02
        sim.rho[drop_map] = 0.02
        sim.phi[drop_map] = 0.2
        
class SCSim(LBBinaryFluidShanChen):
    subdomain = DropSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'G11': -4.8,
            'visc': 1.0 / 6.0,
            'periodic_x': True,
            'periodic_y': True,
            'sc_potential': 'classic',
            'every': 20,
            })


if __name__ == '__main__':
    LBSimulationController(SCSim, LBGeometry2D).run()

#!/usr/bin/python

from sailfish.geo import LBGeometry2D
from sailfish.geo_block import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBSingleFluidShanChen

class DropSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        drop_map = (self.gx /  2 - hx)**2 + (self.gy / 2 - hy)**2 <= 40**2
        sim.rho[:] = 0.2
        sim.rho[drop_map] = 1.8


class SCSim(LBSingleFluidShanChen):
    subdomain = DropSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'G': 5.0,
            'visc': 1.0 / 6.0,
            'periodic_x': True,
            'periodic_y': True,
            'sc_potential': 'classic',
            'every': 20,
            })


if __name__ == '__main__':
    LBSimulationController(SCSim, LBGeometry2D).run()

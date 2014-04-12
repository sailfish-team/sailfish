#!/usr/bin/env python

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen

class DropSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        drop_map = (self.gx/2-hx)**2 + (self.gy/2-hy)**2 <= 64**2
        sim.g0m0[:] = 2.0
        sim.g1m0[:] = 0.02
        sim.g0m0[drop_map] = 0.02
        sim.g1m0[drop_map] = 0.2

class SCSim(LBMultiFluidShanChen):
    subdomain = DropSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 2,
            'lat_nx': 256,
            'lat_ny': 256,
            'G00': -4.8,
            'visc0': 1.0 / 6.0,
            'visc1': 1.0 / 6.0,
            'periodic_x': True,
            'periodic_y': True,
            'sc_potential': 'classic',
            'every': 20,
            })


if __name__ == '__main__':
    LBSimulationController(SCSim, LBGeometry2D).run()

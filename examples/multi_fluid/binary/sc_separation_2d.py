#!/usr/bin/env python

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen

class SeparationDomain(Subdomain2D):
    def initial_conditions(self, sim, hx, hy):
        sim.g0m0[:] = 1.0 + np.random.rand(*sim.g0m0.shape) / 1000.0
        sim.g1m0[:] = 1.0 + np.random.rand(*sim.g1m0.shape) / 1000.0

    def boundary_conditions(self, hx, hy):
        pass

class SeparationSCSim(LBMultiFluidShanChen):
    subdomain = SeparationDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 2,
            'lat_nx': 256,
            'lat_ny': 256,
            'grid': 'D2Q9',
            'G01': 1.2,
            'visc0': 1.0/6.0,
            'visc1': 1.0/6.0,
            'periodic_x': True,
            'periodic_y': True})


if __name__ == '__main__':
    ctrl = LBSimulationController(SeparationSCSim, LBGeometry2D)
    ctrl.run()

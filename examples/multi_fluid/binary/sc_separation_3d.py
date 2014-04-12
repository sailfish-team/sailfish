#!/usr/bin/env python

import numpy as np

from sailfish.geo import LBGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen

class SeparationDomain(Subdomain3D):
    def initial_conditions(self, sim, hx, hy, hz):
        sim.g0m0[:] = 1.0 + np.random.rand(*sim.g0m0.shape) / 1000.0
        sim.g1m0[:] = 1.0 + np.random.rand(*sim.g1m0.shape) / 1000.0

    def boundary_conditions(self, hx, hy, hz):
        pass


class SeparationSCSim(LBMultiFluidShanChen):
    subdomain = SeparationDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nc': 2,
            'lat_nx': 192,
            'lat_ny': 192,
            'lat_nz': 192,
            'grid': 'D3Q19',
            'G01': 1.2,
            'visc0': 1.0/6.0,
            'visc1': 1.0/6.0,
            'periodic_x': True,
            'periodic_y': True,
            'periodic_z': True})


if __name__ == '__main__':
    ctrl = LBSimulationController(SeparationSCSim, LBGeometry3D)
    ctrl.run()

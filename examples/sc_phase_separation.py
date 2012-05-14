#!/usr/bin/python

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBSingleFluidShanChen


class SeparationSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = np.random.rand(*sim.rho.shape) / 100 + 0.693


class SCSim(LBSingleFluidShanChen):
    subdomain = SeparationSubdomain

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
            'every': 20})


if __name__ == '__main__':
    LBSimulationController(SCSim, LBGeometry2D).run()

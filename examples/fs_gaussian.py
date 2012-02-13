#!/usr/bin/python

import numpy as np
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import Subdomain2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFreeSurface, LBForcedSim


class FSSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        sigma = min(self.gy, self.gx) / 12.0
        amp = 0.4

        sim.rho[:] = 1.0 + amp * np.exp(-(np.square(hx - self.gx / 2.0) +
            np.square(hy - self.gy / 2.0)) / sigma**2)


class FSSim(LBFreeSurface, LBForcedSim):
    subdomain = FSSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'every': 10,
            'visc': 0.005})


if __name__ == '__main__':
    LBSimulationController(FSSim, LBGeometry2D).run()

#!/usr/bin/env python
"""A low Reynolds number flow of a drop through a capillary channel."""

import numpy as np

from sailfish import sym
from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_multi import LBMultiFluidShanChen
from sailfish.lb_base import LBForcedSim


class CapillaryDomain(Subdomain2D):
    max_v = 0.005

    def boundary_conditions(self, hx, hy):
        chan_diam = 32 * self.gy / 200.0
        chan_len = 200 * self.gy / 200.0
        rem_y = (self.gy - chan_diam) / 2

        geometry = np.zeros(hx.shape, dtype=np.bool)
        geometry[hy == 0] = True
        geometry[hy == self.gy - 1] = True
        geometry[np.logical_and(
                    hy < rem_y,
                    hy < rem_y - (np.abs((hx - self.gx/2)) - chan_len/2)
                )] = True
        geometry[np.logical_and(
                    (self.gy - hy) < rem_y,
                    (self.gy - hy) < rem_y - (np.abs((hx - self.gx/2)) - chan_len/2)
                )] = True

        self.set_node(geometry, NTFullBBWall)

    def initial_conditions(self, sim, hx, hy):
        drop_diam = 30 * self.gy / 200.0
        sim.g0m0[:] = 1.0
        sim.g1m0[:] = 0.124
        sim.g0m0[(hx - drop_diam * 2) ** 2 + (hy - self.gy / 2.0)**2 < drop_diam**2] = 0.124
        sim.g1m0[(hx - drop_diam * 2) ** 2 + (hy - self.gy / 2.0)**2 < drop_diam**2] = 1.0


class CapillarySCSim(LBMultiFluidShanChen, LBForcedSim):
    subdomain = CapillaryDomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
                'lat_nc': 2,
                'lat_nx': 640,
                'lat_ny': 200,
                'grid': 'D2Q9',
                'G01': 1.6,
                'visc0': 1.0/6.0,
                'visc1': 1.0/6.0,
                'periodic_x': True,
                'periodic_y': False,
            })

    def __init__(self, config):
        super(CapillarySCSim, self).__init__(config)
        f1 = self.subdomain.max_v * (8.0 * config.visc0) / config.lat_ny
        self.add_body_force((f1, 0.0), grid=0)
        self.add_body_force((f1, 0.0), grid=1)


if __name__ == '__main__':
    ctrl = LBSimulationController(CapillarySCSim, LBGeometry2D)
    ctrl.run()

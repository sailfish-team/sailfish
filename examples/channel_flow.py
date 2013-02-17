#!/usr/bin/env python
"""
Flow between two parallel plates driven by a body force.
"""

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim
from sailfish.node_type import NTWallTMS, NTEquilibriumVelocity

# Channel half-height.
H = 120
Re = 180
visc = 0.1
u_tau = Re * visc / H

# 2.3 max stability with bounce back
# 3.0 shocks with ELBM
print 'Delta = %e' % (u_tau / visc)

class ChannelSubdomain(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        wall_map = (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, NTWallTMS)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

class ChannelSim(LBFluidSim, LBForcedSim):
    subdomain = ChannelSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 4 * H,
            'lat_ny': 2 * H,
            'lat_nz': H,
            'grid': 'D3Q15',
            'model': 'elbm',
            'visc': visc,
            'max_iters': 100000,
            'every': 10000,
            'periodic_x': True,
            'periodic_z': True,
            'output': 'channel'})

    def __init__(self, config):
        super(ChannelSim, self).__init__(config)
        self.add_body_force((Re**2 * visc**2 / H**3, 0.0, 0.0))

if __name__ == '__main__':
    ctrl = LBSimulationController(ChannelSim, EqualSubdomainsGeometry3D)
    ctrl.run()

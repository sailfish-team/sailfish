#!/usr/bin/python

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidFreeEnergy
from sailfish.lb_base import LBForcedSim


class MicrochannelDomain(Subdomain2D):
    wall_thickness = 2

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.phi[:] = 1.0
        sim.phi[(hx >= self.gx/3) & (hx < self.gx * 2/3) &
                (hy >= self.config.film_thickness + self.wall_thickness) &
                (hy < self.gy - self.wall_thickness - self.config.film_thickness)] = -1.0

    def boundary_conditions(self, hx, hy):
        wall_map = (hy < self.wall_thickness) | (hy >= self.gy - self.wall_thickness)
        self.set_node(wall_map, NTFullBBWall)


class MicrochannelSim(LBBinaryFluidFreeEnergy, LBForcedSim):
    subdomain = MicrochannelDomain

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--H', type=int, default=51,
                help='channel height')
        group.add_argument('--film_thickness', type=int,
                default=6, help='film thickness in nodes')

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'tau_a': 2.5,
            'tau_b': 0.7,
            'tau_phi': 1.0,
            'kappa': 0.04,
            'A': 0.04,
            'Gamma': 1.0,
            'periodic_x': True})

    @classmethod
    def modify_config(cls, config):
        config.lat_nx = 15 * config.H
        config.lat_ny = config.H + 2 + MicrochannelDomain.wall_thickness

    def __init__(self, config):
        super(MicrochannelSim, self).__init__(config)

        self.add_body_force((6.0e-6, 0.0), grid=0, accel=False)

        # Use the fluid velocity in the relaxation of the order parameter field,
        # and the molecular velocity in the relaxation of the density field.
        self.use_force_for_equilibrium(None, target_grid=0)
        self.use_force_for_equilibrium(0, target_grid=1)

if __name__ == '__main__':
    ctrl = LBSimulationController(MicrochannelSim, EqualSubdomainsGeometry2D)
    ctrl.run()

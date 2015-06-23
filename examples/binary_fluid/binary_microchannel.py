#!/usr/bin/python

from math import sqrt
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
        group.add_argument('--Ca', type=float, default=1.0,
                help='target capillary number')
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
        # Lattice size based on a 15 H x H geometry.
        config.lat_nx = 15 * config.H

        # 2 additional nodes because of NTFullBB walls.
        config.lat_ny = config.H + 2 + MicrochannelDomain.wall_thickness

    def __init__(self, config):
        super(MicrochannelSim, self).__init__(config)

        # Interface tension and width.
        gamma = sqrt(8 * config.kappa * config.A / 9.0)
        xi = 2 * sqrt(2 * config.kappa / config.A)

        visc_liq = 1.0/3.0 * (config.tau_a - 0.5)
        u_bubble = config.Ca * gamma / config.tau_a
        Rey = config.H * u_bubble / visc_liq
        force = u_bubble * 8.0 * visc_liq / config.H**2

        config.logger.info('Ca:{0:.2f}   Re:{1:.2f}  u_bubble:{2:.4e}  force:{3:.4e}'.format(
            config.Ca, Rey, u_bubble, force))

        self.add_body_force((force, 0.0), grid=0)
        self.use_force_for_equilibrium(0, target_grid=1)


if __name__ == '__main__':
    ctrl = LBSimulationController(MicrochannelSim, EqualSubdomainsGeometry2D)
    ctrl.run()

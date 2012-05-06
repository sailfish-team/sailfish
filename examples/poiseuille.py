#!/usr/bin/python -u

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.geo_block import Subdomain2D
from sailfish.node_type import NTFullBBWall, NTHalfBBWall, NTEquilibriumDensity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim


class PoiseuilleSubdomain(Subdomain2D):
    """2D Poiseuille geometry."""

    max_v = 0.02
    wall_bc = NTFullBBWall

    def _set_pressure_bc(self, hx, hy):
        """Adds pressure boundary conditions at the ends of the pipe."""
        pressure_bc = NTEquilibriumDensity
        land = np.logical_and

        if self.config.horizontal:
            pressure = (self.max_v * (8.0 * self.config.visc) /
                (self.channel_width(self.config)**2) * self.gx)

            not_wall = land(hy > 0, hy < self.gy - 1)
            self.set_node(land(not_wall, hx == 0), pressure_bc(1.0 + 3.0 * pressure/2.0))
            self.set_node(land(not_wall, hx == self.gx - 1), pressure_bc(1.0 - 3.0 * pressure/2.0))
        else:
            pressure = (self.max_v * (8.0 * self.config.visc) /
                (self.channel_width(self.config)**2) * self.gy)

            not_wall = land(hx > 0, hx < self.gx - 1)
            self.set_node(land(not_wall, hy == 0), pressure_bc(1.0 + 3 * pressure/2.0))
            self.set_node(land(not_wall, hy == self.gy - 1), pressure_bc(1.0 - 3 * pressure/2.0))

    def boundary_conditions(self, hx, hy):
        if self.config.drive == 'pressure':
            self._set_pressure_bc(hx, hy)

        # Set walls.
        if self.config.horizontal:
            self.set_node(hy == 0, self.wall_bc)
            self.set_node(hy == self.gy - 1, self.wall_bc)
        else:
            self.set_node(hx == 0, self.wall_bc)
            self.set_node(hx == self.gx - 1, self.wall_bc)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0

        if not self.config.stationary:
            return

        if self.config.drive == 'pressure':
            # Start with correct pressure profile.
            pressure = (self.max_v * (8.0 * self.config.visc) /
                    (self.channel_width(self.config)**2))

            if self.config.horizontal:
                sim.rho[:] = 1.0 + 3.0 * pressure * (self.gx/2.0 - hx)
            else:
                sim.rho[:] = 1.0 + 3.0 * pressure * (self.gy/2.0 - hy)
        else:
            # Start with correct velocity profile.
            if self.config.horizontal:
                sim.vx[:] = self.velocity_profile(self.config, hy)
            else:
                sim.vy[:] = self.velocity_profile(self.config, hx)

    @classmethod
    def velocity_profile(cls, config, hi):
        width = cls.channel_width(config)
        h = cls.wall_bc.location

        return (4.0 * cls.max_v / width**2 * (hi + h) * (width - hi - h))

    @classmethod
    def channel_width(cls, config):
        return cls.width(config) - 1 - 2 * cls.wall_bc.location

    @classmethod
    def width(cls, config):
        if config.horizontal:
            return config.lat_ny
        else:
            return config.lat_nx


class PoiseuilleSim(LBFluidSim, LBForcedSim):
    subdomain = PoiseuilleSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 128,
            'lat_ny': 128,
            'visc': 0.1,
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--horizontal', type=bool, default=False,
                help='simulate a horizontal flow (along the X axis)')
        group.add_argument('--stationary', type=bool, default=False,
                help='start with the correct velocity profile in the whole domain')
        group.add_argument('--drive', type=str, default='force',
                choices=['force', 'pressure'])
        group.add_argument('--wall', type=str, choices=['fullbb', 'halfbb'])

    @classmethod
    def modify_config(cls, config):
        if config.drive == 'force':
            config.periodic_x = config.horizontal
            config.periodic_y = not config.horizontal

        if config.wall == 'halfbb':
            cls.subdomain.wall_bc = NTHalfBBWall

    def __init__(self, config):
        super(PoiseuilleSim, self).__init__(config)

        if config.drive == 'force':
            channel_width = self.subdomain.channel_width(config)
            accel = self.subdomain.max_v * (8.0 * config.visc) / channel_width**2
            force_vec = (accel, 0.0) if config.horizontal else (0.0, accel)
            self.add_body_force(force_vec)


if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim, LBGeometry2D).run()

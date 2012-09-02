#!/usr/bin/python -u

import math
import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.node_type import NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim


class PoiseuilleSubdomain(Subdomain3D):
    """3D Poiseuille geometry."""

    max_v = 0.02
    wall_bc = NTFullBBWall

    def boundary_conditions(self, hx, hy, hz):
        radiussq = (self.channel_width(self.config) / 2.0)**2

        if self.config.flow_direction == 'z':
            wall_map = (hx - (self.gx / 2 - 0.5))**2 + (hy - (self.gy / 2 - 0.5))**2 >= radiussq
        elif self.config.flow_direction == 'y':
            wall_map = (hx - (self.gx / 2 - 0.5))**2 + (hz - (self.gz / 2 - 0.5))**2 >= radiussq
        else:
            wall_map = (hy - (self.gy / 2 - 0.5))**2 + (hz - (self.gz / 2 - 0.5))**2 >= radiussq

        self.set_node(wall_map, self.wall_bc)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

        if not self.config.stationary:
            return

        if self.config.drive == 'pressure':
            # Start with correct pressure profile.
            pressure = (self.max_v * (16.0 * self.config.visc) /
                    (self.channel_width(self.config)**2))

            if self.config.flow_direction == 'x':
                sim.rho[:] = 1.0 + 3.0 * pressure * (self.gx / 2.0 - hx)
            elif self.config.flow_direction == 'y':
                sim.rho[:] = 1.0 + 3.0 * pressure * (self.gy / 2.0 - hy)
            else:
                sim.rho[:] = 1.0 + 3.0 * pressure * (self.gz / 2.0 - hz)
        else:
            # Start with correct velocity profile.
            h = -0.5
            radius = self.get_chan_width() / 2.0

            if self.config.flow_direction == 'z':
                rc = np.sqrt((hx - self.gx / 2.0 - h)**2 + (hy - self.gy / 2.0 - h)**2)
                self.sim.vz[rc <= radius] = self._velocity_profile(rc[rc <= radius])
            elif self.config.flow_direction == 'y':
                rc = np.sqrt((hx - self.gx / 2.0 - h)**2 + (hz - self.gz / 2.0 - h)**2)
                self.sim.vy[rc <= radius] = self._velocity_profile(rc[rc <= radius])
            else:
                rc = np.sqrt((hz - self.gz / 2.0 - h)**2 + (hy - self.gy / 2.0 - h)**2)
                self.sim.vx[rc <= radius] = self._velocity_profile(rc[rc <= radius])

    # Schematic drawing of the simulated system with both on-grid and mid-grid
    # bondary conditions.
    #
    # Columns:
    # 1st: linear distance from one of the pipe walls
    # 2nd: radial distance from the axis of the pipe
    # 3rd: node index
    #
    # width: 6
    #
    # Midgrid BC:
    # chan_width: 4
    #
    # wwww -0.5  2.5  0     -
    # -     0    2.0        |-
    # fff   0.5  1.5  1     |----
    # -     1    1.0        |-----
    # fff   1.5  0.5  2     |------
    # -     2    0.0        |------*
    # fff   2.5  0.5  3     |------
    # -     3    1.0        |-----
    # fff   3.5  1.5  4     |----
    # -     4    2.0        |-
    # wwww  4.5  2.5  5     -
    #
    # On-grid BC:
    # chan_width: 5
    #
    # wwww  0.0  2.5  0     |-
    # -     0.5  2.0        |---
    # fff   1.0  1.5  1     |-----
    # -     1.5  1.0        |------
    # fff   2.0  0.5  2     |-------
    # -     2.5  0.0        |-------*
    # fff   3.0  0.5  3     |-------
    # -     3.5  1.0        |------
    # fff   4.0  1.5  4     |-----
    # -     4.5  2.0        |---
    # wwww  5.0  2.5  5     |-

    # TODO(mjanusz): Verify the correctness of this.
    def _velocity_profile(self, r):
        width = self.channel_width(self.config)
        return self.max_v / (width / 2.0)**2 * ((width / 2.0)**2 - r**2)

    @classmethod
    def channel_width(cls, config):
        return cls.width(config) - 1 - 2 * cls.wall_bc.location

    @classmethod
    def width(cls, config):
        if config.flow_direction == 'x':
            return min(config.lat_ny, config.lat_nz)
        elif config.flow_direction == 'y':
            return min(config.lat_nx, config.lat_nz)
        else:
            return min(config.lat_nx, config.lat_ny)


class PoiseuilleSim(LBFluidSim, LBForcedSim):
    subdomain = PoiseuilleSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 64,
            'lat_ny': 64,
            'lat_nz': 64,
            'visc': 0.1,
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)

        group.add_argument('--flow_direction', type=str, default='x',
                choices=['x', 'y', 'z'],
                help='direction along which the fluid is to flow')
        group.add_argument('--stationary', type=bool, default=False,
                help='start with the correct velocity profile in the whole domain')
        group.add_argument('--drive', type=str, default='force',
                choices=['force', 'pressure'])


    @classmethod
    def modify_config(cls, config):
        if config.drive == 'force':
            config.periodic_x = config.flow_direction == 'x'
            config.periodic_y = config.flow_direction == 'y'
            config.periodic_z = config.flow_direction == 'z'


    def __init__(self, config):
        super(PoiseuilleSim, self).__init__(config)

        if config.drive == 'force':
            channel_width = self.subdomain.channel_width(config)
            accel = self.subdomain.max_v * (16.0 * config.visc) / channel_width**2
            if config.flow_direction == 'x':
                force_vec = (accel, 0.0, 0.0)
            elif config.flow_direction == 'y':
                force_vec = (0.0, accel, 0.0)
            else:
                force_vec = (0.0, 0.0, accel)
            self.add_body_force(force_vec)


if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim, EqualSubdomainsGeometry3D).run()

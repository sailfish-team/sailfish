#!/usr/bin/python -u

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall, NTHalfBBWall, NTEquilibriumDensity, DynamicValue
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim
from sailfish.sym import S
from sympy import sin

import poiseuille


class PulsatileSubdomain(poiseuille.PoiseuilleSubdomain):
    max_v = 0.02
    wall_bc = NTFullBBWall

    def _set_pressure_bc(self, hx, hy):
        """Adds pressure boundary conditions at the ends of the pipe."""
        pressure_bc = NTEquilibriumDensity
        land = np.logical_and

        if self.config.horizontal:
            pressure = (self.max_v * sin(S.time) * (8.0 * self.config.visc) /
                (self.channel_width(self.config)**2) * S.gx)

            not_wall = land(hy > 0, hy < self.gy - 1)
            self.set_node(land(not_wall, hx == 0),
                    pressure_bc(DynamicValue(1.0 + 3.0 * pressure/2.0)))
            self.set_node(land(not_wall, hx == self.gx - 1),
                    pressure_bc(DynamicValue(1.0 - 3.0 * pressure/2.0)))
        else:
            pressure = (self.max_v * sin(S.time) * (8.0 * self.config.visc) /
                (self.channel_width(self.config)**2) * S.gy)

            not_wall = land(hx > 0, hx < self.gx - 1)
            self.set_node(land(not_wall, hy == 0),
                    pressure_bc(DynamicValue(1.0 + 3 * pressure/2.0)))
            self.set_node(land(not_wall, hy == self.gy - 1),
                    pressure_bc(DynamicValue(1.0 - 3 * pressure/2.0)))


class PulsatileSim(poiseuille.PoiseuilleSim):
    subdomain = PulsatileSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        poiseuille.PoiseuilleSim.update_defaults(defaults)
        defaults.update({
            'drive': 'pressure',
            'dt_per_lattice_time_unit': 0.001
            })


if __name__ == '__main__':
    LBSimulationController(PulsatileSim, LBGeometry2D).run()

#!/usr/bin/env python -u

import numpy as np

from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall, NTHalfBBWall, NTEquilibriumDensity, NTEquilibriumVelocity, SpatialArray, DynamicValue
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim
from sailfish.sym import S, D3Q19

from sympy import sin, Piecewise

class PoiseuilleSubdomain(Subdomain2D):
    """2D Poiseuille geometry."""

    max_v = 0.02
    wall_bc = NTFullBBWall
    velocity_bc = NTEquilibriumVelocity


    def boundary_conditions(self, hx, hy):
        land=np.logical_and
        
        # Set walls.
        self.set_node(hy == 0, self.wall_bc)
        self.set_node(hy == self.gy - 1, self.wall_bc)
        
        not_wall = land(hy > 0, hy < self.gy - 1)
        width = self.channel_width(self.config)
        radius = width / 2.0
        radius_sq = radius**2
        
        # Add 0.5 to the grid symbols to indicate that the node is located in the
        # middle of the grid cell.
        # The velocity vector direction matches the flow orientation vector.
        if self.config.velocity =="equation":
            vv =  self.max_v * (1.0 -  (S.gy + 0.5 - radius)**2 / radius_sq)* \
                        Piecewise((S.time / 5000, S.time < 5000),(1.0, True))
            self.set_node(land(not_wall, hx == 0), self.velocity_bc(DynamicValue(vv,0.0)))
        elif self.config.velocity =="spatial_array":
            vx = self.max_v * (1 - (hy + 0.5 - radius)**2/radius_sq)
            where = (hx==0) & not_wall
            self.set_node(where, self.velocity_bc(DynamicValue( \
                                SpatialArray(vx, index="x", where=where)* \
                                    Piecewise((S.time / 5000, S.time < 5000),(1.0, True)), 0.0))) 
        

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0


    @classmethod
    def velocity_profile(cls, config, hi):
        width = cls.channel_width(config)
        hx = hi - cls.wall_bc.location
        a = width / 2.0
        rx = np.abs(a - hx)
        return 4.0 * cls.max_v / width**2 * (a**2 - rx**2)

    @classmethod
    def channel_width(cls, config):
        return cls.width(config) - 1 - 2 * cls.wall_bc.location

    @classmethod
    def width(cls, config):
        return config.lat_ny
        


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
        group.add_argument('--stationary', action='store_true', default=False,
                help='start with the correct velocity profile in the whole domain')
        group.add_argument('--wall', type=str, choices=['fullbb', 'halfbb'])
        group.add_argument('--velocity', type=str, choices=['equation', 'spatial_array'],
                default='equation')

    @classmethod
    def modify_config(cls, config):
        if config.wall == 'halfbb':
            cls.subdomain.wall_bc = NTHalfBBWall

    def __init__(self, config):
        super(PoiseuilleSim, self).__init__(config)

        

    _ref = None
    _prev_l2 = 0
    def after_step(self, runner):
        every = 1000
        mod = self.iteration % every

        if mod == every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            # Cache the reference solution.
            if self._ref is None:
                hx, hy = runner._subdomain._get_mgrid()
                self._ref = runner._subdomain.velocity_profile(self.config, hx)

            # Output data useful for monitoring the state of the simulation.
            m = runner._output._fluid_map
            l2 = (np.linalg.norm(self._ref[m] - runner._sim.vy[m]) /
                  np.linalg.norm(self._ref[m]))

            self.config.logger.info('%d %e %e' % (self.iteration, l2, np.nanmax(runner._sim.vy)))

            if (np.abs(self._prev_l2 - l2) / l2 < 1e-6):
                runner._quit_event.set()

            self._prev_l2 = l2


if __name__ == '__main__':
    LBSimulationController(PoiseuilleSim, LBGeometry2D).run()

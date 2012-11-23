#!/usr/bin/env python
"""
Four rolls mill flow.

This flow is closely related to the 2D Taylor-Green vortex flow. The initial
conditions for both cases are the same, but the current simulation introduces
an additional body force which prevents the vortex from decaying. The solution
does not depend on time and the inital state is a steady state.
"""

from math import sqrt
from sailfish.controller import LBSimulationController
from sailfish.lb_base import LBForcedSim
from sailfish.node_type import DynamicValue
from sailfish.sym import S
from taylor_green_2d import TaylorGreenSubdomain, TaylorGreenSim
from sympy import sin, cos


class FourRollsMill(TaylorGreenSim, LBForcedSim):
    # The reference solution of the four rolls mill case is the same as
    # that of the Taylor Green case at t = 0.
    def reference_solution(self, hx, hy, nx, ny, iteration, Ma):
        return self.subdomain.solution(self.config, hx, hy, nx, ny,
                                       0, Ma)

    def __init__(self, config):
        super(FourRollsMill, self).__init__(config)

        ny, nx = self.config.lat_ny, self.config.lat_nx
        kx, ky, ksq, k = TaylorGreenSubdomain.get_k(config, nx, ny)
        f = ksq * config.visc * config.max_v

        accel_vec = (-f * ky / k * sin(ky * S.gy) * cos(kx * S.gx),
                     +f * kx / k * sin(kx * S.gx) * cos(ky * S.gy))

        self.add_body_force(DynamicValue(*accel_vec))


if __name__ == '__main__':
    ctrl = LBSimulationController(FourRollsMill)
    ctrl.run()

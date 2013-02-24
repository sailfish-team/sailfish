#!/usr/bin/env python
"""Verifies the node scratch space functionality."""

import unittest
import numpy as np

from sailfish.subdomain import Subdomain2D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.node_type import NTFullBBWall, NTDoNothing, NTRegularizedVelocity
from sailfish.lb_single import LBFluidSim
from sailfish.controller import LBSimulationController
from sailfish.sym import D2Q9


class TestSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        wall_map = (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, NTFullBBWall)
        self.set_node(np.logical_not(wall_map) & (hx == 0),
                      NTRegularizedVelocity((0.05, 0.0)))
        self.set_node(np.logical_not(wall_map) & (hx == self.gx - 1),
                      NTDoNothing)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[:] = 0.05


class TestSim(LBFluidSim):
    subdomain = TestSubdomain


class Test2DDoNothing(unittest.TestCase):
    nx = 64
    ny = 64

    def test_do_nothing(self):
        settings = {
            'debug_single_process': True,
            'quiet': True,
            'access_pattern': 'AB',
            'lat_nx': self.nx,
            'lat_ny': self.ny,
            'max_iters': 1000
        }

        ctrl_ab = LBSimulationController(TestSim, default_config=settings)
        ctrl_ab.run(ignore_cmdline=True)
        sim_ab = ctrl_ab.master.sim

        settings['access_pattern'] = 'AA'

        ctrl_aa = LBSimulationController(TestSim, default_config=settings)
        ctrl_aa.run(ignore_cmdline=True)
        sim_aa = ctrl_aa.master.sim

        np.testing.assert_allclose(sim_ab.vx, sim_aa.vx)
        np.testing.assert_allclose(sim_ab.vy, sim_aa.vy)
        np.testing.assert_allclose(sim_ab.rho, sim_aa.rho)


if __name__ == '__main__':
    unittest.main()

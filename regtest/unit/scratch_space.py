#!/usr/bin/env python

import unittest

from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTGrad
from sailfish.lb_single import LBFluidSim
from sailfish.controller import LBSimulationController


class TestSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        self.set_node(hx == 0, NTGrad)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0

class TestSim(LBFluidSim):
    subdomain = TestSubdomain


class Test2DScratchSpace(unittest.TestCase):

    def test_scratch_space(self):
        ctrl = LBSimulationController(TestSim, default_config={
            'lat_nx': 64,
            'lat_ny': 64,
            'max_iters': 10})
        ctrl.run(ignore_cmdline=True)


if __name__ == '__main__':
    unittest.main()

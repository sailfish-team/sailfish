#!/usr/bin/env python
"""Verifies the node scratch space functionality."""

import unittest
import numpy as np

from sailfish.subdomain import Subdomain2D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.node_type import NTGradFreeflow
from sailfish.lb_base import LBSim, ScalarField
from sailfish.controller import LBSimulationController
from sailfish.sym import D2Q9


class TestSubdomainRunner(SubdomainRunner):
    def step(self, output_req):
        geo_map = self.gpu_geo_map()

        self.exec_kernel('TestNodeScratchSpaceWrite', [geo_map,
            self.gpu_scratch_space], 'PP')
        self.exec_kernel('TestNodeScratchSpaceRead', [geo_map,
            self.gpu_field(self._sim.output_x),
            self.gpu_field(self._sim.output_y),
            self.gpu_field(self._sim.output_xy),
            self.gpu_scratch_space], 'PPPPP')

        self._fields_to_host()
        self._sim.iteration += 1


class TestSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        self.set_node(hx == 0, NTGradFreeflow)

    def initial_conditions(self, sim, hx, hy):
        pass


class TestSim(LBSim):
    subdomain = TestSubdomain
    subdomain_runner = TestSubdomainRunner
    kernel_file = "scratch_space.mako"

    @classmethod
    def modify_config(cls, config):
        config.unit_test = True

    @property
    def grid(self):
        return D2Q9

    @property
    def grids(self):
        return [D2Q9]

    @classmethod
    def fields(cls):
        return [ScalarField('output_x'), ScalarField('output_y'),
                ScalarField('output_xy')]


class Test2DScratchSpace(unittest.TestCase):
    nx = 64
    ny = 64

    def test_scratch_space(self):
        ctrl = LBSimulationController(TestSim, default_config={
            'debug_single_process': True,
            'quiet': True,
            'lat_nx': self.nx,
            'lat_ny': self.ny,
            'max_iters': 1})
        ctrl.run(ignore_cmdline=True)
        sim = ctrl.master.sim

        hy, hx = np.mgrid[0:self.ny,0:self.nx]

        np.testing.assert_allclose(sim.output_x[hx == 0], (hx[hx == 0] + 1) / 10.0)
        np.testing.assert_allclose(sim.output_y[hx == 0], (hy[hx == 0] + 1) / 10.0)
        np.testing.assert_allclose(sim.output_xy[hx == 0],
                ((hx[hx == 0] + 1) * (hy[hx == 0] + 1) / 10.0))


if __name__ == '__main__':
    unittest.main()

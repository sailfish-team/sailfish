#!/usr/bin/env python

import unittest
import numpy as np


from sailfish.lb_base import LBSim, ScalarField
from sailfish.subdomain import Subdomain2D, Subdomain3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.controller import LBSimulationController

NX = 128
NY = 80
NZ = 40

class TestSubdomain2D(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        pass

    def initial_conditions(self, sim, hx, hy):
        sim.data[:] = np.random.random(sim.data.shape)

class TestSubdomainRunner(SubdomainRunner):
    def step(self, output_req):
        gpu_data = self.gpu_field(self._sim.data)
        self.ret_x = np.zeros(NX, dtype=np.float32)
        gpu_ret_x = self.backend.alloc_buf(like=self.ret_x)

        self.ret_y = np.zeros(NY, dtype=np.float32)
        gpu_ret_y = self.backend.alloc_buf(like=self.ret_y)

        # Reduce over Y.
        k = self.get_kernel('Reducetestx2d', [gpu_data, gpu_ret_x],
                            'PP', block_size=1024)
        self.backend.run_kernel(k, [1, 1])
        # Reduce over X.
        k = self.get_kernel('Reducetesty2d', [gpu_data, gpu_ret_y],
                            'PP', shared=NX * 4, block_size=1024)
        self.backend.run_kernel(k, [1, NY])

        self.backend.from_buf(gpu_ret_x)
        self.backend.from_buf(gpu_ret_y)
        self.backend.sync()

class TestSim2D(LBSim):
    subdomain = TestSubdomain2D
    subdomain_runner = TestSubdomainRunner
    kernel_file = 'reduce.mako'

    @classmethod
    def modify_config(cls, config):
        config.unit_test = True

    @classmethod
    def fields(cls):
        return [ScalarField('data')]

settings = {
    'debug_single_process': True,
    'quiet': True,
    'check_invalid_results_gpu': False,
    'check_invalid_results_host': False,
    'max_iters': 1}

class TestXReduction(unittest.TestCase):
    def test_2d(self):
        s = settings
        s.update({
            'lat_nx': NX,
            'lat_ny': NY})
        ctrl = LBSimulationController(TestSim2D, default_config=s)
        ctrl.run(ignore_cmdline=True)

        sim = ctrl.master.sim
        exp = np.sum(sim.rho, axis=1)

        print sim._runner

    def test_3d(self):
        pass

if __name__ == '__main__':
    unittest.main()

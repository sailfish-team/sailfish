#!/usr/bin/env python

import unittest
import numpy as np
import math

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
        sim.data[:] = hx * 3 + hy * 7

class TestSubdomain3D(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        pass

    def initial_conditions(self, sim, hx, hy, hz):
        sim.data[:] = hx * 3 + hy * 7 + hz * 11

class TestSubdomainRunner2D(SubdomainRunner):
    def step(self, output_req):
        gpu_data = self.gpu_field(self._sim.data)
        self.ret_x = np.zeros(NX, dtype=np.float32)
        gpu_ret_x = self.backend.alloc_buf(like=self.ret_x)

        self.ret_y = np.zeros(NY, dtype=np.float32)
        gpu_ret_y = self.backend.alloc_buf(like=self.ret_y)

        # Reduce over Y.
        k = self.get_kernel('ReduceTestX', [gpu_data, gpu_ret_x],
                            'PP', block_size=1024)
        self.backend.run_kernel(k, [1, 1])
        # Reduce over X.
        k = self.get_kernel('ReduceTestY', [gpu_data, gpu_ret_y],
                            'PP', block_size=1024)
        self.backend.run_kernel(k, [1, NY])

        h = (NX + 2 + 31) / 32
        self.ret_y2 = np.zeros(NY, dtype=np.float32)
        gpu_ret_y2 = self.backend.alloc_buf(like=self.ret_y2)
        temp_y2 = np.zeros((NY, h), dtype=np.float32)
        gpu_temp_y2 = self.backend.alloc_buf(like=temp_y2)

        # Reduce over X.
        k = self.get_kernel('ReduceTestY2', [gpu_data, gpu_temp_y2],
                            'PP', block_size=32)
        self.backend.run_kernel(k, [h, NY])
        k = self.get_kernel('FinalizeReduceTestY2', [gpu_temp_y2, gpu_ret_y2],
                            'PP', block_size=int(pow(2, math.ceil(math.log(h, 2)))))
        self.backend.run_kernel(k, [1, NY])

        self.backend.from_buf(gpu_ret_x)
        self.backend.from_buf(gpu_ret_y)
        self.backend.from_buf(gpu_ret_y2)
        self.backend.from_buf(gpu_temp_y2)

        self.backend.sync()
        self._sim.iteration += 1

class TestSubdomainRunner2DLong(SubdomainRunner):
    def step(self, output_req):
        gpu_data = self.gpu_field(self._sim.data)

class TestSubdomainRunner3D(SubdomainRunner):
    def step(self, output_req):
        gpu_data = self.gpu_field(self._sim.data)
        self.ret_x = np.zeros(NX, dtype=np.float32)
        gpu_ret_x = self.backend.alloc_buf(like=self.ret_x)

        self.ret_y = np.zeros(NY, dtype=np.float32)
        gpu_ret_y = self.backend.alloc_buf(like=self.ret_y)

        self.ret_z = np.zeros(NZ, dtype=np.float32)
        gpu_ret_z = self.backend.alloc_buf(like=self.ret_z)

        k = self.get_kernel('ReduceTestX', [gpu_data, gpu_ret_x],
                            'PP', block_size=1024)
        self.backend.run_kernel(k, [1, 1])

        k = self.get_kernel('ReduceTestY', [gpu_data, gpu_ret_y],
                            'PP', block_size=1024)
        self.backend.run_kernel(k, [1, NY])

        k = self.get_kernel('ReduceTestZ', [gpu_data, gpu_ret_z],
                            'PP', block_size=1024)
        self.backend.run_kernel(k, [1, NZ])

        self.backend.from_buf(gpu_ret_x)
        self.backend.from_buf(gpu_ret_y)
        self.backend.from_buf(gpu_ret_z)
        self.backend.sync()
        self._sim.iteration += 1

class TestSim(LBSim):
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
    'save_src': '/tmp/foo.cu',
    'check_invalid_results_gpu': False,
    'check_invalid_results_host': False,
    'max_iters': 1}

class TestReduction(unittest.TestCase):
    def test_2d(self):
        s = settings
        s.update({
            'lat_nx': NX,
            'lat_ny': NY})

        TestSim.subdomain = TestSubdomain2D
        TestSim.subdomain_runner = TestSubdomainRunner2D

        ctrl = LBSimulationController(TestSim, default_config=s)
        ctrl.run(ignore_cmdline=True)

        sim = ctrl.master.sim
        # reduction over Y
        np.testing.assert_array_almost_equal(ctrl.master.runner.ret_x,
                                             np.sum(sim.data, axis=0))
        # reduction over X
        np.testing.assert_array_almost_equal(ctrl.master.runner.ret_y,
                                             np.sum(sim.data, axis=1))
        # reduction over X with a finalization step
        np.testing.assert_array_almost_equal(ctrl.master.runner.ret_y2,
                                             np.sum(sim.data, axis=1))
    def test_3d(self):
        s = settings
        s.update({
            'lat_nx': NX,
            'lat_ny': NY,
            'lat_nz': NZ})

        TestSim.subdomain = TestSubdomain3D
        TestSim.subdomain_runner = TestSubdomainRunner3D

        ctrl = LBSimulationController(TestSim, default_config=s)
        ctrl.run(ignore_cmdline=True)

        sim = ctrl.master.sim
        # reduction over Z, Y
        np.testing.assert_array_almost_equal(
            ctrl.master.runner.ret_x,
            np.sum(np.sum(sim.data, axis=0), axis=0))

        # reduction over Z, X
        np.testing.assert_array_almost_equal(
            ctrl.master.runner.ret_y,
            np.sum(np.sum(sim.data, axis=0), axis=1))

        # reduction over Y, X
        np.testing.assert_array_almost_equal(
            ctrl.master.runner.ret_z,
            np.sum(np.sum(sim.data, axis=1), axis=1))


if __name__ == '__main__':
    unittest.main()

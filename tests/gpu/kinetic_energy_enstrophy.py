#!/usr/bin/env python
"""Computes kinetic energy and enstrophy on the GPU and compares them with
values computed on the host using operations on numpy arrays."""

import unittest
import numpy as np
import math

from sailfish.lb_base import ScalarField
from sailfish.lb_single import LBFluidSim
from sailfish.subdomain import Subdomain3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.controller import LBSimulationController
from sailfish import util

NX = 50
NY = 30
NZ = 17

class TestSubdomain3D(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        pass

    def initial_conditions(self, sim, hx, hy, hz):
        sim.vx[:] = hx * 3 + hy * 7 + hz * 11;
        sim.vy[:] = hx * 2 + hy * 7 + hz * 3;
        sim.vz[:] = hx * 5 + hy * 7 + hz * 17;

class TestSubdomainRunner3D(SubdomainRunner):
    def step(self, output_req):
        gpu_map = self.gpu_geo_map()
        gpu_v = self.gpu_field(self._sim.v)
        gpu_usq = self.gpu_field(self._sim.usq)
        gpu_vortsq = self.gpu_field(self._sim.vortsq)

        self.exec_kernel('ComputeSquareVelocityAndVorticity',
                         [gpu_map] + gpu_v + [gpu_usq, gpu_vortsq],
                         'PPPPPP')

        b = self.backend
        b.from_buf(gpu_usq)
        b.from_buf(gpu_vortsq)
        b.sync()
        self._sim.iteration += 1

        arr_usq = b.get_array(gpu_usq)
        arr_vortsq = b.get_array(gpu_vortsq)

        div = 2.0 * self._spec.num_nodes

        # Compute the sum in double precision.
        self._sim.kinetic_energy = b.array.sum(arr_usq, dtype=np.float64).get() / div
        self._sim.enstrophy = b.array.sum(arr_vortsq, dtype=np.float64).get() / div

class TestSim(LBFluidSim):
    aux_code = ['data_processing.mako']
    subdomain = TestSubdomain3D
    subdomain_runner = TestSubdomainRunner3D

    @classmethod
    def fields(cls):
        return LBFluidSim.fields() + [
            ScalarField('usq', gpu_array=True, init=0.0),
            ScalarField('vortsq', gpu_array=True, init=0.0)]

class TestKineticEnergyEnstrophy(unittest.TestCase):
    def test_3d(self):
        settings = {
            'debug_single_process': True,
            'quiet': True,
            'check_invalid_results_gpu': False,
            'check_invalid_results_host': False,
            'max_iters': 1,
            'lat_nx': NX,
            'lat_ny': NY,
            'lat_nz': NZ}

        ctrl = LBSimulationController(TestSim, default_config=settings)
        ctrl.run(ignore_cmdline=True)
        sim = ctrl.master.sim

        vort = util.vorticity(sim.v)
        vort_sq = vort[0]**2 + vort[1]**2 + vort[2]**2

        # Convert the velocity field to double precision.
        dbl_v = [x.astype(np.float64) for x in sim.v]

        self.assertAlmostEqual(util.kinetic_energy(dbl_v), sim.kinetic_energy)
        self.assertAlmostEqual(util.enstrophy(dbl_v, dx=1.0), sim.enstrophy)

        np.testing.assert_array_almost_equal(sim.usq, sim.vx**2 + sim.vy**2 + sim.vz**2)
        np.testing.assert_array_almost_equal(sim.vortsq, vort_sq)

if __name__ == '__main__':
    unittest.main()

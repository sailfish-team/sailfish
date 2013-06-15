#!/usr/bin/env python
"""Computes kinetic energy and enstrophy on the GPU and compares them with
values computed on the host using operations on numpy arrays."""

import unittest
import numpy as np
import math

from sailfish.lb_single import LBFluidSim
from sailfish.subdomain import Subdomain3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.controller import LBSimulationController
from sailfish.stats import KineticEnergyEnstrophyMixIn
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
        ke, ens = self._sim.compute_ke_enstropy(self)
        self._sim.iteration += 1
        self._sim.kinetic_energy = ke
        self._sim.enstrophy = ens
        self._fields_to_host()


class TestSim(LBFluidSim, KineticEnergyEnstrophyMixIn):
    subdomain = TestSubdomain3D
    subdomain_runner = TestSubdomainRunner3D


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

        np.testing.assert_array_almost_equal(sim.v_sq, sim.vx**2 + sim.vy**2 + sim.vz**2)
        np.testing.assert_array_almost_equal(sim.vort_sq, vort_sq)

if __name__ == '__main__':
    unittest.main()

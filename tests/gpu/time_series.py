#!/usr/bin/env python

import unittest
import numpy as np
from scipy.interpolate import interp1d

from sailfish.subdomain import Subdomain2D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.node_type import NTEquilibriumDensity, DynamicValue, LinearlyInterpolatedTimeSeries
from sailfish.lb_single import LBFluidSim
from sailfish.controller import LBSimulationController
from sailfish.sym import D2Q9

sin_timeseries = np.sin(2.0 * np.pi * np.linspace(0.0, 1.0, 20))
cos_timeseries = np.cos(2.0 * np.pi * np.linspace(0.0, 1.0, 20))

class TestSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        self.set_node((hx == 5) & (hy == 0),
                      NTEquilibriumDensity(DynamicValue(LinearlyInterpolatedTimeSeries(
                          sin_timeseries, 8))))
        self.set_node((hx == 6) & (hy == 0),
                      NTEquilibriumDensity(DynamicValue(LinearlyInterpolatedTimeSeries(
                          cos_timeseries, 1.61))))
        self.set_node((hx == 7) & (hy == 0),
                      NTEquilibriumDensity(DynamicValue(2.0 * LinearlyInterpolatedTimeSeries(
                          sin_timeseries, 4))))

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[:] = 0.0
        sim.vy[:] = 0.0

class TestSim(LBFluidSim):
    subdomain = TestSubdomain

class TestTimeseries(unittest.TestCase):
    def test_timeseries(self):
        settings = {
            'debug_single_process': True,
            'quiet': True,
            'lat_nx': 64,
            'lat_ny': 32,
            'max_iters': 1,
            'check_invalid_results_host': False,
            'check_invalid_results_gpu': False,
            'every': 1,
        }

        x1 = np.arange(0, 20 * 8, 8)
        x2 = np.arange(0, 60 * 1.61, 1.61)
        x3 = np.arange(0, 20 * 4, 4)

        # This one test time-wrapping and a non-integer step size.
        cc = np.hstack([cos_timeseries, cos_timeseries, cos_timeseries])

        f1 = interp1d(x1, sin_timeseries, kind='linear')
        f2 = interp1d(x2, cc, kind='linear')
        f3 = interp1d(x3, 2.0 * sin_timeseries, kind='linear')

        ctrl = LBSimulationController(TestSim, default_config=settings)
        ctrl.run(ignore_cmdline=True)
        runner = ctrl.master.runner
        sim = ctrl.master.sim

        while sim.iteration < 60:
            runner.step(True)
            runner._fields_to_host(True)
            self.assertAlmostEqual(sim.rho[0,5], f1(sim.iteration - 1), places=6)
            self.assertAlmostEqual(sim.rho[0,6], f2(sim.iteration - 1), places=6)
            self.assertAlmostEqual(sim.rho[0,7], f3(sim.iteration - 1), places=6)

if __name__ == '__main__':
    unittest.main()

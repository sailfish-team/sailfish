#!/usr/bin/env python
"""Verifies the "do nothing" node type."""

import operator
import unittest
import numpy as np

from sailfish.subdomain import Subdomain2D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.node_type import NTWallTMS
from sailfish.lb_single import LBFluidSim
from sailfish.controller import LBSimulationController
from sailfish.sym import D2Q9, relaxation_time
from sailfish.sym_equilibrium import bgk_equilibrium
from sailfish.config import LBConfig


class TestSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        wall_map = (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, NTWallTMS)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[:] = 0.05


vi = lambda x, y: D2Q9.vec_idx([x, y])
fi_start = {
    vi(0, 0): 0.4745,
    vi(1, 0): 0.1179,
    vi(-1, 0): 0.1045,

    vi(0, -1): 0.1809,
    vi(-1, -1): 0.03613,
    vi(1, -1): 0.00946,

    # These distributions should be overriden by the BB rule.
    vi(1, 1): 0.02946,
    vi(0, 1): 0.1110,
    vi(-1, 1): 0.02613,
}

class TestSim(LBFluidSim):
    subdomain = TestSubdomain

    @classmethod
    def modify_config(cls, config):
        config.propagation_enabled = False

    def initial_conditions(self, runner):
        dist = runner._debug_get_dist()
        dist[:] = 0.0

        for k, v in fi_start.iteritems():
            dist[k, 1, 16] = v

        runner._debug_set_dist(dist)
        runner._debug_set_dist(dist, False)


class Test2DTMS(unittest.TestCase):
    nx = 64
    ny = 16

    def test_TMS(self):
        settings = {
            'debug_single_process': True,
            'quiet': True,
            'precision': 'double',
            'access_pattern': 'AB',
            'check_invalid_results_gpu': False,
            'check_invalid_results_host': False,
            'lat_nx': self.nx,
            'lat_ny': self.ny,
            'save_src': '/tmp/test.cu',
            'max_iters': 1,
            'visc': 1.0/12.0,
        }

        ctrl = LBSimulationController(TestSim, default_config=settings)
        ctrl.run(ignore_cmdline=True)
        runner = ctrl.master.runner
        dist = runner._debug_get_dist()

        rho_bb = 0.0
        ux_bb = 0.0
        uy_bb = 0.0

        for k, v in fi_start.iteritems():
            if D2Q9.basis[k][1] == 1:
                v = fi_start[D2Q9.idx_opposite[k]]

            rho_bb += v
            ux_bb += D2Q9.basis[k][0] * v
            uy_bb += D2Q9.basis[k][1] * v

        ux_bb /= rho_bb
        uy_bb /= rho_bb

        rho = 0.0
        ux = 0.0
        uy = 0.0

        cfg = LBConfig()
        cfg.incompressible = False
        cfg.minimize_roundoff = False
        eq = bgk_equilibrium(D2Q9, cfg).expression

        for k, v in fi_start.iteritems():
            if D2Q9.basis[k][1] == 1:
                v = eq[k].evalf(subs={'g0m0': rho_bb, 'g0m1x': ux_bb,
                                      'g0m1y': uy_bb})

            rho += v
            ux += D2Q9.basis[k][0] * v
            uy += D2Q9.basis[k][1] * v

        ux /= rho
        uy /= rho

        print 'Target values are rho=%e, ux=%e, uy=%e' % (rho_bb, ux_bb, uy_bb)
        print 'Instantaneous values are rho=%e, ux=%e, uy=%e' % (rho, ux, uy)

        fneq = {}

        for k, v in fi_start.iteritems():
            if D2Q9.basis[k][1] == 1:
                fneq[k] = (eq[k].evalf(subs={'g0m0': rho_bb, 'g0m1x': ux_bb, 'g0m1y': uy_bb}) -
                           eq[k].evalf(subs={'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy}))
            else:
                fneq[k] = v - eq[k].evalf(subs={'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy})

        tau = relaxation_time(1.0/12.0)
        omega = 1.0 / tau

        res = {}
        for k, v in fi_start.iteritems():
            if D2Q9.basis[k][1] == 1:
                res[k] = ((1.0 - (omega - 1)) *
                          eq[k].evalf(subs={'g0m0': rho_bb, 'g0m1x': ux_bb, 'g0m1y': uy_bb}) +
                          (omega - 1) * eq[k].evalf(subs={'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy}))
            else:
                res[k] = (v + omega * (eq[k].evalf(subs={'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy}) - v) +
                          eq[k].evalf(subs={'g0m0': rho_bb, 'g0m1x': ux_bb, 'g0m1y': uy_bb}) -
                          eq[k].evalf(subs={'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy}))

        for k, v in res.iteritems():
            np.testing.assert_allclose(dist[k,1,16], np.float64(v))


if __name__ == '__main__':
    unittest.main()

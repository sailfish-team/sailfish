#!/usr/bin/env python
"""Verifies the "do nothing" node type."""

import operator
import unittest
import numpy as np

from sailfish.subdomain import Subdomain2D, Subdomain3D
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.node_type import NTWallTMS
from sailfish.lb_single import LBFluidSim
from sailfish.controller import LBSimulationController
from sailfish.sym import D2Q9, D3Q15, relaxation_time
from sailfish.sym_equilibrium import bgk_equilibrium
from sailfish.config import LBConfig


vi = lambda x, y: D2Q9.vec_idx([x, y])
vi3 = lambda x, y, z: D3Q15.vec_idx([x, y, z])

fi_start_2d = {
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

fi_start_3d = {
    vi3(0, 0, 0): 0.2218,
    vi3(1, 0, 0): 0.11771,
    vi3(-1, 0, 0): 0.10437,
    vi3(0, 1, 0): 0.11094,
    vi3(0, -1, 0): 0.110944,
    vi3(0, 0, 1): 0.124677,
    vi3(0, 0, -1): 0.098011,
    vi3(1, 1, 1): 0.016480,
    vi3(-1, 1, 1): 0.014713,
    vi3(1, -1, 1): 0.016480,
    vi3(-1, -1, 1): 0.014713,
    vi3(1, 1, -1): 0.013047,
    vi3(-1, 1, -1): 0.011480,
    vi3(1, -1, -1): 0.013047,
    vi3(-1, -1, -1): 0.011480,
}


class Test2DSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        wall_map = (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, NTWallTMS)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[:] = 0.05


class Test2DSim(LBFluidSim):
    subdomain = Test2DSubdomain

    @classmethod
    def modify_config(cls, config):
        config.propagation_enabled = False

    def initial_conditions(self, runner):
        dist = runner._debug_get_dist()
        dist[:] = 0.0

        for k, v in fi_start_2d.iteritems():
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

        ctrl = LBSimulationController(Test2DSim, default_config=settings)
        ctrl.run(ignore_cmdline=True)
        runner = ctrl.master.runner
        dist = runner._debug_get_dist()

        rho_bb = 0.0
        ux_bb = 0.0
        uy_bb = 0.0

        for k, v in fi_start_2d.iteritems():
            if D2Q9.basis[k][1] == 1:
                v = fi_start_2d[D2Q9.idx_opposite[k]]

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

        for k, v in fi_start_2d.iteritems():
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

        for k, v in fi_start_2d.iteritems():
            if D2Q9.basis[k][1] == 1:
                fneq[k] = (eq[k].evalf(subs={'g0m0': rho_bb, 'g0m1x': ux_bb, 'g0m1y': uy_bb}) -
                           eq[k].evalf(subs={'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy}))
            else:
                fneq[k] = v - eq[k].evalf(subs={'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy})

        tau = relaxation_time(1.0/12.0)
        omega = 1.0 / tau

        res = {}
        for k, v in fi_start_2d.iteritems():
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

# =============================================================================

class Test3DSubdomain(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        wall_map = (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, NTWallTMS)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        sim.vx[:] = 0.05


class Test3DSim(LBFluidSim):
    subdomain = Test3DSubdomain

    @classmethod
    def modify_config(cls, config):
        config.propagation_enabled = False

    def initial_conditions(self, runner):
        dist = runner._debug_get_dist()
        dist[:] = 0.0

        for k, v in fi_start_3d.iteritems():
            dist[k, 4, 1, 16] = v

        runner._debug_set_dist(dist)
        runner._debug_set_dist(dist, False)


class Test3DTMS(unittest.TestCase):
    nx = 64
    ny = 16
    nz = 16

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
            'lat_nz': self.nz,
            'grid': 'D3Q15',
            'save_src': '/tmp/test.cu',
            'max_iters': 1,
            'visc': 1.0/12.0,
        }

        ctrl = LBSimulationController(Test3DSim, default_config=settings)
        ctrl.run(ignore_cmdline=True)
        runner = ctrl.master.runner
        dist = runner._debug_get_dist()

        rho_bb = 0.0
        ux_bb = 0.0
        uy_bb = 0.0
        uz_bb = 0.0

        for k, v in fi_start_3d.iteritems():
            if D3Q15.basis[k][1] == 1:
                v = fi_start_3d[D3Q15.idx_opposite[k]]

            rho_bb += v
            ux_bb += D3Q15.basis[k][0] * v
            uy_bb += D3Q15.basis[k][1] * v
            uz_bb += D3Q15.basis[k][2] * v

        ux_bb /= rho_bb
        uy_bb /= rho_bb
        uz_bb /= rho_bb

        rho = 0.0
        ux = 0.0
        uy = 0.0
        uz = 0.0

        cfg = LBConfig()
        cfg.incompressible = False
        cfg.minimize_roundoff = False
        eq = bgk_equilibrium(D3Q15, cfg).expression

        bb_subs = {'g0m0': rho_bb, 'g0m1x': ux_bb, 'g0m1y': uy_bb, 'g0m1z': uz_bb}

        for k, v in fi_start_3d.iteritems():
            if D3Q15.basis[k][1] == 1:
                v = eq[k].evalf(subs=bb_subs)

            rho += v
            ux += D3Q15.basis[k][0] * v
            uy += D3Q15.basis[k][1] * v
            uz += D3Q15.basis[k][2] * v

        ux /= rho
        uy /= rho
        uz /= rho

        print 'Target values are rho=%e, ux=%e, uy=%e, uz=%e' % (rho_bb, ux_bb, uy_bb, uz_bb)
        print 'Instantaneous values are rho=%e, ux=%e, uy=%e, uz=%e' % (rho, ux, uy, uz)

        fneq = {}
        inst_subs = {'g0m0': rho, 'g0m1x': ux, 'g0m1y': uy, 'g0m1z': uz}

        for k, v in fi_start_3d.iteritems():
            if D3Q15.basis[k][1] == 1:
                fneq[k] = (eq[k].evalf(subs=bb_subs) -
                           eq[k].evalf(subs=inst_subs))
            else:
                fneq[k] = v - eq[k].evalf(subs=inst_subs)

        tau = relaxation_time(1.0/12.0)
        omega = 1.0 / tau

        res = {}
        for k, v in fi_start_3d.iteritems():
            if D3Q15.basis[k][1] == 1:
                res[k] = ((1.0 - (omega - 1)) *
                          eq[k].evalf(subs=bb_subs) +
                          (omega - 1) * eq[k].evalf(subs=inst_subs))
            else:
                res[k] = (v + omega * (eq[k].evalf(subs=inst_subs) - v) +
                          eq[k].evalf(subs=bb_subs) - eq[k].evalf(subs=inst_subs))

        for k, v in res.iteritems():
            np.testing.assert_allclose(dist[k, 4, 1, 16], np.float64(v))


if __name__ == '__main__':
    unittest.main()

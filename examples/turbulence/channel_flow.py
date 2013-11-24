#!/usr/bin/env python
"""Flow between two parallel plates driven by a body force.

The simulation is optimized for distributed runs (flow along the Z
direction so that copying data between subdomains is fast).
"""

import time
import math
import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim
from sailfish.node_type import NTWallTMS, NTHalfBBWall, NTFullBBWall
from sailfish.stats import ReynoldsStatsMixIn

import scipy.ndimage.filters

# Channel half-height.
H = 40
Re_tau = 395

# Max velocity.
u0 = 0.05
u_tau = u0 / (2.5 * math.log(Re_tau) + 6)
visc = u_tau * H / Re_tau

NX = 2 * H  # wall normal
NY = 2 * H  # spanwise (PBC)
NZ = 6 * H  # streamwise

# Literature stability limits:
# 2.3 max stability with bounce back
# 3.0 shocks with ELBM
print 'Delta = %e' % (u_tau / visc)
print 'visc = %e' % visc
print 'u = %e' % u_tau
print 'force = %e' % (Re_tau**2 * visc**2 / H**3)


class ChannelSubdomain(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        wall_map = ((hx == 0) | (hx == self.gx - 1))
        self.set_node(wall_map, NTHalfBBWall)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

        y_plus = (H - np.abs(hx + 0.5 - H)) * u_tau / visc
        # Log-law.
        u = (1/0.41 * np.log(y_plus) + 5.5) * u_tau
        # Linear scaling close to the wall.
        u[y_plus < 11.6] = y_plus[y_plus < 11.6] * u_tau
        sim.vz[:] = u

        # Determine subdomain span.
        x0, x1 = np.min(hx), np.max(hx)
        y0, y1 = np.min(hy), np.max(hy)
        z0, z1 = np.min(hz), np.max(hz)

        # Buffer size (used to make the random perturbation continuous
        # along the streamwise direction.
        B = 40
        hB = B / 2

        def _make_gradients():
            n1 = np.random.random((NZ + B, NY, NX)).astype(np.float32) * 2.0 - 1.0
            # Make the field continous along the streamwise direction.
            n1[-hB:,:,:] = n1[hB:B,:,:]
            n1[:hB,:,:] = n1[-B:-hB,:,:]
            nn1 = scipy.ndimage.filters.gaussian_filter(n1, 5)
            return np.gradient(nn1)

        np.random.seed(11341351351)
        _, dy1, dz1 = _make_gradients()
        dx2, _, dz2 = _make_gradients()
        dx3, dy3, _ = _make_gradients()

        # Compute curl of the random field.
        dvx = dy3 - dz2
        dvy = dz1 - dx3
        dvz = dx2 - dy1

        # Remove the buffer layer.
        dvx = dvx[hB:-hB,:,:]
        dvy = dvy[hB:-hB,:,:]
        dvz = dvz[hB:-hB,:,:]

        assert np.sum(np.isnan(dvx)) == 0
        assert np.sum(np.isnan(dvy)) == 0
        assert np.sum(np.isnan(dvz)) == 0

        scale = np.max([np.max(np.abs(dvx)), np.max(np.abs(dvy)), np.max(np.abs(dvz))])

        # Select the right part of the random field for this subdomain.
        dvx = dvx[z0:z1+1, y0:y1+1, x0:x1+1]
        dvy = dvy[z0:z1+1, y0:y1+1, x0:x1+1]
        dvz = dvz[z0:z1+1, y0:y1+1, x0:x1+1]

        # Add random perturbation to the initial flow field. The numerical
        # factor deterines the largest perturbation value. We also force
        # the perturbations to be smaller close to the wall.
        sim.vx[:] += dvx / scale * 0.05 * u / u0
        sim.vy[:] += dvy / scale * 0.05
        sim.vz[:] += dvz / scale * 0.05  # streamwise


class ChannelSim(LBFluidSim, LBForcedSim, ReynoldsStatsMixIn):
    subdomain = ChannelSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': NX,
            'lat_ny': NY,
            'lat_nz': NZ,
            'access_pattern': 'AA',
            'grid': 'D3Q19',
            'force_implementation': 'guo',
            'model': 'bgk',
            'minimize_roundoff': True,
            'visc': visc,
            'seed': 11341351351,
            'block_size': 128,
            'precision': 'single',
            'check_invalid_results_gpu': False,
            #'cuda_minimize_cpu': True,
            #'cuda_sched_yield': True,
            'cuda_disable_l1': True,
            'conn_axis': 'z',
            'max_iters': 3500000,
            'every': 200000,
            'perf_stats_every': 5000,
            'periodic_y': True,
            'periodic_z': True,
            'output': 'bgk395_hbb_120z',
            'final_checkpoint': True,
            'checkpoint_file': 'bgk395_hbb_120z',
            #'restore_from': 'bgk_hbb_moser_dbl.0200000',
            'checkpoint_every': 200000,
            'subdomains': 2,
            'cluster_interface': 'ib0'
            })

    def __init__(self, config):
        super(ChannelSim, self).__init__(config)
        self.add_body_force((0.0, 0.0, 0.99 * Re_tau**2 * visc**2 / H**3))

    def before_main_loop(self, runner):
        self.prepare_reynolds_stats(runner, axis='x')

    def after_step(self, runner):
        if self.iteration < 1500000:
           return

        every = 20
        mod = self.iteration % every

        if mod == every - 1:
            self.need_fields_flag = True
        elif mod == 0:
            stats = self.collect_reynolds_stats(runner)
            if stats is not None:
                np.savez('%s_reyn_stat_%s.%s' % (self.config.output, runner._spec.id,
                                                 self.iteration),
                         stats)

if __name__ == '__main__':
    ctrl = LBSimulationController(ChannelSim, EqualSubdomainsGeometry3D)
    ctrl.run()

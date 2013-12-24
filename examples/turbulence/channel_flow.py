#!/usr/bin/env python
# coding=utf-8
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
from sailfish.vis_mixin import Vis2DSliceMixIn

import scipy.ndimage.filters

class ChannelSubdomain(Subdomain3D):
    wall_bc = NTHalfBBWall
    u0 = 0.05

    @classmethod
    def u_tau(cls, Re_tau):
        # Max velocity is attained at the center of the channel. The center of the
        # channel in wall units is at Re_tau.
        return cls.u0 / (1.0/0.41 * math.log(Re_tau) + 5.5)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        self.set_profile(sim, hx, hy, hz, self.gx, self.gy, self.gz)

    def boundary_conditions(self, hx, hy, hz):
        wall_map = ((hx == 0) | (hx == self.gx - 1))
        self.set_node(wall_map, self.wall_bc)

    def make_gradients(self, NX, NY, NZ):
        # Buffer size (used to make the random perturbation continuous
        # along the streamwise direction.
        B = 40
        hB = B / 2

        n1 = np.random.random((NZ + B, NY + B, NX)).astype(np.float32) * 2.0 - 1.0
        # Make the field continous along the streamwise and spanwise direction.
        n1[-hB:,:,:] = n1[hB:B,:,:]
        n1[:hB,:,:] = n1[-B:-hB,:,:]

        n1[:,-hB:,:] = n1[:,hB:B,:]
        n1[:,:hB,:] = n1[:,-B:-hB,:]

        nn1 = scipy.ndimage.filters.gaussian_filter(n1, 5)
        # Remove the buffer layer.
        return [x[hB:-hB,hB:-hB,:] for x in np.gradient(nn1)]

    def select_subdomain(self, field, hx, hy, hz):
        # Determine subdomain span.
        x0, x1 = np.min(hx), np.max(hx)
        y0, y1 = np.min(hy), np.max(hy)
        z0, z1 = np.min(hz), np.max(hz)
        return field[z0:z1+1, y0:y1+1, x0:x1+1]

    def set_profile(self, sim, hx, hy, hz, NX, NY, NZ, pert=0.03):
        H = NX / 2

        u_tau = self.u_tau(self.config.Re_tau)
        hhx = np.abs(hx - self.wall_bc.location - H)
        # Sanity checks.
        assert np.all((H - hhx)[hx == 0] == -self.wall_bc.location)

        y_plus = (H - hhx) * u_tau / self.config.visc
        # Log-law.
        u = (1/0.41 * np.log(y_plus) + 5.5) * u_tau
        # Linear scaling close to the wall. y0 is chosen to make
        # the profile continuous.
        y0 = 11.44532166
        u[y_plus < y0] = y_plus[y_plus < y0] * u_tau
        sim.vz[:] = u

        np.random.seed(11341351351)
        _, dy1, dz1 = self.make_gradients(NX, NY, NZ)
        dx2, _, dz2 = self.make_gradients(NX, NY, NZ)
        dx3, dy3, _ = self.make_gradients(NX, NY, NZ)

        # Compute curl of the random field.
        dvx = dy3 - dz2
        dvy = dz1 - dx3
        dvz = dx2 - dy1

        assert np.sum(np.isnan(dvx)) == 0
        assert np.sum(np.isnan(dvy)) == 0
        assert np.sum(np.isnan(dvz)) == 0

        scale = np.max([np.max(np.abs(dvx)), np.max(np.abs(dvy)), np.max(np.abs(dvz))])

        # Select the right part of the random field for this subdomain.
        dvx = self.select_subdomain(dvx, hx, hy, hz)
        dvy = self.select_subdomain(dvy, hx, hy, hz)
        dvz = self.select_subdomain(dvz, hx, hy, hz)

        # Add random perturbation to the initial flow field. The numerical
        # factor determines the largest perturbation value. We also force
        # the perturbations to be smaller close to the wall in the wall-normal
        # direction.
        sim.vx[:] += dvx / scale * pert * u / self.u0
        sim.vy[:] += dvy / scale * pert * u / self.u0
        sim.vz[:] += dvz / scale * pert * u / self.u0  # streamwise


class ChannelSim(LBFluidSim, LBForcedSim, ReynoldsStatsMixIn, Vis2DSliceMixIn):
    subdomain = ChannelSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'access_pattern': 'AA',
            'grid': 'D3Q19',
            'force_implementation': 'guo',
            'model': 'bgk',
            'minimize_roundoff': True,
            'precision': 'single',
            'seed': 11341351351,
            'periodic_y': True,
            'periodic_z': True,

            # Performance tuning.
            'check_invalid_results_gpu': False,
            'block_size': 128,
            'conn_axis': 'z',

            # Output.
            'max_iters': 3500000,
            'every': 200000,
            'perf_stats_every': 5000,
            })

    @classmethod
    def modify_config(cls, config):
        h = 2 if ChannelSubdomain.wall_bc.location == 0.5 else 0

        az = 6
        ay = 2

        # (2, 4, 12) * H is approximately the reference geometry from Moser.
        config.lat_nx = config.H * 2 + h  # wall normal
        config.lat_ny = config.H * ay  # spanwise (PBC)
        config.lat_nz = config.H * az  # streamwise (PBC)
        config.visc = ChannelSubdomain.u_tau(config.Re_tau) * config.H / config.Re_tau

        u_tau = ChannelSubdomain.u_tau(config.Re_tau)
        Re = ChannelSubdomain.u0 * 2.0 * config.H / config.visc
        print 'Delta_+ = %.2f' % (u_tau / config.visc)
        print 'Re_tau = %.2f' % (config.Re_tau)
        print 'Re = %.2f' % Re
        print 'visc = %e' % config.visc
        print 'u_tau = %e' % u_tau
        print 'eta = %e' % (2.0 * config.H / Re**0.75)  # Kolmogorov scale
        print 'force = %e' % (config.Re_tau**2 * config.visc**2 / config.H**3)

        # Timescales: large eddies, flow-through time in the wall layer.
        print 't_eddy = %d' % (config.H * 2.0 / ChannelSubdomain.u0)
        print 't_flow = %d' % (config.H / u_tau * az)

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)
        Vis2DSliceMixIn.add_options(group, dim)
        group.add_argument('--H', type=int, default=40, help='channel half-height')
        group.add_argument('--Re_tau', type=float, default=180.0, help='Re_tau')

    def __init__(self, config):
        super(ChannelSim, self).__init__(config)
        self.add_body_force((0.0, 0.0, config.Re_tau**2 * config.visc**2 /
                             config.H**3))

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

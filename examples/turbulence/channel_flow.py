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
    wall_bc = NTFullBBWall
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

    def make_gradients(self, NX, NY, NZ, hx, hy, hz, u):
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

        nn1 = scipy.ndimage.filters.gaussian_filter(n1, 5 * self.config.H / 40)
        # Remove the buffer layer. We also force the perturbations to be
        # smaller close to the wall. Select the right part of the random
        # field for this subdomain.
        return [self.select_subdomain(x[hB:-hB,hB:-hB,:], hx, hy, hz) * u / self.u0 for x in np.gradient(nn1)]

    def select_subdomain(self, field, hx, hy, hz):
        # Determine subdomain span.
        x0, x1 = np.min(hx), np.max(hx)
        y0, y1 = np.min(hy), np.max(hy)
        z0, z1 = np.min(hz), np.max(hz)
        return field[z0:z1+1, y0:y1+1, x0:x1+1]

    def set_profile(self, sim, hx, hy, hz, NX, NY, NZ, pert=0.03):
        H = self.config.H

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
        _, dy1, dz1 = self.make_gradients(NX, NY, NZ, hx, hy, hz, u)
        dx2, _, dz2 = self.make_gradients(NX, NY, NZ, hx, hy, hz, u)
        dx3, dy3, _ = self.make_gradients(NX, NY, NZ, hx, hy, hz, u)

        # Compute curl of the random field.
        dvx = dy3 - dz2
        dvy = dz1 - dx3
        dvz = dx2 - dy1

        assert np.sum(np.isnan(dvx)) == 0
        assert np.sum(np.isnan(dvy)) == 0
        assert np.sum(np.isnan(dvz)) == 0

        scale = np.max([np.max(np.abs(dvx)), np.max(np.abs(dvy)), np.max(np.abs(dvz))])

        # Add random perturbation to the initial flow field. The numerical
        # factor determines the largest perturbation value.
        sim.vx[:] += dvx / scale * pert
        sim.vy[:] += dvy / scale * pert
        sim.vz[:] += dvz / scale * pert  # streamwise


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
            'final_checkpoint': True,
            'checkpoint_every': 500000,
            })

    @classmethod
    def modify_config(cls, config):
        h = 2 if cls.subdomain.wall_bc.location == 0.5 else 0

        az = 6
        ay = 2

        # (2, 4, 12) * H is approximately the reference geometry from Moser.
        config.lat_nx = config.H * 2 + h  # wall normal
        config.lat_ny = config.H * ay  # spanwise (PBC)
        config.lat_nz = config.H * az  # streamwise (PBC)
        config.visc = cls.subdomain.u_tau(config.Re_tau) * config.H / config.Re_tau

        # Show data early. This is helpful for quick debugging.
        print '\n'.join(cls.get_info(config))

    @classmethod
    def get_info(cls, config):
        u_tau = cls.subdomain.u_tau(config.Re_tau)
        Re = cls.subdomain.u0 * config.H / config.visc

        y_plus = (np.arange(config.H) + 0.5) * u_tau / config.visc
        u = (1/0.41 * np.log(y_plus) + 5.5) * u_tau
        u_bulk = np.sum(u) / config.H
        Re_bulk = u_bulk * config.H / config.visc

        ret = []
        ret.append('Delta_+ = %.2f' % (u_tau / config.visc))
        ret.append('Re_tau = %.2f' % (config.Re_tau))
        ret.append('Re_H,max = %.2f' % Re)
        ret.append('Re_H,bulk = %.2f' % Re_bulk)
        ret.append('visc = %e' % config.visc)
        ret.append('u_b = %e' % u_bulk)
        ret.append('u_tau = %e' % u_tau)
        ret.append('eta = %e' % (2.0 * config.H / Re**0.75))  # Kolmogorov scale
        ret.append('force = %e' % (config.Re_tau**2 * config.visc**2 /
                                   config.H**3))

        # Timescales: large eddies, flow-through time in the wall layer.
        ret.append('t_eddy = %d' % (config.H * 2.0 / cls.subdomain.u0))
        ret.append('t_flow = %d' % cls.t_flow(config))
        ret.append('t_char = %d' % cls.t_char(config))
        return ret

    @classmethod
    def t_flow(cls, config):
        """Flow-through time."""
        return cls.t_char(config) * (config.lat_nz / config.H)

    @classmethod
    def t_char(cls, config):
        """Characteristic time."""
        u_tau = cls.subdomain.u_tau(config.Re_tau)
        return config.H / u_tau

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--H', type=int, default=40, help='channel half-height')
        group.add_argument('--Re_tau', type=float, default=180.0, help='Re_tau')

    def __init__(self, config):
        super(ChannelSim, self).__init__(config)
        # force = u_tau^2 / H
        self.add_body_force((0.0, 0.0, config.Re_tau**2 * config.visc**2 /
                             config.H**3))

        for line in self.get_info(self.config):
            self.config.logger.info(line)

    def before_main_loop(self, runner):
        self.prepare_reynolds_stats(runner, axis='x')

    def after_step(self, runner):
        # Ignore transients.
        if self.iteration < 2 * self.t_flow(self.config):
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
                         **stats)

if __name__ == '__main__':
    ctrl = LBSimulationController(ChannelSim, EqualSubdomainsGeometry3D)
    ctrl.run()

#!/usr/bin/env python
"""
Turbulent channel flow around a wall-mounted cube.

The simulation is driven by a fully developed turbulent channel
flow simulated in a separate subdomain (id=0), called the recirculation
buffer. This subdomain has periodic boundary conditions enabled in the
streamwise direction and is completely independent of the main simulation
area.

  buffer:               main:
=============|==========================O
P            >                          O
P            >                          O
P            >                          O
P            >                          O
=============|==========================O

Legend:
 | - subdomain boundary
 P - PBC within the buffer subdomain
 > - PBC within the buffer subdomain; data is transferred from buffer to main,
     but not vice-versa
 = - wall
 R - replicated node (all distributions are synced from buffer to main after
     every step
 O - outflow nodes

Reference data is for Re_m = 5610. The original geometry is (current values are
provided in the :
 X - streamwise (Z)
 Y - channel height (X)
 Z - spanwise (Y)

Available measurements:
 - streamlines / pressure in the X-Y symmetry plane
 - streamlines / pressure in the X-Z plane at y/h = 0.003
 - streamlines in the X-Z plane at y/h = 0.1, 0.25, 0.5, 0.75
 - streamwise vorticity in the Y-Z plane at x/h = 3.11, 3.50, 3.91 (horse-shoe
   vortex)
 - streamwise velocity in the symmetry plane at y/h = 0.1, 0.31, 0.5, 0.69, 0.9
 - power spectrum of the spanwise velocity in the rear wake
 - Reynolds stresses in the X-Y symmetry plane (u'^2, v'^2, w'^2)
 - turbulent kinetic energy in the X-Z plane at y/h = 0.1, 0.25, 0.50, 0.75
   TKE = 0.5 (u'^2 + v'^2 + w'^2)
"""

import math
import numpy as np
from sailfish.node_type import NTHalfBBWall, NTDoNothing, NTCopy, NTEquilibriumDensity, NTFullBBWall
from sailfish.geo import LBGeometry3D
from sailfish.controller import LBSimulationController, GeometryError
from sailfish.subdomain import Subdomain3D, SubdomainSpec3D
from sailfish.stats import ReynoldsStatsMixIn
from sailfish.subdomain_runner import SubdomainRunner
from sailfish.lb_single import LBFluidSim
from sailfish.lb_base import LBForcedSim
from sailfish.sym import D3Q19

from channel_flow import ChannelSubdomain, ChannelSim
import scipy.ndimage.filters


class CubeChannelGeometry(LBGeometry3D):
    @classmethod
    def add_options(cls, group):
        LBGeometry3D.add_options(group)
        group.add_argument('--subdomains', help='number of subdomains for '
                           'the real simulation region',
                           type=int, default=1)
    @staticmethod
    def cube_h(config):
        return config.H * 2 / 3

    @staticmethod
    def buf_nz(config):
        return int(config.buf_az * CubeChannelGeometry.cube_h(config))

    @staticmethod
    def main_nz(config):
        return int(config.main_az * CubeChannelGeometry.cube_h(config))

    def subdomains(self):
        c = self.config

        # Recirculation buffer.
        buf = SubdomainSpec3D((0, 0, 0), (c.lat_nx, c.lat_ny, self.buf_nz(c)))
        # Enable PBC along the Z axis.
        buf.enable_local_periodicity(2)
        ret = [buf]

        # Actual simulation domain.
        n = self.config.subdomains
        z = self.buf_nz(c)
        dz = self.main_nz(c) / n
        rz = self.main_nz(c) % n
        for i in range(0, n):
            ret.append(SubdomainSpec3D((0, 0, z),
                                       (c.lat_nx, c.lat_ny, dz if i < n - 1 else dz + rz)))
            z += dz
        return ret


class CubeChannelSubdomain(ChannelSubdomain):
    wall_bc = NTFullBBWall
    u0 = 0.025

    def boundary_conditions(self, hx, hy, hz):
        # Channel walls.
        wall_map = ((hx == 0) | (hx == self.gx - 1))
        self.set_node(wall_map, self.wall_bc)

        h = self.config.H * 2 / 3
        buf_len = CubeChannelGeometry.buf_nz(self.config)

        # Cube.
        cube_map = ((hx > 0) & (hx < h) &
                    (hz >= buf_len + 3 * h) & (hz < buf_len + 4 * h) &
                    (hy >= 2.7 * h) & (hy < 3.7 * h))
        self.set_node(cube_map, self.wall_bc)

        # Outlet
        outlet_map = (hz == self.gz - 1) & np.logical_not(wall_map)
        self.set_node(outlet_map, NTEquilibriumDensity(1.0,
                                                      orientation=D3Q19.vec_to_dir([0,0,-1])))


class CubeChannelSubdomainRunner(SubdomainRunner):
    def _init_distrib_kernels(self, *args, **kwargs):
        # No distribution in the recirculation buffer.
        if self._spec.id == 0:
            return [], []
        else:
            return super(CubeChannelSubdomainRunner, self)._init_distrib_kernels(
                *args, **kwargs)


class CubeChannelSim(ChannelSim):
    subdomain = CubeChannelSubdomain
    subdomain_runner = CubeChannelSubdomainRunner

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
            'check_invalid_results_gpu': True,
            'block_size': 128,

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

        # 1/3 of the channel height
        cube_h = CubeChannelGeometry.cube_h(config)

        config.lat_nx = config.H * 2 + h  # wall normal
        config.lat_ny = int(config.ay * cube_h)  # spanwise (PBC)
        config.lat_nz = (int(config.buf_az * cube_h) +
                         int(config.main_az * cube_h))  # streamwise
        config.visc = cls.subdomain.u_tau(config.Re_tau) * config.H / config.Re_tau

        cls.show_info(config)

    @classmethod
    def show_info(cls, config):
        cube_h = config.H * 2 / 3
        print 'cube:   %d' % cube_h
        print 'buffer: %d x %d x %d' % (int(config.buf_az * cube_h), config.lat_ny, config.lat_nx)
        print 'main:   %d x %d x %d' % (int(config.main_az * cube_h), config.lat_ny, config.lat_nx)
        ChannelSim.show_info(config)

    @classmethod
    def add_options(cls, group, dim):
        ChannelSim.add_options(group, dim)
        # The reference DNS simulation uses: 9, 14, 6.4, respectively.
        group.add_argument('--buf_az', type=float, default=9.0)
        group.add_argument('--main_az', type=float, default=14.0)
        group.add_argument('--ay', type=float, default=6.4)

    def before_main_loop(self, runner):
        self._prepare_reynolds_stats_global(runner)

    def _prepare_reynolds_stats_global(self, runner):
        # All components of the Reynolds stress tensor and mean velocities.
        num_stats = 3 * 2 + 3
        self._stats = []
        self._gpu_stats = []

        for i in range(0, num_stats):
            f, _ = runner.make_scalar_field(dtype=np.float64, register=False)
            f[:] = 0.0
            self._stats.append(f)
            self._gpu_stats.append(runner.backend.alloc_buf(like=runner.field_base(f)))

        gpu_v = runner.gpu_field(self.v)
        self._stats_kern = runner.get_kernel(
            'ReynoldsGlobal', gpu_v + self._gpu_stats, 'PPP' + 'P' * num_stats)

    max_stats = 10000
    num_stats = 0
    def _collect_stats(self, runner):
        runner.backend.run_kernel(self._stats_kern, runner._kernel_grid_full)
        self.num_stats += 1

        # Periodically dump the data.
        if self.num_stats == self.max_stats:
            for gpu_buf in self._gpu_stats:
                runner.backend.from_buf(gpu_buf)

            # The order of the statistics collected here has to match the
            # definition in reynolds_statistics.mako.
            np.savez('%s_reyn_stat_%s.%s' % (self.config.output, runner._spec.id,
                                             self.iteration),
                     ux_m1=self._stats[0] / self.num_stats,
                     ux_m2=self._stats[1] / self.num_stats,
                     uy_m1=self._stats[2] / self.num_stats,
                     uy_m2=self._stats[3] / self.num_stats,
                     uz_m1=self._stats[4] / self.num_stats,
                     uz_m2=self._stats[5] / self.num_stats,
                     ux_uy=self._stats[6] / self.num_stats,
                     ux_uz=self._stats[7] / self.num_stats,
                     uy_uz=self._stats[8] / self.num_stats)

            self.num_stats = 0
            for buf in self._stats:
                buf[:] = 0.0
            for gpu_buf in self._gpu_stats:
                runner.backend.to_buf(gpu_buf)

    def after_step(self, runner):
        # Allow 2 flow-through times to remove transients.
        transients = 2 * self.t_flow(self.config)
        transients = ((transients + self.max_stats - 1) / self.max_stats) * self.max_stats

        # Do not generate stats for the recirculation buffer.
        if self.iteration < transients or runner._spec.id == 0:
            return

        # How often to collect data for the Reynolds stats.
        every = 10
        mod = self.iteration % every

        if mod == every - 1:
            self.need_fields_flag = True
        elif mod == 0:
            self._collect_stats(runner)


if __name__ == '__main__':
    ctrl = LBSimulationController(CubeChannelSim, CubeChannelGeometry)
    ctrl.run()

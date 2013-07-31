#!/usr/bin/env python
"""
Flow between two parallel plates driven by a body force.
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

import scipy.ndimage.filters

# Channel half-height.
H = 120
Re = 395
u0 = 0.05
u_tau = u0 / (2.5 * math.log(Re) + 6)
visc = u_tau * H / Re

NX = 2 * H  # z
NY = 2 * H  # y
NZ = 6 * H  # x -> flow is along this axis

# 2.3 max stability with bounce back
# 3.0 shocks with ELBM
print 'visc = %e' % visc
print 'Delta = %e' % (u_tau / visc)
print 'u = %e' % u_tau
print 'force = %e' % (Re**2 * visc**2 / H**3)
print 'force2 = %e' % (u_tau**2 / H)

class ChannelSubdomain(Subdomain3D):
    def boundary_conditions(self, hx, hy, hz):
        wall_map = ((hx == 0) | (hx == self.gx - 1))
        self.set_node(wall_map, NTHalfBBWall)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

        y_plus = (H - np.abs(hx + 0.5 - H)) * u_tau / visc
        u = (1/0.41 * np.log(y_plus) + 5.5) * u_tau
        u[y_plus < 11.6] = y_plus[y_plus < 11.6] * u_tau
        sim.vz[:] = u

        x0 = np.min(hx)
        x1 = np.max(hx)
        y0 = np.min(hy)
        y1 = np.max(hy)
        z0 = np.min(hz)
        z1 = np.max(hz)

        np.random.seed(11341351351) # + 123479337 * (H * 3 - 1))

        def _make_gradients():
            n1 = np.random.random((6 * H + 40, 2 * H, 2 * H)).astype(np.float32) * 2.0 - 1.0
            n1[-20:,:,:] = n1[20:40,:,:]
            n1[:20,:,:] = n1[-40:-20,:,:]
            nn1 = scipy.ndimage.filters.gaussian_filter(n1, 5)
            return np.gradient(nn1)

        _, dy1, dz1 = _make_gradients()
        dx2, _, dz2 = _make_gradients()
        dx3, dy3, _ = _make_gradients()
        print 'gradients done'

        dvx = dy3 - dz2
        dvy = dz1 - dx3
        dvz = dx2 - dy1

        dvx = dvx[20:-20,:,:]
        dvy = dvy[20:-20,:,:]
        dvz = dvz[20:-20,:,:]

        assert np.sum(np.isnan(dvx)) == 0
        assert np.sum(np.isnan(dvy)) == 0
        assert np.sum(np.isnan(dvz)) == 0

        scale = np.max([np.max(np.abs(dvx)), np.max(np.abs(dvy)), np.max(np.abs(dvz))])

        _, _, hhx = np.mgrid[0:6 * H, 0:2*H, 0:2*H]
        dvx = dvx[z0:z1+1, y0:y1+1, x0:x1+1]
        dvy = dvy[z0:z1+1, y0:y1+1, x0:x1+1]
        dvz = dvz[z0:z1+1, y0:y1+1, x0:x1+1]
        hhx = hhx[z0:z1+1, y0:y1+1, x0:x1+1]

        print 'grid built'

        # Remember the 0.5 shift here to avoid inf values!
        y_plus = (H - np.abs(hhx + 0.5 - H)) * u_tau / visc
        u = (1/0.41 * np.log(y_plus) + 5.5) * u_tau

        sim.vx[:] += dvx / scale * 0.05
        sim.vy[:] += dvy / scale * 0.05 * u / u0
        sim.vz[:] += dvz / scale * 0.05  # streamwise

class ChannelSim(LBFluidSim, LBForcedSim):
    subdomain = ChannelSubdomain
    aux_code = ['reynolds_statistics.mako']

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
            'verbose': True,
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
        self.add_body_force((0.0, 0.0, 0.99 * Re**2 * visc**2 / H**3))

    # Number of copies of the stats buffers to keep in GPU memory
    # between host syncs.
    stat_buf_size = 1024
    def before_main_loop(self, runner):
        def _alloc_stat_buf(name, helper_size, need_finalize):
            h = np.zeros([self.stat_buf_size, NX], dtype=np.float64)
            setattr(self, 'stat_%s' % name, h)
            setattr(self, 'gpu_stat_%s' % name, runner.backend.alloc_buf(like=h))

            if need_finalize:
                h = np.zeros([NX, helper_size], dtype=np.float64)
                setattr(self, 'stat_tmp_%s' % name, h)
                setattr(self, 'gpu_stat_tmp_%s' % name, runner.backend.alloc_buf(like=h))

        lat_nx = runner._lat_size[-1]

        # Buffers for moments of the hydrodynamic variables.
        cm_bs = 128   # block_size, keep in sync with template
        self.stat_cm_grid_size = (NX + 2 + cm_bs - 1) / cm_bs
        cm_finalize = False #lat_nx >= cm_bs
        _alloc_stat_buf('ux_m1', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('ux_m2', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('ux_m3', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('ux_m4', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uy_m1', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uy_m2', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uy_m3', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uy_m4', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uz_m1', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uz_m2', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uz_m3', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('uz_m4', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('rho_m1', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('rho_m2', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('rho_m3', self.stat_cm_grid_size, cm_finalize)
        _alloc_stat_buf('rho_m4', self.stat_cm_grid_size, cm_finalize)

        # Buffer for correlations of hydrodynamic variables.
        corr_bs = 128    # block_size, keep in sync with template
        self.stat_corr_grid_size = (NX + 2 + corr_bs - 1) / corr_bs
        corr_finalize = False #lat_nx >= corr_bs
        _alloc_stat_buf('ux_uy', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('ux_uz', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('uy_uz', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('ux_rho', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('uy_rho', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('uz_rho', self.stat_corr_grid_size, corr_finalize)

        gpu_rho = runner.gpu_field(self.rho)
        gpu_v = runner.gpu_field(self.v)
        args = gpu_v + [gpu_rho]

        # Save these as attributes so that they are accessible in
        # after_step().
        self.cm_finalize = cm_finalize
        self.corr_finalize = corr_finalize

        print 'corr_finalize is %s' % corr_finalize
        print 'cm_finalize is %s' % cm_finalize

        for field, name in zip(args, ('ux', 'uy', 'uz', 'rho')):
            if cm_finalize:
                setattr(self, 'stat_kern_cm_%s' % name,
                        runner.get_kernel(
                            'ReduceComputeMoments64X', [
                                field,
                                getattr(self, 'gpu_stat_tmp_%s_m1' % name),
                                getattr(self, 'gpu_stat_tmp_%s_m2' % name),
                                getattr(self, 'gpu_stat_tmp_%s_m3' % name),
                                getattr(self, 'gpu_stat_tmp_%s_m4' % name)],
                    'PPPPP', block_size=(cm_bs,), more_shared=True))
                nbs = int(pow(2, math.ceil(math.log(self.stat_cm_grid_size, 2))))
                setattr(self, 'stat_kern_cm_%s_fin' % name,
                        runner.get_kernel(
                    'FinalizeReduceComputeMoments64X', [
                        getattr(self, 'gpu_stat_tmp_%s_m1' % name),
                        getattr(self, 'gpu_stat_tmp_%s_m2' % name),
                        getattr(self, 'gpu_stat_tmp_%s_m3' % name),
                        getattr(self, 'gpu_stat_tmp_%s_m4' % name),
                        getattr(self, 'gpu_stat_%s_m1' % name),
                        getattr(self, 'gpu_stat_%s_m2' % name),
                        getattr(self, 'gpu_stat_%s_m3' % name),
                        getattr(self, 'gpu_stat_%s_m4' % name), 0],
                    'PPPPPPPPi', block_size=(nbs,), more_shared=True))
            else:
                setattr(self, 'stat_kern_cm_%s' % name,
                        runner.get_kernel(
                            'ReduceComputeMoments64X', [
                                field,
                                getattr(self, 'gpu_stat_%s_m1' % name),
                                getattr(self, 'gpu_stat_%s_m2' % name),
                                getattr(self, 'gpu_stat_%s_m3' % name),
                                getattr(self, 'gpu_stat_%s_m4' % name), 0],
                    'PPPPPi', block_size=(cm_bs,), more_shared=True))

        if corr_finalize:
            self.stat_kern_corr = runner.get_kernel(
                'ReduceComputeCorrelations64X', gpu_v + [
                    gpu_rho, self.gpu_stat_tmp_ux_uy, self.gpu_stat_tmp_ux_uz,
                    self.gpu_stat_tmp_uy_uz, self.gpu_stat_tmp_ux_rho,
                    self.gpu_stat_tmp_uy_rho, self.gpu_stat_tmp_uz_rho],
                'PPPPPPPPPP', block_size=(corr_bs,), more_shared=True)

            nbs = int(pow(2, math.ceil(math.log(self.stat_corr_grid_size, 2))))
            self.stat_kern_corr_fin = runner.get_kernel(
                'FinalizeReduceComputeCorrelations64X',
                [self.gpu_stat_tmp_ux_uy, self.gpu_stat_tmp_ux_uz,
                 self.gpu_stat_tmp_uy_uz, self.gpu_stat_tmp_ux_rho,
                 self.gpu_stat_tmp_uy_rho, self.gpu_stat_tmp_uz_rho,
                 self.gpu_stat_ux_uy, self.gpu_stat_ux_uz,
                 self.gpu_stat_uy_uz, self.gpu_stat_ux_rho,
                 self.gpu_stat_uy_rho, self.gpu_stat_uz_rho, 0],
                'PPPPPPPPPPPPi', block_size=(nbs,), more_shared=True)
        else:
            self.stat_kern_corr = runner.get_kernel(
                'ReduceComputeCorrelations64X', gpu_v + [
                    gpu_rho, self.gpu_stat_ux_uy, self.gpu_stat_ux_uz,
                    self.gpu_stat_uy_uz, self.gpu_stat_ux_rho,
                    self.gpu_stat_uy_rho, self.gpu_stat_uz_rho, 0],
                'PPPPPPPPPPi', block_size=(corr_bs,), more_shared=True)

    stat_cnt = 0
    def after_step(self, runner):
        if self.iteration < 1500000:
           return

        every = 20
        mod = self.iteration % every

        if mod == every - 1:
            self.need_fields_flag = True
        elif mod == 0:
            # Collect Reynolds statistics.
            # TODO(mjanusz): Run these kernels in a stream other than main.
            grid = [self.stat_cm_grid_size]
            self.stat_cnt += 1

            if self.cm_finalize:
                runner.backend.run_kernel(self.stat_kern_cm_ux, grid)
                runner.backend.run_kernel(self.stat_kern_cm_ux_fin, [1, NX])
                runner.backend.run_kernel(self.stat_kern_cm_uy, grid)
                runner.backend.run_kernel(self.stat_kern_cm_uy_fin, [1, NX])
                runner.backend.run_kernel(self.stat_kern_cm_uz, grid)
                runner.backend.run_kernel(self.stat_kern_cm_uz_fin, [1, NX])
                runner.backend.run_kernel(self.stat_kern_cm_rho, grid)
                runner.backend.run_kernel(self.stat_kern_cm_rho_fin, [1, NX])
            else:
                runner.backend.run_kernel(self.stat_kern_cm_ux, grid)
                runner.backend.run_kernel(self.stat_kern_cm_uy, grid)
                runner.backend.run_kernel(self.stat_kern_cm_uz, grid)
                runner.backend.run_kernel(self.stat_kern_cm_rho, grid)

            if self.corr_finalize:
                runner.backend.run_kernel(self.stat_kern_corr, [self.stat_corr_grid_size, NX])
                runner.backend.run_kernel(self.stat_kern_corr_fin, [1, NX])
            else:
                runner.backend.run_kernel(self.stat_kern_corr, [self.stat_corr_grid_size])

            # Stat buffer full?
            if self.stat_cnt == self.stat_buf_size:
                print 'preparing to dump stats'
                self.stat_cnt = 0
                if self.cm_finalize:
                    self.stat_kern_cm_ux_fin.args[-1] = 0
                    self.stat_kern_cm_uy_fin.args[-1] = 0
                    self.stat_kern_cm_uz_fin.args[-1] = 0
                    self.stat_kern_cm_rho_fin.args[-1] = 0
                else:
                    self.stat_kern_cm_ux.args[-1] = 0
                    self.stat_kern_cm_uy.args[-1] = 0
                    self.stat_kern_cm_uz.args[-1] = 0
                    self.stat_kern_cm_rho.args[-1] = 0

                if self.corr_finalize:
                    self.stat_kern_corr_fin.args[-1] = 0
                else:
                    self.stat_kern_corr.args[-1] = 0

                # Load stats from GPU memory.
                for stat in ('ux', 'uy', 'uz', 'rho'):
                    runner.backend.from_buf(getattr(self, 'gpu_stat_%s_m1' % stat))
                    runner.backend.from_buf(getattr(self, 'gpu_stat_%s_m2' % stat))
                    runner.backend.from_buf(getattr(self, 'gpu_stat_%s_m3' % stat))
                    runner.backend.from_buf(getattr(self, 'gpu_stat_%s_m4' % stat))

                runner.backend.from_buf(self.gpu_stat_ux_uy)
                runner.backend.from_buf(self.gpu_stat_ux_uz)
                runner.backend.from_buf(self.gpu_stat_uy_uz)
                runner.backend.from_buf(self.gpu_stat_ux_rho)
                runner.backend.from_buf(self.gpu_stat_uy_rho)
                runner.backend.from_buf(self.gpu_stat_uz_rho)

                # Divide the stats by this value to get an average over all nodes.
                div = NY * NZ
                np.savez('%s_stat_%s.%s' % (self.config.output, runner._spec.id,
                                            self.iteration),
                         ux_m1=self.stat_ux_m1 / div,
                         ux_m2=self.stat_ux_m2 / div,
                         ux_m3=self.stat_ux_m3 / div,
                         ux_m4=self.stat_ux_m4 / div,
                         uy_m1=self.stat_uy_m1 / div,
                         uy_m2=self.stat_uy_m2 / div,
                         uy_m3=self.stat_uy_m3 / div,
                         uy_m4=self.stat_uy_m4 / div,
                         uz_m1=self.stat_uz_m1 / div,
                         uz_m2=self.stat_uz_m2 / div,
                         uz_m3=self.stat_uz_m3 / div,
                         uz_m4=self.stat_uz_m4 / div,
                         rho_m1=self.stat_rho_m1 / div,
                         rho_m2=self.stat_rho_m2 / div,
                         rho_m3=self.stat_rho_m3 / div,
                         rho_m4=self.stat_rho_m4 / div,
                         ux_uy=self.stat_ux_uy / div,
                         ux_uz=self.stat_ux_uz / div,
                         uy_uz=self.stat_uy_uz / div,
                         ux_rho=self.stat_ux_rho / div,
                         uy_rho=self.stat_uy_rho / div,
                         uz_rho=self.stat_uz_rho / div)
            else:
                if self.cm_finalize:
                    self.stat_kern_cm_ux_fin.args[-1] += NX
                    self.stat_kern_cm_uy_fin.args[-1] += NX
                    self.stat_kern_cm_uz_fin.args[-1] += NX
                    self.stat_kern_cm_rho_fin.args[-1] += NX
                else:
                    self.stat_kern_cm_ux.args[-1] += NX
                    self.stat_kern_cm_uy.args[-1] += NX
                    self.stat_kern_cm_uz.args[-1] += NX
                    self.stat_kern_cm_rho.args[-1] += NX

                if self.corr_finalize:
                    self.stat_kern_corr_fin.args[-1] += NX
                else:
                    self.stat_kern_corr.args[-1] += NX

if __name__ == '__main__':
    ctrl = LBSimulationController(ChannelSim, EqualSubdomainsGeometry3D)
    ctrl.run()

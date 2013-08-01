"""Auxilliary classes for computing flow statistics."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from sailfish.lb_base import ScalarField, LBMixIn
import numpy as np

class FlowStatsMixIn(LBMixIn):
    """When mixed with an LBFluidSim-descendant class, provides easy access
    to various flow statistics"""


class KineticEnergyEnstrophyMixIn(FlowStatsMixIn):
    """Computes global kinetic energy and enstrophy densities."""
    aux_code = ['data_processing.mako']

    @classmethod
    def fields(cls):
        return [ScalarField('v_sq', gpu_array=True, init=0.0),
                ScalarField('vort_sq', gpu_array=True, init=0.0)]

    def before_main_loop(self, runner):
        gpu_map = runner.gpu_geo_map()
        gpu_v = runner.gpu_field(self.v)
        gpu_vsq = runner.gpu_field(self.v_sq)
        gpu_vortsq = runner.gpu_field(self.vort_sq)

        self._vsq_vort_kernel = runner.get_kernel(
            'ComputeSquareVelocityAndVorticity',
            [gpu_map] + gpu_v + [gpu_vsq, gpu_vortsq],
            'PPPPPP')
        self._arr_vsq = runner.backend.get_array(gpu_vsq)
        self._arr_vortsq = runner.backend.get_array(gpu_vortsq)

    def compute_ke_enstropy(self, runner):
        """Computes kinetic energy and estrophy densities on the compute device.

        :rvalue: kinetic energy, enstrophy (per node)
        """
        b = runner.backend
        b.run_kernel(self._vsq_vort_kernel, runner._kernel_grid_full)
        div = 2.0 * runner._spec.num_nodes

        # Compute the sum in double precision.
        kinetic_energy = b.array.sum(self._arr_vsq,
                                     dtype=np.dtype(np.float64)).get() / div
        enstrophy = b.array.sum(self._arr_vortsq,
                                dtype=np.dtype(np.float64)).get() / div

        return kinetic_energy, enstrophy


class ReynoldsStatsMixIn(FlowStatsMixIn):
    """Computes statistics used to characterize turbulent flows:
    - first 4 moments of any quantity (velocity, density)
    - correlations between any 2 qauntities
    """
    aux_code = ['reynolds_statistics.mako']

    #: Number of copies of the stats buffers to keep in GPU memory
    #: between host syncs.
    stat_buf_size = 1024

    def prepare_reynolds_stats(self, runner, moments=True,
                               correlations=True,
                               axis='x'):
        if axis == 'x':
            NX = self.config.lat_nx
            normalizer = runner._spec.ny * runner._spec.nz
        elif axis == 'y':
            NX = self.config.lat_ny
            normalizer = runner._spec.nx * runner._spec.nz
        else:
            NX = self.config.lat_nz
            normalizer = runner._spec.nx * runner._spec.ny

        # Save data used in collect_reynolds_stats().
        self._reyn_points = NX
        self._reyn_moments = moments
        self._reyn_corr = correlations
        self._reyn_normalizer = normalizer

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
        cm_finalize = lat_nx >= cm_bs and axis != 'x'
        self.cm_finalize = cm_finalize
        for m in range(1, 5):
            for f in 'ux', 'uy', 'uz', 'rho':
                _alloc_stat_buf('%s_m%d' % (f, m), self.stat_cm_grid_size, cm_finalize)

        # Buffer for correlations of hydrodynamic variables.
        corr_bs = 128    # block_size, keep in sync with template
        self.stat_corr_grid_size = (NX + 2 + corr_bs - 1) / corr_bs
        corr_finalize = lat_nx >= corr_bs and axis != 'x'
        self.corr_finalize = corr_finalize
        _alloc_stat_buf('ux_uy', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('ux_uz', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('uy_uz', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('ux_rho', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('uy_rho', self.stat_corr_grid_size, corr_finalize)
        _alloc_stat_buf('uz_rho', self.stat_corr_grid_size, corr_finalize)

        gpu_rho = runner.gpu_field(self.rho)
        gpu_v = runner.gpu_field(self.v)
        args = gpu_v + [gpu_rho]

        for field, name in zip(args, ('ux', 'uy', 'uz', 'rho')):
            if cm_finalize:
                setattr(self, 'stat_kern_cm_%s' % name,
                        runner.get_kernel(
                            'ReduceComputeMoments%s64' % axis.upper(), [
                                field,
                                getattr(self, 'gpu_stat_tmp_%s_m1' % name),
                                getattr(self, 'gpu_stat_tmp_%s_m2' % name),
                                getattr(self, 'gpu_stat_tmp_%s_m3' % name),
                                getattr(self, 'gpu_stat_tmp_%s_m4' % name)],
                    'PPPPP', block_size=(cm_bs,), more_shared=True))
                nbs = int(pow(2, math.ceil(math.log(self.stat_cm_grid_size, 2))))
                setattr(self, 'stat_kern_cm_%s_fin' % name,
                        runner.get_kernel(
                    'FinalizeReduceComputeMoments%s64' % axis.upper(), [
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
                            'ReduceComputeMoments%s64' % axis.upper(), [
                                field,
                                getattr(self, 'gpu_stat_%s_m1' % name),
                                getattr(self, 'gpu_stat_%s_m2' % name),
                                getattr(self, 'gpu_stat_%s_m3' % name),
                                getattr(self, 'gpu_stat_%s_m4' % name), 0],
                    'PPPPPi', block_size=(cm_bs,), more_shared=True))

        if corr_finalize:
            self.stat_kern_corr = runner.get_kernel(
                'ReduceComputeCorrelations%s64' % axis.upper(), gpu_v + [
                    gpu_rho, self.gpu_stat_tmp_ux_uy, self.gpu_stat_tmp_ux_uz,
                    self.gpu_stat_tmp_uy_uz, self.gpu_stat_tmp_ux_rho,
                    self.gpu_stat_tmp_uy_rho, self.gpu_stat_tmp_uz_rho],
                'PPPPPPPPPP', block_size=(corr_bs,), more_shared=True)

            nbs = int(pow(2, math.ceil(math.log(self.stat_corr_grid_size, 2))))
            self.stat_kern_corr_fin = runner.get_kernel(
                'FinalizeReduceComputeCorrelations%s64' % axis.upper(),
                [self.gpu_stat_tmp_ux_uy, self.gpu_stat_tmp_ux_uz,
                 self.gpu_stat_tmp_uy_uz, self.gpu_stat_tmp_ux_rho,
                 self.gpu_stat_tmp_uy_rho, self.gpu_stat_tmp_uz_rho,
                 self.gpu_stat_ux_uy, self.gpu_stat_ux_uz,
                 self.gpu_stat_uy_uz, self.gpu_stat_ux_rho,
                 self.gpu_stat_uy_rho, self.gpu_stat_uz_rho, 0],
                'PPPPPPPPPPPPi', block_size=(nbs,), more_shared=True)
        else:
            self.stat_kern_corr = runner.get_kernel(
                'ReduceComputeCorrelations%s64' % axis.upper(), gpu_v + [
                    gpu_rho, self.gpu_stat_ux_uy, self.gpu_stat_ux_uz,
                    self.gpu_stat_uy_uz, self.gpu_stat_ux_rho,
                    self.gpu_stat_uy_rho, self.gpu_stat_uz_rho, 0],
                'PPPPPPPPPPi', block_size=(corr_bs,), more_shared=True)

    stat_cnt = 0
    def collect_reynolds_stats(self, runner):
        """Collects Reynolds statistics."""
        # TODO(mjanusz): Run these kernels in a stream other than main.
        self.stat_cnt += 1

        NX = self._reyn_points
        grid = [self.stat_cm_grid_size]
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
            self.stat_cnt = 0
            # Reset stat buffer offset.
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
            div = self._reyn_normalizer
            return {
                'ux_m1': self.stat_ux_m1 / div,
                'ux_m2': self.stat_ux_m2 / div,
                'ux_m3': self.stat_ux_m3 / div,
                'ux_m4': self.stat_ux_m4 / div,
                'uy_m1': self.stat_uy_m1 / div,
                'uy_m2': self.stat_uy_m2 / div,
                'uy_m3': self.stat_uy_m3 / div,
                'uy_m4': self.stat_uy_m4 / div,
                'uz_m1': self.stat_uz_m1 / div,
                'uz_m2': self.stat_uz_m2 / div,
                'uz_m3': self.stat_uz_m3 / div,
                'uz_m4': self.stat_uz_m4 / div,
                'rho_m1': self.stat_rho_m1 / div,
                'rho_m2': self.stat_rho_m2 / div,
                'rho_m3': self.stat_rho_m3 / div,
                'rho_m4': self.stat_rho_m4 / div,
                'ux_uy': self.stat_ux_uy / div,
                'ux_uz': self.stat_ux_uz / div,
                'uy_uz': self.stat_uy_uz / div,
                'ux_rho': self.stat_ux_rho / div,
                'uy_rho': self.stat_uy_rho / div,
                'uz_rho': self.stat_uz_rho / div}
        else:
            # Update buffer offset.
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


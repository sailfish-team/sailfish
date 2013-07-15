#!/usr/bin/env python

import numpy as np

from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.subdomain import Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim

class KidaSubdomain(Subdomain3D):
    max_v = 0.05

    def boundary_conditions(self, hx, hy, hz):
        pass

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0

        sin = np.sin
        cos = np.cos

        x = (hx + self.config.shift_x) * np.pi * 2.0 / self.gx
        y = (hy + self.config.shift_y) * np.pi * 2.0 / self.gy
        z = (hz + self.config.shift_z) * np.pi * 2.0 / self.gz

        sim.vx[:] = self.max_v * sin(x) * (cos(3 * y) * cos(z) - cos(y) * cos(3 * z))
        sim.vy[:] = self.max_v * sin(y) * (cos(3 * z) * cos(x) - cos(z) * cos(3 * x))
        sim.vz[:] = self.max_v * sin(z) * (cos(3 * x) * cos(y) - cos(x) * cos(3 * y))


class KidaSim(LBFluidSim):
    subdomain = KidaSubdomain
    aux_code = ['data_processing.mako']

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'periodic_x': True,
            'periodic_y': True,
            'periodic_z': True,
            'lat_nx': 110,
            'lat_ny': 110,
            'lat_nz': 110,
            'grid': 'D3Q15',
            'visc': 0.001375,
            'access_pattern': 'AA',
            'minimize_roundoff': True,
            'perf_stats_every': 200,
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)

        # Parameters used to verify phase shift invariance.
        group.add_argument('--shift_x', type=int, default=0)
        group.add_argument('--shift_y', type=int, default=0)
        group.add_argument('--shift_z', type=int, default=0)

    @classmethod
    def modify_config(cls, config):
        print 'Re = {0}'.format(config.lat_nx *
                cls.subdomain.max_v / config.visc)

    axis = 0
    axis_pos = 0
    def before_main_loop(self, runner):
        runner._subdomain._buf = np.zeros((self.config.lat_nx, self.config.lat_nx), dtype=np.float32)
        self._gpu_buf = runner.backend.alloc_buf(like=runner._subdomain._buf)

        gpu_v = runner.gpu_field(self.v)
        self.extract_k = runner.get_kernel('ExtractSlice',
                                           [self.axis, self.axis_pos] + gpu_v +
                                           [self._gpu_buf],
                                           'iiPPPP', block_size=(128,))

    def after_step(self, runner):
        every = 5
        mod = self.iteration % every

        if mod == every - 1:
            self.need_fields_flag = True
        elif mod == 0:
            runner.backend.run_kernel(self.extract_k, ((self.config.lat_nx + 127)
                                                        / 128, self.config.lat_ny))

if __name__ == '__main__':
    ctrl = LBSimulationController(KidaSim, EqualSubdomainsGeometry3D)
    ctrl.run()

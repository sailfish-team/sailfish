#!/usr/bin/python

import numpy as np
from sailfish.controller import LBSimulationController
from examples.ldc_2d import LDCGeometry, LDCBlock, LDCSim

class LDCSimUnorm(LDCSim):
    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 128,
            'lat_ny': 128,
            'max_iters': 30000,
            'every': 250,
            'visc': 0.16011,
            'model':'mrt',
            })

    u_norm_table = []
    def after_step(self, runner):
        every_n = 523

        # Request the velocity field one step before actual processing.
        if self.iteration % (every_n - 1) == 0:
            self.need_sync_flag = True

        # Calculate and save the norm of valocity field.
        if self.iteration == every_n:
            self.u_old = np.sqrt(runner._sim.vx**2 + runner._sim.vy**2)

        if self.iteration % every_n == 0 and self.iteration > every_n:
            u = np.sqrt(runner._sim.vx**2 + runner._sim.vy**2)
            du_norm = np.linalg.norm(u - self.u_old) / u.size
            u_norm = np.linalg.norm(u) / u.size
            self.u_old = u
            self.u_norm_table.append((self.iteration, du_norm,u_norm))

        if self.iteration == self.config.max_iters - 1:
            u_norm_table_np = np.array(self.u_norm_table)
            np.savez('unorm', it=u_norm_table_np[:, 0], du_norm=u_norm_table_np[:, 1], u_norm=u_norm_table_np[:, 2])


if __name__ == '__main__':
    LDCBlock.max_v = 0.05
    ctrl = LBSimulationController(LDCSimUnorm, LDCGeometry)
    ctrl.run()

    # simple pylab code displaying time evolution of the norm ||u-u0||
    from pylab import *
    figure(1)
    clf()
    data = np.load('unorm.npz')
    semilogy(data['it'], data['du_norm'], 'ro-')
    grid(True)
    savefig("unorm.png")

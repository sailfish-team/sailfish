import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish import io

import common

from basilar import BasilarSubdomain, BasilarSim


class ProfileBasilarSim(BasilarSim):
    lb_v = 0.05

    @classmethod
    def update_defaults(cls, defaults):
        super(ProfileBasilarSim, cls).update_defaults(defaults)
        defaults.update({
            'velocity': 'external',
            'velocity_profile': 'profiles/basilar_valencia_profile_uniform_lin_int.dat',
            'reynolds': 100,
        })

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--midslice_every', type=int, default=0,
                help='how often to save data from the middle of the '
                'simulation domain')

    def _get_midslice(self, runner):
        wall_nonghost = self.config._wall_map[1:-1,1:-1,1:-1]
        size_y = wall_nonghost.shape[1]
        m = size_y / 2
        self.midslice = [slice(None), slice(m - 3, m + 3), slice(None)]
        return wall_nonghost

    midslice = None
    mask = None
    def after_step(self, runner):
        super(ProfileBasilarSim, self).after_step(runner)

        if self.config.midslice_every == 0:
            return

        mod = self.iteration % self.config.midslice_every
        if mod == self.config.midslice_every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            if self.midslice is None:
                wall_nonghost = self._get_midslice(runner)
                hx, hy, hz = runner._subdomain._get_mgrid()
                hx = hx[self.midslice]
                hy = hy[self.midslice]
                hz = hz[self.midslice]
                self.mask = wall_nonghost[hz, hy, hx]

            ms = self.midslice
            rho = self.rho[ms]
            vx = self.vx[ms]
            vy = self.vy[ms]
            vz = self.vz[ms]
            rho[self.mask] = np.nan
            vx[self.mask] = np.nan
            vy[self.mask] = np.nan
            vz[self.mask] = np.nan

            np.savez(io.filename(self.config.base_name +
                '_midslice', 7, 0, self.iteration),
                rho=rho, vx=vx, vy=vy, vz=vz)

if __name__ == '__main__':
    LBSimulationController(ProfileBasilarSim, EqualSubdomainsGeometry3D).run()

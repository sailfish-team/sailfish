import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish import io

import common

from basilar_profile import ProfileBasilarSim


class WSSBasilarSim(ProfileBasilarSim, common.StressMixIn):
    def after_step(self, runner):
        super(WSSBasilarSim, self).after_step(runner)

        if not self.need_output():
            return
        self.get_stress(runner)

        # iter + 1 here to match the way standard output is saved.
        np.savez_compressed(io.filename(self.config.base_name +
                 '_stress', 7, 0, self.iteration + 1),
                 xx=self.stress_xx,
                 xy=self.stress_xy,
                 xz=self.stress_xz,
                 yy=self.stress_yy,
                 yz=self.stress_yz,
                 zz=self.stress_zz)


if __name__ == '__main__':
    LBSimulationController(WSSBasilarSim, EqualSubdomainsGeometry3D).run()

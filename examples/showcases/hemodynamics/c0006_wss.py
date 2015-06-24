import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish import io

import common

from c0006_profile import ProfileC0006Sim


class WSSC0006Sim(ProfileC0006Sim, common.StressMixIn):
    def after_step(self, runner):
        super(WSSC0006Sim, self).after_step(runner)

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
    LBSimulationController(WSSC0006Sim, EqualSubdomainsGeometry3D).run()

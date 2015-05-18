from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D

import ushape_base

class UshapeSim(ushape_base.UshapeBaseSim):
    lb_v = 0.17

    @classmethod
    def update_defaults(cls, defaults):
        super(UshapeSim, cls).update_defaults(defaults)
        defaults.update({
            'velocity': 'constant',
            # For benchmarking, disable error checking.
        })

if __name__ == '__main__':
    LBSimulationController(UshapeSim, EqualSubdomainsGeometry3D).run()

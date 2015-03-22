import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D

import common

from c0006 import C0006Subdomain, C0006Sim


class OscillatoryC0006Sim(C0006Sim):
    lb_v = 0.001

    @classmethod
    def update_defaults(cls, defaults):
        super(OscillatoryC0006Sim, cls).update_defaults(defaults)
        defaults.update({
            'velocity': 'oscillatory',
            'model': 'mrt',
            'reynolds': 100,
        })


if __name__ == '__main__':
    LBSimulationController(OsciallatoryC0006Sim, EqualSubdomainsGeometry3D).run()

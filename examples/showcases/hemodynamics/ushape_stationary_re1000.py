from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D

import ushape_base

class UshapeSim(ushape_base.UshapeBaseSim, ushape_base.DynamicViscosity):
    lb_v = 0.05

    @classmethod
    def update_defaults(cls, defaults):
        super(UshapeSim, cls).update_defaults(defaults)
        defaults.update({
            'velocity': 'constant',
            'reynolds': 1000,
            # For benchmarking, disable error checking.
        })

    @classmethod
    def modify_config(cls, config):
        ushape_base.UshapeBaseSim.modify_config(config)
        # At 10k iterations, lower viscosity to reach Re = 1k. This is
        # necessary for the simulation to remain stable. Starting at Re = 1k
        # does not work since the initial conditions are very far from
        # equilibrium.
        cls.viscosity_map = {10000: config.visc}
        config.visc *= 10  # start with Re=100

if __name__ == '__main__':
    LBSimulationController(UshapeSim, EqualSubdomainsGeometry3D).run()

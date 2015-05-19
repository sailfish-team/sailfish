import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D

import ushape_base

class OscillatoryUshapeSubdomain(ushape_base.UshapeSubdomain):
    oscillatory_delay = 0 #100000

    def initial_conditions(self, sim, hx, hy, hz):
        # Oscillatory delay also works, but convergence is slower. At 1M
        # iterations, distortions from the original pressure wave are still
        # visible.
        a = np.load('results/ushape/ushape_re100_inc_500_mv0.17/data.0.75000.npz')
        v = a['v']
        v[np.isnan(v)] = 0.0
        rho = a['rho']
        rho[np.isnan(rho)] = 1.0

        old_v_max = 0.17

        sim.vx[:] = v[0,:,:,:] / old_v_max * UshapeSim.lb_v
        sim.vy[:] = v[1,:,:,:] / old_v_max * UshapeSim.lb_v
        sim.vz[:] = v[2,:,:,:] / old_v_max * UshapeSim.lb_v
        sim.rho[:] = (rho - 1.0) * (UshapeSim.lb_v / old_v_max)**2  + 1.0

        assert np.all(np.isfinite(sim.rho[:]))

class UshapeSim(ushape_base.UshapeBaseSim):
    lb_v = 0.00025

    subdomain = OscillatoryUshapeSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        super(UshapeSim, cls).update_defaults(defaults)
        defaults.update({
            'velocity': 'oscillatory',
            #'model': 'mrt',
            'reynolds': 100,
            # Necessary since density variance in the simulation is of the order
            # of machine epsilon in single precision.
            'minimize_roundoff': True

            # For benchmarking, disable error checking.
        })

if __name__ == '__main__':
    LBSimulationController(UshapeSim, EqualSubdomainsGeometry3D).run()

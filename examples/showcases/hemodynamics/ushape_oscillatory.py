import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D

import ushape_base

class OscillatoryUshapeSubdomain(ushape_base.UshapeSubdomain):
    oscillatory_delay = 0 #100000

    def initial_conditions(self, sim, hx, hy, hz):
        a = np.load('results/ushape/old/ushape_re100_inc_500.0.150000.npz')
        v = a['v']
        v[np.isnan(v)] = 0.0
        rho = a['rho']
        rho[np.isnan(rho)] = 1.0

        sim.vx[:] = v[0,:,:,:] / 0.05 * UshapeSim.lb_v
        sim.vy[:] = v[1,:,:,:] / 0.05 * UshapeSim.lb_v
        sim.vz[:] = v[2,:,:,:] / 0.05 * UshapeSim.lb_v
        sim.rho[:] = (rho - 1.0) * (UshapeSim.lb_v / 0.05)**2  + 1.0


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

            # For benchmarking, disable error checking.
        })

if __name__ == '__main__':
    LBSimulationController(UshapeSim, EqualSubdomainsGeometry3D).run()

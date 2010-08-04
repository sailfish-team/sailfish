#!/usr/bin/python

import numpy as np
from sailfish import geo, lbm

class TestGeo2D(geo.LBMGeo2D):
    def init_fields(self):
        self.sim.rho[:] = 1.0
        if isinstance(self.sim, lbm.ShanChenBinary):
            self.sim.phi[:] = 1.0
            self.sim.rho[:] += np.random.rand(*self.sim.rho.shape) / 1000
            self.sim.phi[:] += np.random.rand(*self.sim.phi.shape) / 1000
        else:
            self.sim.phi[:] = np.random.rand(*self.sim.phi.shape) / 100.0

class TestGeo3D(geo.LBMGeo3D):
    def init_fields(self):
        self.sim.rho[:] = 1.0

        if isinstance(self.sim, lbm.ShanChenBinary):
            self.sim.phi[:] = 1.0
            self.sim.rho[:] += np.random.rand(*self.sim.rho.shape) / 1000
            self.sim.phi[:] += np.random.rand(*self.sim.phi.shape) / 1000
        else:
            self.sim.phi[:] = np.random.rand(*self.sim.phi.shape) / 100.0

class SCTestSim(lbm.ShanChenBinary):
    def __init__(self, geo_class, defaults={}):
        settings = {'visc': 0.166666666666, 'periodic_x': True, 'periodic_y':
                True, 'periodic_z': True, 'tau_phi': 1.0, 'G': -1.2}
        settings.update(defaults)
        super(SCTestSim, self).__init__(geo_class, defaults=settings)

class FETestSim(lbm.BinaryFluidFreeEnergy):
    def __init__(self, geo_class, defaults={}):
        settings = {'visc': 0.166666666666, 'periodic_x': True, 'periodic_y':
                True, 'periodic_z': True, 'tau_phi': 1.0,
                'kappa': 2e-4, 'Gamma': 25.0, 'A': 1e-4,
                'tau_a': 4.5, 'tau_b': 0.8}
        settings.update(defaults)
        super(FETestSim, self).__init__(geo_class, defaults=settings)

if __name__ == '__main__':
    SCTestSim(TestGeo2D).run()

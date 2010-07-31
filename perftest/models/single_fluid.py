#!/usr/bin/python

import numpy as np
from sailfish import geo, lbm

class TestGeo2D(geo.LBMGeo2D):
    def init_fields(self):
        self.sim.rho[:] = 1.0

class TestGeo3D(geo.LBMGeo3D):
    def init_fields(self):
        self.sim.rho[:] = 1.0

class TestSim(lbm.FluidLBMSim):
    def __init__(self, geo_class, defaults={}):
        settings = {'visc': 0.166666666666, 'periodic_x': True, 'periodic_y': True, 'periodic_z': True}
        settings.update(defaults)
        super(TestSim, self).__init__(geo_class, defaults=settings)


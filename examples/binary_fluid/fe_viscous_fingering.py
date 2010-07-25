#!/usr/bin/python

import numpy as np
from sailfish import geo, lbm

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

class GeoFEFinger(geo.LBMGeo3D):

    def define_nodes(self):
        hz, hy, hx = np.mgrid[0:self.lat_nz, 0:self.lat_ny, 0:self.lat_nx]
        self.set_geo(np.logical_or(hz == 0, hz == self.lat_nz-1), self.NODE_WALL)

    def init_dist(self, dist):
        hz, hy, hx = np.mgrid[0:self.lat_nz, 0:self.lat_ny, 0:self.lat_nx]

        a = 100.0 - 8.0 * np.cos(2.0 * np.pi * hy / self.lat_ny)
        b = 200.0 - 8.0 * np.cos(2.0 * np.pi * hy / self.lat_ny)

        self.sim.rho[:] = 1.0
        self.sim.phi[:] = 1.0
        self.sim.phi[np.logical_or(hx <= a, hx >= b)] = -1.0

        self.sim.ic_fields = True

class FEFingerSim(lbm.BinaryFluidFreeEnergy):

    filename = 'fe_fingering'

    def __init__(self, geo_class, defaults={}):

        settings = {'bc_velocity': 'equilibrium', 'verbose': True, 'lat_nx': 640,
                    'lat_ny': 101, 'lat_nz': 37, 'grid': 'D3Q19',
                    'tau_a': 4.5, 'tau_b': 0.6, 'tau_phi': 1.0, 'kappa': 9.18e-5,
                    'Gamma': 25.0, 'A': 1.41e-4, 'lambda_': 0.0, 'model': 'femrt',
                    'periodic_x': True, 'periodic_y': True, 'scr_scale': 1,
                    'periodic_z': True}
        settings.update(defaults)

        lbm.BinaryFluidFreeEnergy.__init__(self, geo_class, options=[], defaults=settings)

        self.add_body_force((3.0e-5, 0.0, 0.0), grid=0, accel=False)

        # Use the fluid velocity in the relaxation of the order parameter field,
        # and the molecular velocity in the relaxation of the density field.
        self.use_force_for_eq(None, 0)
        self.use_force_for_eq(0, 1)

if __name__ == '__main__':
    sim = FEFingerSim(GeoFEFinger)
    sim.run()

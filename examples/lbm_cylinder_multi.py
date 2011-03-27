#!/usr/bin/python -u

import numpy as np
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D, GeoBlock2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim

class CylinderGeometry(LBGeometry2D):
    def blocks(self, n=None):
        blocks = []
        q = self.gx / 2
        blocks.append(LBBlock2D((0, 0), (q, self.gy)))
        blocks.append(LBBlock2D((q, 0), (q, self.gy)))
        return blocks


class CylinderBlock(GeoBlock2D):
    def _define_nodes(self, hx, hy):
        diam = self.gy / 3
        x0 = 2 * diam
        y0 = self.gy / 2

        self.set_geo(hy == 0, self.NODE_WALL)
        self.set_geo(hy == self.gy-1, self.NODE_WALL)

#        cylinder_map = np.square(hx - x0) + np.square(hy - y0) < diam**2 / 4.0
#        self.set_geo(cylinder_map, self.NODE_WALL)

    def _init_fields(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vy[:] = 0.0
        sim.vx[:] = 0.0


class CylinderSimulation(LBFluidSim, LBForcedSim):
    geo = CylinderBlock

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 320,
            'lat_ny': 128,
            'visc': 0.1,
            'periodic_x': True})


    def __init__(self, config):
        super(CylinderSimulation, self).__init__(config)
        self.add_body_force((1e-5, 0.0))

if __name__ == '__main__':
    ctrl = LBSimulationController(CylinderSimulation, CylinderGeometry)
    ctrl.run()

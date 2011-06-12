import unittest

from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D, GeoBlock2D
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim, LBForcedSim

#
#  b1  b3
#
#  b0  b2
#

class GeometryTest(LBGeometry2D):
    def blocks(self, n=None):
        blocks = []
        q = 128
        for i in range(0, 2):
            for j in range(0, 2):
                blocks.append(LBBlock2D((i * q, j * q), (q, q)))

        return blocks

class BlockTest(GeoBlock2D):
    def _define_nodes(self, hx, hy):
        pass

    def _init_fields(self, sim, hx, hy):
        pass

class SimulationTest(LBFluidSim, LBForcedSim):
    geo = BlockTest

    @classmethod
    def modify_config(cls, config):
        config.relaxation_enabled = False

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 256,
            'lat_ny': 256,
            'max_iters': 3,
            'every': 1,
            'quiet': True,
            'output': '/tmp/foo',
            'debug_dump_dists': True,
            })

    def initial_conditions(self, runner):
        dbuf = runner._debug_get_dist()
        dbuf[:] = 0.0

        if runner._block.id == 0:
            dist_idx = self.grid.vec_idx([1, 1])
            dbuf[dist_idx,128,128] = 0.11
        elif runner._block.id == 1:
            dist_idx = self.grid.vec_idx([1, -1])
            dbuf[dist_idx,128,1] = 0.22
        elif runner._block.id == 2:
            dist_idx = self.grid.vec_idx([-1, 1])
            dbuf[dist_idx,1,128] = 0.33
        else:
            dist_idx = self.grid.vec_idx([-1, -1])
            dbuf[dist_idx,1,1] = 0.44

        runner._debug_set_dist(dbuf)
        runner._debug_set_dist(dbuf, False)


ctrl = LBSimulationController(SimulationTest, GeometryTest)
ctrl.run()

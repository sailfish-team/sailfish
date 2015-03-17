# :,1,: is input
# -2,:,: is output
# inlet at  c = (0.0507043 0.0279591 0.052244), r = 0.0019868 (0.94%)
#  U_avg = 0.15m/s U_max = 0.3m/s

# process geometry with 011000

import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.vis_mixin import Vis2DSliceMixIn
from sailfish import io
from sailfish.sym import S, D3Q19

import common

class C0006Subdomain(common.InflowOutflowSubdomain):
    _flow_orient = D3Q19.vec_to_dir([0, 1, 0])
    _outlet_orient = D3Q19.vec_to_dir([-1, 0, 0])
    inflow_loc = [0.0507043, 0.0279591, 0.052244]
    inflow_rad = 0.0019868

    def _inflow_outflow(self, hx, hy, hz, wall_map):
        inlet = None
        outlet = None

        if np.min(hy) <= 0:
            inlet = np.logical_not(wall_map) & (hy == 0)

        if np.max(hx) >= self.gx - 1:
            outlet = np.logical_not(wall_map) & (hx == self.gx - 1)

        return inlet, outlet


class C0006Sim(common.HemoSim, Vis2DSliceMixIn):
    subdomain = C0006Subdomain
    phys_diam = 0.0019868 * 2
    lb_v = 0.05

    @classmethod
    def update_defaults(cls, defaults):
        super(C0006Sim, cls).update_defaults(defaults)
        defaults.update({
            'max_iters': 500000,
            'checkpoint_every': 200000,
            'checkpoint_from': 200000,
            'every': 50000,
            'from_': 0,
            'model': 'bgk',

            'subdomains': 1,
            'node_addressing': 'direct',
            'reynolds': 100,
            'velocity': 'constant',
            'geometry': 'geo/proc_c0006_500.npy.gz',
        })

    @classmethod
    def set_walls(cls, config):
        if not config.geometry:
            return

        wall_map = cls.load_geometry(config.geometry)

        # TODO: Flip the domain so that the longest dimension is X.
        size = wall_map.shape[1]
        if not config.base_name:
            config.base_name = 'results/re%d_c0006_%d_%s' % (
                config.reynolds, size, config.velocity)

        # Override lattice size based on the geometry file.
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape

        # Add ghost nodes.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map

    @classmethod
    def get_diam(cls, config):
        return 2.0 * np.sqrt(np.sum(np.logical_not(config._wall_map[:,1,:])) /
                             np.pi)

    @classmethod
    def modify_config(cls, config):
        cls.set_walls(config)
        super(C0006Sim, cls).modify_config(config)


if __name__ == '__main__':
    LBSimulationController(C0006Sim, EqualSubdomainsGeometry3D).run()

# input: :,1,:
# output: 2,:,: (need to get rid of one node layer)
#         -2,:,:
# c = (0.0369357 0.028672 0.0126818), r = 0.00168079 (0.06%)
#   U_avg = 0.1m/s U_max = 0.2m/s
#
# process geometry with 221000

import numpy as np
from scipy import ndimage

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.vis_mixin import Vis2DSliceMixIn
from sailfish import io
from sailfish.sym import S, D3Q19

import common

class BasilarSubdomain(common.InflowOutflowSubdomain):
    _flow_orient = D3Q19.vec_to_dir([0, 1, 0])
    _outlet_orient = (D3Q19.vec_to_dir([-1, 0, 0]),
                      D3Q19.vec_to_dir([-1, 0, 0]),
                      D3Q19.vec_to_dir([ 1, 0, 0]),
                      D3Q19.vec_to_dir([ 1, 0, 0]))
    inflow_loc = [0.0369357, 0.028672, 0.0126818]
    inflow_rad = 0.00168079

    def _inflow_outflow(self, hx, hy, hz, wall_map):
        inlet = None
        outlet = []

        if np.min(hy) <= 0:
            inlet = np.logical_not(wall_map) & (hy == 0)

        if np.max(hx) >= self.gx - 1:
            outlets = np.logical_not(wall_map) & (hx == self.gx - 1)
            label, nb_labels = ndimage.label(outlets)
            assert nb_labels == 2, 'Expected two outflows at high x.'
            outlet.extend([label == 1, label == 2])
        else:
            outlet.extend([None, None])

        if np.min(hx) <= 0:
            outlets = np.logical_not(wall_map) & (hx == 0)
            label, nb_labels = ndimage.label(outlets)
            assert nb_labels == 2, 'Expected two outflows at low x.'
            outlet.extend([label == 1, label == 2])
        else:
            oulet.extend([None, None])

        return inlet, outlet


class BasilarSim(common.HemoSim, Vis2DSliceMixIn):
    subdomain = BasilarSubdomain
    phys_diam = BasilarSubdomain.inflow_rad * 2
    lb_v = 0.05

    @classmethod
    def update_defaults(cls, defaults):
        super(BasilarSim, cls).update_defaults(defaults)
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
            'geometry': 'geo/proc_basilar_500.npy.gz',
        })

    @classmethod
    def get_diam(cls, config):
        return 2.0 * np.sqrt(np.sum(np.logical_not(config._wall_map[:,1,:])) /
                             np.pi)

    @classmethod
    def modify_config(cls, config):
        super(BasilarSim, cls).modify_config(config)

        size = config._wall_map.shape[2]
        if not config.base_name:
            config.base_name = 'results/re%d_basilar_%d_%s' % (
                config.reynolds, size, config.velocity)


if __name__ == '__main__':
    LBSimulationController(BasilarSim, EqualSubdomainsGeometry3D).run()

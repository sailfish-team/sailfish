"""pygame visualization backend for 3d simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import numpy as np
from sailfish import vis_2d

class Fluid3DVisCutplane(vis_2d.Fluid2DVis):
    name = '3dcutplane'
    dims = [3]

    def __init__(self, config, blocks, quit_event, sim_quit_event, vis_config):
        super(Fluid3DVisCutplane, self).__init__(config, blocks,
                                                 quit_event, sim_quit_event,
                                                 vis_config)

    # Since we inherit from Fluid2DVis, we need to make sure arguments are not
    # registered twice.
    @classmethod
    def add_options(cls, group):
        pass

    @property
    def size(self):
        return self._blocks[self._vis_config.block].size[:-1]

    def size3d(self):
        return list(reversed(self._blocks[self._vis_config.block].size))

    def _get_geo_map(self, width, height, block):
        size = self.size3d()
        t = np.zeros(size, dtype=np.uint8)
        t.ravel()[:] = block.vis_geo_buffer[:]
        geo_map = np.zeros((height, width), dtype=np.uint8)
        geo_map[:] = t[5,:,:]
        return geo_map

    def _get_field(self, width, height, block):
        # FIXME(michalj): This is horribly inefficient.  We should only recreate
        # the array if the data has been updated.

        size = self.size3d()
        t = np.zeros(size, dtype=np.float32)
        t.ravel()[:] = block.vis_buffer[:]

        tmp = np.zeros((height, width), dtype=np.float32)
        tmp[:] = t[5,:,:]
        return tmp

engine=Fluid3DVisCutplane

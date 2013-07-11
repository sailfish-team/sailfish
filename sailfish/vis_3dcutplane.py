"""pygame visualization backend for 3d simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import numpy as np
import pygame

from sailfish import vis_2d

class Fluid3DVisCutplane(vis_2d.Fluid2DVis):
    name = '3dcutplane'
    dims = [3]

    def __init__(self, config, subdomains, quit_event, sim_quit_event,
                 vis_config, geo_queues):
        self._slice_position = [0, 0, 0]
        self._slice_axis = 0
        super(Fluid3DVisCutplane, self).__init__(config, subdomains,
                                                 quit_event, sim_quit_event,
                                                 vis_config, geo_queues)

    # Since we inherit from Fluid2DVis, we need to make sure arguments are not
    # registered twice.
    @classmethod
    def add_options(cls, group):
        pass

    @property
    def size(self):
        size = list(self._subdomains[self._vis_config.subdomain].size)
        del size[self._slice_axis]
        return size

    @property
    def selector(self):
        sel = [slice(None), slice(None), slice(None)]
        sel[(2 - self._slice_axis)] = self._slice_position[self._slice_axis]
        return sel

    def size3d(self):
        return list(reversed(self._subdomains[self._vis_config.subdomain].size))

    def _get_geo_map(self, width, height, subdomain):
        size = self.size3d()
        t = np.zeros(size, dtype=np.uint8)
        t.ravel()[:] = subdomain.vis_geo_buffer[:]
        geo_map = np.zeros((height, width), dtype=np.uint8)
        geo_map[:] = t[self.selector]
        return geo_map

    def _get_field(self, width, height, subdomain):
        # FIXME(michalj): This is horribly inefficient.  We should only recreate
        # the array if the data has been updated.
        size = self.size3d()
        t = np.zeros(size, dtype=np.float32)
        t.ravel()[:] = subdomain.vis_buffer[:]

        tmp = np.zeros((height, width), dtype=np.float32)
        tmp[:] = t[self.selector]
        return tmp

    def _visualize(self):
        ret = super(Fluid3DVisCutplane, self)._visualize()
        axes = ['x', 'y', 'z']
        ret.append('slice @ {0} = {1}'.format(axes[self._slice_axis],
                                              self._slice_position[self._slice_axis]))
        return ret

    # Drawing new nodes is not supported in 3D.
    def _draw_wall(self, unused):
        pass

    def _process_misc_event(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_x:
                self._slice_axis = 0
                self.resize()
            elif event.key == pygame.K_y:
                self._slice_axis = 1
                self.resize()
            elif event.key == pygame.K_z:
                self._slice_axis = 2
                self.resize()
            # Move the slice along the selected axis.
            elif event.key == pygame.K_QUOTE:
                if (self._slice_position[self._slice_axis] <
                    self.size3d()[(2 - self._slice_axis)] - 1):
                    self._slice_position[self._slice_axis] += 1
            elif event.key == pygame.K_SEMICOLON:
                if self._slice_position[self._slice_axis] > 0:
                    self._slice_position[self._slice_axis] -= 1

engine=Fluid3DVisCutplane

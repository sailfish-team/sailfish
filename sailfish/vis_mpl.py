"""matplotlib visualization backend."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import math
import os
import sys
import time

import numpy as np

from sailfish import lb_base, vis, geo_block, util


class Fluid2DVis(vis.FluidVis):
    name = 'matplotlib'
    dims = [2]

    def __init__(self, config, blocks, quit_event, sim_quit_event, vis_config):
        super(Fluid2DVis, self).__init__()
        self.config = config
        self.config.logger.info("Initializating matplotlib 2D vis. engine.")
        self._quit_event = quit_event
        self._sim_quit_event = sim_quit_event
        self._vis_config = vis_config
        self._blocks = blocks

        self.background = None

    @property
    def size(self):
        return self._blocks[self._vis_config.block].size

    def update(self, event):
        if self._quit_event.is_set():
            sys.exit()

        curr_iter = self._vis_config.iteration
        if curr_iter < 0:
            return

        # XXX
        self._vis_config.field = 0
        width, height = self.size
        block = self._blocks[self._vis_config.block]
        buffer = np.zeros((height, width))
        buffer.ravel()[:] = block.vis_buffer[:]

        self.ax.set_title('Iteration: {0}'.format(curr_iter))
        self.plot.set_data(buffer)
        self.plot.set_clim(np.min(buffer), np.max(buffer))
        self.fig.canvas.draw_idle()

    def run(self):
        import matplotlib
        matplotlib.use('WXAgg')
        matplotlib.rcParams['toolbar'] = 'None'
        import matplotlib.pyplot as plt

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        width, height = self.size
        buffer = np.zeros((height, width))
        self.plot = self.ax.imshow(buffer)
        self.fig.colorbar(self.plot)
        import wx
        wx.EVT_IDLE(wx.GetApp(), self.update)
        plt.show()
        self._sim_quit_event.set()

engine=Fluid2DVis

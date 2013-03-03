"""matplotlib visualization backend."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import sys

import numpy as np

from sailfish import vis


class Fluid2DVis(vis.FluidVis):
    name = 'matplotlib'
    dims = [2]

    def __init__(self, config, subdomains, quit_event, sim_quit_event, vis_config):
        super(Fluid2DVis, self).__init__()
        self.config = config
        self.config.logger.info("Initializating matplotlib 2D vis. engine.")
        self._quit_event = quit_event
        self._sim_quit_event = sim_quit_event
        self._vis_config = vis_config
        self._subdomains = subdomains

        self.background = None

    @property
    def size(self):
        return self._subdomains[self._vis_config.subdomain].size

    def update(self, event):
        if self._quit_event.is_set():
            sys.exit()

        curr_iter = self._vis_config.iteration
        if curr_iter < 0:
            return

        # XXX
        self._vis_config.field = 1
        width, height = self.size
        subdomain = self._subdomains[self._vis_config.subdomain]
        buffer = np.zeros((height, width))
        buffer.ravel()[:] = subdomain.vis_buffer[:]

        self.ax.set_title('Iteration: {0}'.format(curr_iter))
        self.plot.set_data(buffer)
        self.plot.set_clim(np.nanmin(buffer), np.nanmax(buffer))
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
        self.plot = self.ax.imshow(buffer, origin='lower')
        self.fig.colorbar(self.plot)
        import wx
        wx.EVT_IDLE(wx.GetApp(), self.update)
        plt.show()
        self._sim_quit_event.set()

engine=Fluid2DVis

import math
import numpy
import pygame
import os
import sys
import time

from sailfish import geo
from sailfish import sym
from sailfish import vis

from enthought.mayavi import mlab

class Fluid3DVis(vis.FluidVis):

    def __init__(self, sim):
        super(Fluid3DVis, self).__init__()
        self._tracers = True
        self.sim = sim

    def visualize(self):
        self.sim.sim_step(self._iter, self._tracers)

        if self._iter % self.sim.options.every == 0:
            self.density.mlab_source.set(scalars=self.sim.rho.transpose())
            self.vx.mlab_source.set(scalars=(self.sim.vy**2 + self.sim.vx**2 + self.sim.vz**2).transpose())
            self.velocity.mlab_source.set(u=self.sim.vx.transpose(), v=self.sim.vy.transpose(), w=self.sim.vz.transpose())
            self.trc.mlab_source.set(x=self.sim.tracer_x, y=self.sim.tracer_y, z=self.sim.tracer_z)

        self._iter += 1

    def main(self):
        self._iter = 1
        self.density = mlab.pipeline.scalar_field(self.sim.rho.transpose())
        self.vx = mlab.pipeline.scalar_field((self.sim.vx**2 + self.sim.vy**2 + self.sim.vz**2).transpose())
        self.velocity = mlab.pipeline.vector_field(self.sim.vx.transpose(), self.sim.vy.transpose(), self.sim.vz.transpose())
        self.trc = mlab.points3d(self.sim.tracer_x, self.sim.tracer_y, self.sim.tracer_z, scale_factor=0.75)

        mlab.pipeline.image_plane_widget(self.vx, plane_orientation='x_axes', slice_index=10)
        mlab.pipeline.image_plane_widget(self.vx, plane_orientation='y_axes', slice_index=10)
        mlab.pipeline.image_plane_widget(self.vx, plane_orientation='z_axes', slice_index=10)
        mlab.axes()

#       mlab.pipeline.vector_cut_plane(self.velocity, mask_points=2, scale_factor=3, plane_orientation='y_axes')
#       mlab.pipeline.vectors(self.velocity, mask_points=20, scale_factor=3.)
        mlab.outline()

        while 1:
            self.visualize()


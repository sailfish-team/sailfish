import math
import numpy
import pygame
import geo2d
import os
import sys
import time
import sym

from enthought.mayavi import mlab

class Fluid3DVis(object):

	def __init__(self):
		self._tracers = True

	def visualize(self):
		self.sim.sim_step(self._iter, self._tracers)

		if self._iter % self.sim.options.every == 0:
			self.density.mlab_source.set(scalars=self.sim.rho.transpose())
			self.vx.mlab_source.set(scalars=(self.sim.vy**2 + self.sim.vx**2 + self.sim.vz**2).transpose())
			self.velocity.mlab_source.set(u=self.sim.vx.transpose(), v=self.sim.vy.transpose(), w=self.sim.vz.transpose())
			self.trc.mlab_source.set(x=self.sim.tracer_x, y=self.sim.tracer_y, z=self.sim.tracer_z)

		self._iter += 1

	def main(self, sim):
		self.sim = sim
		self._iter = 1
		self.density = mlab.pipeline.scalar_field(sim.rho.transpose())
		self.vx = mlab.pipeline.scalar_field((sim.vx**2 + sim.vy**2 + sim.vz**2).transpose())
		self.velocity = mlab.pipeline.vector_field(sim.vx.transpose(), sim.vy.transpose(), sim.vz.transpose())
		self.trc = mlab.points3d(sim.tracer_x, sim.tracer_y, sim.tracer_z, scale_factor=0.75)

		mlab.pipeline.image_plane_widget(self.vx, plane_orientation='x_axes', slice_index=10)
		mlab.pipeline.image_plane_widget(self.vx, plane_orientation='y_axes', slice_index=10)
		mlab.axes()

#		mlab.pipeline.vector_cut_plane(self.velocity, mask_points=2, scale_factor=3, plane_orientation='y_axes')
#		mlab.pipeline.vectors(self.velocity, mask_points=20, scale_factor=3.)
		mlab.outline()

		while 1:
			self.visualize()


#!/usr/bin/python

import pycuda.autoinit
import pycuda.driver as cuda
import math
import numpy
import sys
import time

import vis2d

from sim import *

from optparse import OptionParser, OptionValueError

class LBMGeo(object):
	"""Abstract class for the LBM geometry."""

	def __init__(self, lat_w, lat_h, model):
		self.lat_w = lat_w
		self.lat_h = lat_h
		self.model = model
		self.map = numpy.zeros((lat_h, lat_w), numpy.int32)
		self.gpu_map = cuda.mem_alloc(self.map.size * self.map.dtype.itemsize)
		self.reset()

	def update_map(self):
		cuda.memcpy_htod(self.gpu_map, self.map)

	def reset(self): abstract

	def init_dist(self, dist): abstract

	def velocity_to_dist(self, vx, vy, dist, x, y):
		"""Set the distributions at node (x,y) so that the fluid there has a specific velocity (vx,vy)."""
		cusq = -1.5 * (vx*vx + vy*vy)
		dist[0][y][x] = numpy.float32((1.0 + cusq) * 4.0/9.0)
		dist[4][y][x] = numpy.float32((1.0 + cusq + 3.0*vy + 4.5*vy*vy) / 9.0)
		dist[1][y][x] = numpy.float32((1.0 + cusq + 3.0*vx + 4.5*vx*vx) / 9.0)
		dist[3][y][x] = numpy.float32((1.0 + cusq - 3.0*vy + 4.5*vy*vy) / 9.0)
		dist[2][y][x] = numpy.float32((1.0 + cusq - 3.0*vx + 4.5*vx*vx) / 9.0)
		dist[7][y][x] = numpy.float32((1.0 + cusq + 3.0*(vx+vy) + 4.5*(vx+vy)*(vx+vy)) / 36.0)
		dist[5][y][x] = numpy.float32((1.0 + cusq + 3.0*(vx-vy) + 4.5*(vx-vy)*(vx-vy)) / 36.0)
		dist[6][y][x] = numpy.float32((1.0 + cusq + 3.0*(-vx-vy) + 4.5*(vx+vy)*(vx+vy)) / 36.0)
		dist[8][y][x] = numpy.float32((1.0 + cusq + 3.0*(-vx+vy) + 4.5*(-vx+vy)*(-vx+vy)) / 36.0)


class LBMSim(object):

	def __init__(self, geo_class):
		parser = OptionParser()
		parser.add_option('--lat_w', dest='lat_w', help='lattice width', type='int', action='store', default=128)
		parser.add_option('--lat_h', dest='lat_h', help='lattice height', type='int', action='store', default=128)
		parser.add_option('--visc', dest='visc', help='viscosity', type='float', action='store', default=0.01)
		parser.add_option('--scr_w', dest='scr_w', help='screen width', type='int', action='store', default=0)
		parser.add_option('--scr_h', dest='scr_h', help='screen height', type='int', action='store', default=0)
		parser.add_option('--scr_scale', dest='scr_scale', help='screen scale', type='int', action='store', default=3)
		parser.add_option('--every', dest='every', help='update the visualization every N steps', metavar='N', type='int', action='store', default=100)
		parser.add_option('--tracers', dest='tracers', help='number of tracer particles', type='int', action='store', default=32)
		parser.add_option('--model', dest='model', help='LBE model to use', type='choice', choices=['bgk', 'mrt'], action='store', default='bgk')
		parser.add_option('--vismode', dest='vismode', help='visualization mode', type='choice', choices=vis2d.vis_map.keys(), action='store', default='std')
		parser.add_option('--benchmark', dest='benchmark', help='benchmark mode, implies no visualization', action='store_true', default=False)
		parser.add_option('--benchmark_iters', dest='benchmark_iters', help='number of iterations to run in benchmark mode', action='store', type='int', default=0)

		self.geo_class = geo_class
		self.options, self.args = parser.parse_args()
		self.block_size = 64
		self._mlups_calls = 0
		self._mlups = 0.0

		# If the size of the window has not been explicitly defined, automatically adjust it
		# based on the size of the grid,
		if self.options.scr_w == 0:
			self.options.scr_w = self.options.lat_w * self.options.scr_scale

		if self.options.scr_h == 0:
			self.options.scr_h = self.options.lat_h * self.options.scr_scale

		if not self.options.benchmark:
			self.vis = vis2d.Fluid2DVis(self.options.scr_w, self.options.scr_h,
										self.options.lat_w, self.options.lat_h)

	def _init_code(self):
		fp = open('lbm.cu')
		src = fp.read()
		fp.close()
		src = '#define BLOCK_SIZE %d\n#define LAT_H %d\n#define LAT_W %d\n' % (self.block_size, self.options.lat_h, self.options.lat_w) + src
		src = '#define GEO_FLUID %d\n#define GEO_WALL %d\n#define GEO_INFLOW %d\n' % (GEO_FLUID, GEO_WALL, GEO_INFLOW) + src
		src = '#define RELAXATE RELAX_%s\n' % (self.options.model) + src

		self.mod = cuda.SourceModule(src, options=['--use_fast_math', '-Xptxas', '-v'])
		self.lbm_cnp = self.mod.get_function('LBMCollideAndPropagate')
		self.lbm_tracer = self.mod.get_function('LBMUpdateTracerParticles')

		# Set the 'tau' parameter.
		self.tau = numpy.float32((6.0 * self.options.visc + 1.0)/2.0)
		self.gpu_tau = self.mod.get_global('tau')[0]
		cuda.memcpy_htod(self.gpu_tau, self.tau)

		self.gpu_visc = self.mod.get_global('visc')[0]
		cuda.memcpy_htod(self.gpu_visc, numpy.float32(self.options.visc))

	def _init_lbm(self):
		# Velocity and density.
		self.vx = numpy.zeros((self.options.lat_h, self.options.lat_w), numpy.float32)
		self.vy = numpy.zeros((self.options.lat_h, self.options.lat_w), numpy.float32)
		self.rho = numpy.zeros((self.options.lat_h, self.options.lat_w), numpy.float32)
		self.gpu_vx = cuda.mem_alloc(self.vx.size * self.vx.dtype.itemsize)
		self.gpu_vy = cuda.mem_alloc(self.vy.size * self.vy.dtype.itemsize)
		self.gpu_rho = cuda.mem_alloc(self.rho.size * self.rho.dtype.itemsize)
		cuda.memcpy_htod(self.gpu_vx, self.vx)
		cuda.memcpy_htod(self.gpu_vy, self.vy)
		cuda.memcpy_htod(self.gpu_rho, self.rho)
		cuda.memcpy_htod(self.gpu_rho, self.rho)
		cuda.memcpy_htod(self.gpu_rho, self.rho)

		# Tracer particles.
		self.tracer_x = numpy.random.random_sample(self.options.tracers).astype(numpy.float32) * self.options.lat_w
		self.tracer_y = numpy.random.random_sample(self.options.tracers).astype(numpy.float32) * self.options.lat_h
		self.gpu_tracer_x = cuda.mem_alloc(self.tracer_x.size * self.tracer_x.dtype.itemsize)
		self.gpu_tracer_y = cuda.mem_alloc(self.tracer_y.size * self.tracer_y.dtype.itemsize)
		cuda.memcpy_htod(self.gpu_tracer_x, self.tracer_x)
		cuda.memcpy_htod(self.gpu_tracer_y, self.tracer_y)

		# Particle distributions in host memory.
		self.dist = numpy.zeros((9, self.options.lat_h, self.options.lat_w), numpy.float32)

		# Simulation geometry.
		self.geo = self.geo_class(self.options.lat_w, self.options.lat_h, self.options.model)
		self.geo.init_dist(self.dist)

		# Particle distributions in device memory, A-B access pattern.
		self.gpu_dist1 = []
		self.gpu_dist2 = []
		for i in range(0, 9):
			self.gpu_dist1.append(cuda.mem_alloc(self.vx.size * self.vx.dtype.itemsize))
			self.gpu_dist2.append(cuda.mem_alloc(self.vx.size * self.vx.dtype.itemsize))

		for i, gdist in enumerate(self.gpu_dist1):
			cuda.memcpy_htod(gdist, self.dist[i])
		for i, gdist in enumerate(self.gpu_dist2):
			cuda.memcpy_htod(gdist, self.dist[i])

		# Prepared calls to the kernel.
		self.lbm_cnp.prepare('P' * (4+2*9), block=(self.block_size,1,1), shared=(self.block_size*6*numpy.dtype(numpy.float32()).itemsize))
		self.lbm_tracer.prepare('P' * (12), block=(self.options.tracers,1,1))

		# Kernel arguments.
		self.args_tracer2 = self.gpu_dist1 + [self.geo.gpu_map, self.gpu_tracer_x, self.gpu_tracer_y]
		self.args_tracer1 = self.gpu_dist2 + [self.geo.gpu_map, self.gpu_tracer_x, self.gpu_tracer_y]
		self.args1 = [self.geo.gpu_map] + self.gpu_dist1 + self.gpu_dist2 + [0,0,0]
		self.args2 = [self.geo.gpu_map] + self.gpu_dist2 + self.gpu_dist1 + [0,0,0]

		# Special argument list for the case where macroscopic quantities data is to be
		# saved in global memory, i.e. a visualization step.
		self.args1v = [self.geo.gpu_map] + self.gpu_dist1 + self.gpu_dist2 + [self.gpu_rho, self.gpu_vx, self.gpu_vy]
		self.args2v = [self.geo.gpu_map] + self.gpu_dist2 + self.gpu_dist1 + [self.gpu_rho, self.gpu_vx, self.gpu_vy]

		# Map: iteration parity -> kernel arguments to use.
		self.args_map = {
			0: (self.args1, self.args1v, self.args_tracer1),
			1: (self.args2, self.args2v, self.args_tracer2),
		}

	def sim_step(self, i, tracers=True):
		kargs = self.args_map[i & 1]

		if not self.options.benchmark and i % self.options.every == 0:
			self.lbm_cnp.prepared_call((self.options.lat_w/self.block_size, self.options.lat_h), *kargs[1])
			if tracers:
				self.lbm_tracer.prepared_call((1,1), *kargs[2])
				cuda.memcpy_dtoh(self.tracer_x, self.gpu_tracer_x)
				cuda.memcpy_dtoh(self.tracer_y, self.gpu_tracer_y)

			cuda.memcpy_dtoh(self.vx, self.gpu_vx)
			cuda.memcpy_dtoh(self.vy, self.gpu_vy)
			cuda.memcpy_dtoh(self.rho, self.gpu_rho)
		else:
			self.lbm_cnp.prepared_call((self.options.lat_w/self.block_size, self.options.lat_h), *kargs[0])
			if tracers:
				self.lbm_tracer.prepared_call((1,1), *kargs[2])

	def get_mlups(self, tdiff, iters=None):
		if iters is not None:
			it = iters
		else:
			it = self.options.every

		mlups = float(it) * self.options.lat_w * self.options.lat_h * 1e-6 / tdiff
		self._mlups = (mlups + self._mlups * self._mlups_calls) / (self._mlups_calls + 1)
		self._mlups_calls += 1
		return (self._mlups, mlups)

	def _benchmark(self):
		i = 0

		if self.options.benchmark_iters:
			cycles = self.options.benchmark_iters
		else:
			cycles = 1000
			print '# iters mlups_avg mlups_curr'

		import time

		while True:
			t_prev = time.time()

			for iter in range(0, cycles):
				self.sim_step(i, tracers=False)
				i += 1

			cuda.Context.synchronize()
			t_now = time.time()
			print i,
			print '%.2f %.2f' % self.get_mlups(t_now - t_prev, cycles)

			if self.options.benchmark_iters:
				break

	def run(self):
		self._init_code()
		self._init_lbm()

		if self.options.benchmark:
			self._benchmark()
		else:
			self.vis.main(self)



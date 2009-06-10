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

class LBMSim(object):

	def __init__(self):
		parser = OptionParser()
		parser.add_option('--lat_w', dest='lat_w', help='lattice width', type='int', action='store', default=128)
		parser.add_option('--lat_h', dest='lat_h', help='lattice height', type='int', action='store', default=128)
		parser.add_option('--visc', dest='visc', help='viscosity', type='float', action='store', default=0.01)
		parser.add_option('--scr_w', dest='scr_w', help='screen width', type='int', action='store', default=512)
		parser.add_option('--scr_h', dest='scr_h', help='screen height', type='int', action='store', default=512)

		self.options, self.args = parser.parse_args()
		self.block_size = 64

		self.vis = vis2d.Fluid2DVis(self.options.scr_w, self.options.scr_h,
									self.options.lat_w, self.options.lat_h)

	def _init_code(self):
		fp = open('lbm.cu')
		src = fp.read()
		fp.close()
		src = '#define BLOCK_SIZE %d\n#define LAT_H %d\n#define LAT_W %d\n' % (self.block_size, self.options.lat_h, self.options.lat_w) + src
		src = '#define GEO_FLUID %d\n#define GEO_WALL %d\n#define GEO_INFLOW %d\n' % (GEO_FLUID, GEO_WALL, GEO_INFLOW) + src

		self.mod = cuda.SourceModule(src, options=['--use_fast_math'])
		self.lbm_cnp = self.mod.get_function('LBMCollideAndPropagate')

		# Set the 'tau' parameter.
		self.tau = numpy.float32((6.0 * self.options.visc + 1.0)/2.0)
		self.gpu_tau = self.mod.get_global('tau')[0]
		cuda.memcpy_htod(self.gpu_tau, self.tau)

	def _init_geo(self):
		# Initialize the map.
		self.geo_map = numpy.zeros((self.options.lat_h, self.options.lat_w), numpy.int32)
		self.gpu_geo_map = cuda.mem_alloc(self.geo_map.size * self.geo_map.dtype.itemsize)
		# bottom/top
		for i in range(0, self.options.lat_w):
			self.geo_map[0][i] = numpy.int32(GEO_WALL)
			self.geo_map[self.options.lat_h-1][i] = numpy.int32(GEO_INFLOW)
		# left/right
		for i in range(0, self.options.lat_h):
			self.geo_map[i][0] = self.geo_map[i][self.options.lat_w-1] = numpy.int32(GEO_WALL)
		cuda.memcpy_htod(self.gpu_geo_map, self.geo_map)

	def _init_lbm(self):
		self.vx = numpy.zeros((self.options.lat_h, self.options.lat_w), numpy.float32)
		self.vy = numpy.zeros((self.options.lat_h, self.options.lat_w), numpy.float32)
		self.rho = numpy.zeros((self.options.lat_h, self.options.lat_w), numpy.float32)
		self.gpu_vx = cuda.mem_alloc(self.vx.size * self.vx.dtype.itemsize)
		self.gpu_vy = cuda.mem_alloc(self.vy.size * self.vy.dtype.itemsize)
		self.gpu_rho = cuda.mem_alloc(self.rho.size * self.rho.dtype.itemsize)

		cuda.memcpy_htod(self.gpu_vx, self.vx)
		cuda.memcpy_htod(self.gpu_vy, self.vy)
		cuda.memcpy_htod(self.gpu_rho, self.rho)

		self.dist = numpy.zeros((9, self.options.lat_h, self.options.lat_w), numpy.float32)

		for x in range(0, self.options.lat_w):
			for y in range(0, self.options.lat_h):
				self.dist[0][y][x] = numpy.float32(4.0/9.0)
				self.dist[1][y][x] = self.dist[2][y][x] = self.dist[3][y][x] = self.dist[4][y][x] = numpy.float32(1.0/9.0)
				self.dist[5][y][x] = self.dist[6][y][x] = self.dist[7][y][x] = self.dist[8][y][x] = numpy.float32(1.0/36.0)

		self.gpu_dist1 = []
		self.gpu_dist2 = []
		for i in range(0, 9):
			self.gpu_dist1.append(cuda.mem_alloc(self.vx.size * self.vx.dtype.itemsize))
			self.gpu_dist2.append(cuda.mem_alloc(self.vx.size * self.vx.dtype.itemsize))

		for i, gdist in enumerate(self.gpu_dist1):
			cuda.memcpy_htod(gdist, self.dist[i])
		for i, gdist in enumerate(self.gpu_dist2):
			cuda.memcpy_htod(gdist, self.dist[i])

		self.lbm_cnp.prepare('P' * (4+2*9), block=(self.block_size,1,1), shared=(self.block_size*6*numpy.dtype(numpy.float32()).itemsize))

		self.args1v = [self.gpu_geo_map] + self.gpu_dist1 + self.gpu_dist2 + [self.gpu_rho, self.gpu_vx, self.gpu_vy]
		self.args1 = [self.gpu_geo_map] + self.gpu_dist1 + self.gpu_dist2 + [0,0,0]
		self.args2 = [self.gpu_geo_map] + self.gpu_dist2 + self.gpu_dist1 + [0,0,0]

	def update_map(self):
		cuda.memcpy_htod(self.gpu_geo_map, self.geo_map)

	def sim_step(self, i):
		if i % 2 == 0:
			if i % 100 == 0:
				self.lbm_cnp.prepared_call((self.options.lat_w/self.block_size, self.options.lat_h), *self.args1v)
				cuda.memcpy_dtoh(self.vx, self.gpu_vx)
				cuda.memcpy_dtoh(self.vy, self.gpu_vy)
				cuda.memcpy_dtoh(self.rho, self.gpu_rho)
			else:
				self.lbm_cnp.prepared_call((self.options.lat_w/self.block_size, self.options.lat_h), *self.args1)
		else:
			self.lbm_cnp.prepared_call((self.options.lat_w/self.block_size, self.options.lat_h), *self.args2)

	def run(self):
		self._init_code()
		self._init_geo()
		self._init_lbm()
		self.vis.main(self)

lbm = LBMSim()
lbm.run()

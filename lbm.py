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

parser = OptionParser()
parser.add_option('--lat_w', dest='lat_w', help='lattice width', type='int', action='store', default=128)
parser.add_option('--lat_h', dest='lat_h', help='lattice height', type='int', action='store', default=128)
parser.add_option('--visc', dest='visc', help='viscosity', type='float', action='store', default=0.01)
parser.add_option('--scr_w', dest='scr_w', help='screen width', type='int', action='store', default=512)
parser.add_option('--scr_h', dest='scr_h', help='screen height', type='int', action='store', default=512)

options, args = parser.parse_args()
block_size = 64

vis2d.init(options)

f = open('lbm_py.cu')
src = f.read()
f.close()
src = '#define BLOCK_SIZE %d\n#define LAT_H %d\n#define LAT_W %d\n' % (block_size, options.lat_h, options.lat_w) + src
src = '#define GEO_FLUID %d\n#define GEO_WALL %d\n#define GEO_INFLOW %d\n' % (GEO_FLUID, GEO_WALL, GEO_INFLOW) + src

mod = cuda.SourceModule(src, options=['--use_fast_math'])
lbm_cnp = mod.get_function('LBMCollideAndPropagate')

# Set the 'tau' parameter.
tau = numpy.float32((6.0 * options.visc + 1.0)/2.0)
gpu_tau = mod.get_global('tau')[0]
cuda.memcpy_htod(gpu_tau, tau)

# Initialize the map.
map = numpy.zeros((options.lat_h, options.lat_w), numpy.int32)
gpu_map = cuda.mem_alloc(map.size * map.dtype.itemsize)
# bottom/top
for i in range(0, options.lat_w):
	map[0][i] = numpy.int32(GEO_WALL)
	map[options.lat_h-1][i] = numpy.int32(GEO_INFLOW)
# left/right
for i in range(0, options.lat_h):
	map[i][0] = map[i][options.lat_w-1] = numpy.int32(GEO_WALL)
cuda.memcpy_htod(gpu_map, map)

vx = numpy.zeros((options.lat_h, options.lat_w), numpy.float32)
vy = numpy.zeros((options.lat_h, options.lat_w), numpy.float32)
rho = numpy.zeros((options.lat_h, options.lat_w), numpy.float32)
gpu_vx = cuda.mem_alloc(vx.size * vx.dtype.itemsize)
gpu_vy = cuda.mem_alloc(vy.size * vy.dtype.itemsize)
gpu_rho = cuda.mem_alloc(rho.size * rho.dtype.itemsize)

cuda.memcpy_htod(gpu_vx, vx)
cuda.memcpy_htod(gpu_vy, vy)
cuda.memcpy_htod(gpu_rho, rho)

dist = numpy.zeros((9, options.lat_h, options.lat_w), numpy.float32)

for x in range(0, options.lat_w):
	for y in range(0, options.lat_h):
		dist[0][y][x] = numpy.float32(4.0/9.0)
		dist[1][y][x] = dist[2][y][x] = dist[3][y][x] = dist[4][y][x] = numpy.float32(1.0/9.0)
		dist[5][y][x] = dist[6][y][x] = dist[7][y][x] = dist[8][y][x] = numpy.float32(1.0/36.0)

gpu_dist1 = []
gpu_dist2 = []
for i in range(0, 9):
	gpu_dist1.append(cuda.mem_alloc(vx.size * vx.dtype.itemsize))
	gpu_dist2.append(cuda.mem_alloc(vx.size * vx.dtype.itemsize))

for i, gdist in enumerate(gpu_dist1):
	cuda.memcpy_htod(gdist, dist[i])
for i, gdist in enumerate(gpu_dist2):
	cuda.memcpy_htod(gdist, dist[i])

lbm_cnp.prepare('P' * (4+2*9), block=(block_size,1,1), shared=(block_size*6*numpy.dtype(numpy.float32()).itemsize))

args1v = [gpu_map] + gpu_dist1 + gpu_dist2 + [gpu_rho, gpu_vx, gpu_vy]
args1 = [gpu_map] + gpu_dist1 + gpu_dist2 + [0,0,0]
args2 = [gpu_map] + gpu_dist2 + gpu_dist1 + [0,0,0]

def update_map():
	global map, gpu_map
	cuda.memcpy_htod(gpu_map, map)

def sim_step(i):
	global lbm_cnp, options, vx, vy, gpu_vx, gpu_vy, rho, gpu_rho
	if i % 2 == 0:
		if i % 100 == 0:
			lbm_cnp.prepared_call((options.lat_w/block_size, options.lat_h), *args1v)
			cuda.memcpy_dtoh(vx, gpu_vx)
			cuda.memcpy_dtoh(vy, gpu_vy)
			cuda.memcpy_dtoh(rho, gpu_rho)
		else:
			lbm_cnp.prepared_call((options.lat_w/block_size, options.lat_h), *args1)
	else:
		lbm_cnp.prepared_call((options.lat_w/block_size, options.lat_h), *args2)

vis2d.main(options, update_map, sim_step, map, vx, vy, rho)


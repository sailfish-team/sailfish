#!/usr/bin/python

import pycuda.autoinit
import pycuda.driver as cuda
import math
import numpy
import sys
import time

import pygame

from optparse import OptionParser, OptionValueError


parser = OptionParser()
parser.add_option('--lat_w', dest='lat_w', help='lattice width', type='int', action='store', default=128)
parser.add_option('--lat_h', dest='lat_h', help='lattice height', type='int', action='store', default=128)
parser.add_option('--visc', dest='visc', help='viscosity', type='float', action='store', default=0.01)
parser.add_option('--scr_w', dest='scr_w', help='screen width', type='int', action='store', default=512)
parser.add_option('--scr_h', dest='scr_h', help='screen height', type='int', action='store', default=512)

options, args = parser.parse_args()
block_size = 64

pygame.init()
screen = pygame.display.set_mode((options.scr_w, options.scr_h), pygame.RESIZABLE)
screen.fill((0,0,0))

f = open('lbm_py.cu')
src = f.read()
f.close()
src = '#define BLOCK_SIZE %d\n#define LAT_H %d\n#define LAT_W %d\n' % (block_size, options.lat_h, options.lat_w) + src

mod = cuda.SourceModule(src, options=['--use_fast_math'])
lbm_cnp = mod.get_function('LBMCollideAndPropagate')

GEO_FLUID = 0
GEO_WALL = 1
GEO_INFLOW = 2

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

dist1 = numpy.zeros((9, options.lat_h, options.lat_w), numpy.float32)
dist2 = numpy.zeros((9, options.lat_h, options.lat_w), numpy.float32)

for x in range(0, options.lat_w):
	for y in range(0, options.lat_h):
		dist1[0][y][x] = numpy.float32(4.0/9.0)
		dist1[1][y][x] = dist1[2][y][x] = dist1[3][y][x] = dist1[4][y][x] = numpy.float32(1.0/9.0)
		dist1[5][y][x] = dist1[6][y][x] = dist1[7][y][x] = dist1[8][y][x] = numpy.float32(1.0/36.0)

gpu_dist1 = []
gpu_dist2 = []
for i in range(0, 9):
	gpu_dist1.append(cuda.mem_alloc(vx.size * vx.dtype.itemsize))
	gpu_dist2.append(cuda.mem_alloc(vx.size * vx.dtype.itemsize))

for i, dist in enumerate(gpu_dist1):
	cuda.memcpy_htod(dist, dist1[i])
for i, dist in enumerate(gpu_dist2):
	cuda.memcpy_htod(dist, dist1[i])

lbm_cnp.prepare('P' * (4+2*9), block=(block_size,1,1), shared=(block_size*6*numpy.dtype(numpy.float32()).itemsize))

args1 = [gpu_map] + gpu_dist1 + gpu_dist2 + [gpu_rho, gpu_vx, gpu_vy]
args2 = [gpu_map] + gpu_dist2 + gpu_dist1 + [gpu_rho, gpu_vx, gpu_vy]

pygame.surfarray.use_arraytype('numpy')

def visualize(screen, vx, vy, rho):
	height, width = vx.shape
	srf = pygame.Surface((width, height))

	drw = numpy.sqrt(vx*vx + vy*vy) / 0.1 * 255
	drw = drw.astype(numpy.int32)

	a = pygame.surfarray.pixels3d(srf)
	b = numpy.rot90(map == GEO_WALL, 3)

	a[b] = (0,0,255)
	b = numpy.logical_not(b)

	drw = numpy.rot90(drw, 3).reshape((height, width, 1)) * numpy.int32([1,1,0])
	a[b] = drw[b]

	del a
	pygame.transform.scale(srf, screen.get_size(), screen)


i = 0

font = pygame.font.SysFont('Liberation Mono', 14)

t_prev = time.time()
drawing = False
draw_type = 1

def get_loc(event, screen, options):
	x = event.pos[0] * options.lat_w / screen.get_width()
	y = options.lat_h-1 - (event.pos[1] * options.lat_h / screen.get_height())
	return min(max(x, 0), options.lat_w-1), min(max(y, 0), options.lat_h-1)

while 1:
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			sys.exit()
		elif event.type == pygame.VIDEORESIZE:
			screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)

		elif event.type == pygame.MOUSEBUTTONUP:
			x, y = get_loc(event, screen, options)
			draw_type = event.button
			map[y][x] = draw_type == 1 and GEO_WALL or GEO_FLUID
			drawing = False
			cuda.memcpy_htod(gpu_map, map)
		elif event.type == pygame.MOUSEBUTTONDOWN:
			x, y = get_loc(event, screen, options)
			draw_type = event.button
			map[y][x] = draw_type == 1 and GEO_WALL or GEO_FLUID
			drawing = True
			cuda.memcpy_htod(gpu_map, map)

		elif event.type == pygame.MOUSEMOTION:
			if drawing:
				x, y = get_loc(event, screen, options)
				map[y][x] = draw_type == 1 and GEO_WALL or GEO_FLUID
				cuda.memcpy_htod(gpu_map, map)

	if i % 2 == 0:
		lbm_cnp.prepared_call((options.lat_w/block_size, options.lat_h), *args1)

		if i % 100 == 0:
			cuda.memcpy_dtoh(vx, gpu_vx)
			cuda.memcpy_dtoh(vy, gpu_vy)
			cuda.memcpy_dtoh(rho, gpu_rho)

			t_now = time.time()
			mlups = 100.0 * options.lat_w * options.lat_h * 1e-6 / (t_now - t_prev)
			t_prev = t_now

			visualize(screen, vx, vy, rho)
			perf = font.render('%.2f MLUPS' % mlups, True, (0,255,0))
			screen.blit(perf, (12, 12))
			pygame.display.flip()

			t_prev = time.time()

			print t_prev - t_now
	else:
		lbm_cnp.prepared_call((options.lat_w/block_size, options.lat_h), *args2)

	i += 1


import math
import numpy
import pygame
import geo2d
import os
import sys
import time
import sym

pygame.init()
pygame.surfarray.use_arraytype('numpy')

def hsv_to_rgb(a):
	t = a[:,:,0]*6.0
	i = t.astype(numpy.uint8)
	f = t - numpy.floor(t)

	v = a[:,:,2]

	o = numpy.ones_like(a[:,:,0])
	p = v * (o - a[:,:,1])
	q = v * (o - a[:,:,1]*f)
	t = v * (o - a[:,:,1]*(o - f))

	i = numpy.mod(i, 6)
	sh = i.shape
	i = i.reshape(sh[0], sh[1], 1) * numpy.uint8([1,1,1])

	choices = [numpy.dstack((v, t, p)),
			   numpy.dstack((q, v, p)),
			   numpy.dstack((p, v, t)),
			   numpy.dstack((p, q, v)),
			   numpy.dstack((t, p, v)),
			   numpy.dstack((v, p, q))]

	return numpy.choose(i, choices)

def _vis_hsv(drw, width, height):
	drw = drw.reshape((width, height, 1)) * numpy.float32([1.0, 1.0, 1.0])
	drw[:,:,2] = 1.0
	drw[:,:,1] = 1.0
	drw = hsv_to_rgb(drw) * 255.0
	return drw.astype(numpy.uint8)

def _vis_std(drw, width, height):
	return (drw.reshape((width, height, 1)) * 255.0).astype(numpy.uint8) * numpy.uint8([1,1,0])

def _vis_rgb1(drw, width, height):
	"""This is the default color palette from gnuplot."""
	r = numpy.sqrt(drw)
	g = numpy.power(drw, 3)
	b = numpy.sin(drw * math.pi)

	return (numpy.dstack([r,g,b]) * 250.0).astype(numpy.uint8)


vis_map = {
	'std': _vis_std,
	'rgb1': _vis_rgb1,
	'hsv': _vis_hsv,
	}

class Fluid2DVis(object):

	def __init__(self, width, height, lat_w, lat_h):
		self._vismode = 0
		self._font = pygame.font.SysFont('Liberation Mono', 14)
		self._screen = pygame.display.set_mode((width, height),
				pygame.RESIZABLE)
		self.lat_w = lat_w
		self.lat_h = lat_h

		self._tracers = False
		self._velocity = False
		self._drawing = False
		self._paused = False
		self._draw_type = 1

	def _visualize(self, sim, vx, vy, rho, tx, ty, vismode):

		if sym.GRID.dim == 3:
			vx = vx[10,:,:]
			vy = vy[10,:,:]
			rho = rho[10,:,:]

		height, width = vx.shape
		srf = pygame.Surface((width, height))

		maxv = numpy.max(numpy.sqrt(vx*vx + vy*vy))
		ret = []

		ret.append(('max_v', maxv))
		ret.append(('rho_avg', numpy.average(rho)))

		b = (sim.geo.map_to_node_type(sim.geo.map[10,:,:]) == geo2d.LBMGeo.NODE_WALL)

		if self._vismode == 0:
			drw = numpy.sqrt(vx*vx + vy*vy) / maxv
		elif self._vismode == 1:
			drw = numpy.abs(vx) / maxv
		elif self._vismode == 2:
			drw = numpy.abs(vy) / maxv
		elif self._vismode == 3:
			mrho = numpy.ma.array(rho, mask=(b))
			rho_min = numpy.min(mrho)
			rho_max = numpy.max(mrho)
			drw	= ((rho - rho_min) / (rho_max - rho_min))

		# Rotate the field to the correct position.
		drw = numpy.rot90(drw.astype(numpy.float32), 3)
		a = pygame.surfarray.pixels3d(srf)
		b = numpy.rot90(b, 3)

		# Draw the walls.
		a[b] = (0, 0, 255)

		# Draw the data field for all sites which are not marked as a wall.
		b = numpy.logical_not(b)
		drw = vis_map[vismode](drw, width, height)
		a[b] = drw[b]

		# Unlock the surface and put the picture on screen.
		del a
		pygame.transform.scale(srf, self._screen.get_size(), self._screen)

		sw, sh = self._screen.get_size()

		# Draw the velocity field
		if self._velocity:
			vfsp = 21
			scale = 0.8 * max(sh, sw)/(vfsp-1) / maxv

			for i in range(1, vfsp):
				for j in range(1, vfsp):
					ox = sw*i/vfsp
					oy = sh - sh*j/vfsp

					pygame.draw.line(self._screen, (255, 0, 0),
									 (ox, oy),
									 (ox + vx[height*j/vfsp][width*i/vfsp] * scale,
									  oy - vy[height*j/vfsp][width*i/vfsp] * scale))

		# Draw the tracer particles
		if self._tracers:
			for x, y in zip(tx, ty):
				pygame.draw.circle(self._screen, (0, 255, 0), (int(x * sw / width), int(sh - y * sh / height)), 2)

		return ret

	def _get_loc(self, event):
		x = event.pos[0] * self.lat_w / self._screen.get_width()
		y = self.lat_h-1 - (event.pos[1] * self.lat_h / self._screen.get_height())
		return min(max(x, 0), self.lat_w-1), min(max(y, 0), self.lat_h-1)

	def _draw_wall(self, lbm_sim, event):
		x, y = self._get_loc(event)
		lbm_sim.geo.set_geo((x, y),
				self._draw_type == 1 and geo2d.LBMGeo.NODE_WALL or geo2d.LBMGeo.NODE_FLUID,
				update=True)

	def _process_events(self, lbm_sim, curr_iter):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()
			elif event.type == pygame.VIDEORESIZE:
				self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
			elif event.type == pygame.MOUSEBUTTONUP:
				self._draw_type = event.button
				self._draw_wall(lbm_sim, event)
				self._drawing = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				self._draw_type = event.button
				self._draw_wall(lbm_sim, event)
				self._drawing = True
			elif event.type == pygame.MOUSEMOTION:
				if self._drawing:
					self._draw_wall(lbm_sim, event)
			elif event.type == pygame.KEYUP:
				if event.key == pygame.K_0:
					self._vismode = 0
				elif event.key == pygame.K_1:
					self._vismode = 1
				elif event.key == pygame.K_2:
					self._vismode = 2
				elif event.key == pygame.K_3:
					self._vismode = 3
				elif event.key == pygame.K_v:
					self._velocity = not self._velocity
				elif event.key == pygame.K_t:
					self._tracers = not self._tracers
				elif event.key == pygame.K_p:
					self._paused = not self._paused
					if self._paused:
						print 'Simulation paused @ iter = %d.' % curr_iter
				elif event.key == pygame.K_q:
					sys.exit()
				elif event.key == pygame.K_r:
					lbm_sim.geo.reset()
				elif event.key == pygame.K_s:
					i = 0

					while os.path.exists('%s_%05d.png' % (lbm_sim.filename, i)):
						i += 1
						if i > 99999:
							break

					fname = '%s_%05d.png' % (lbm_sim.filename, i)
					if os.path.exists(fname):
						print 'Could not create screenshot.'

					pygame.image.save(self._screen, fname)
					print 'Saved %s.' % fname

	def main(self, lbm_sim):
		i = 1
		t_prev = time.time()
		avg_mlups = 0.0

		while 1:
			self._process_events(lbm_sim, i)

			if self._paused:
				continue

			lbm_sim.sim_step(i, self._tracers)

			if i % lbm_sim.options.every == 0:
				t_now = time.time()
				avg_mlups, mlups = lbm_sim.get_mlups(t_now - t_prev)
				t_prev = t_now

				ret = self._visualize(lbm_sim, lbm_sim.vx, lbm_sim.vy,
						lbm_sim.rho, lbm_sim.tracer_x, lbm_sim.tracer_y,
						lbm_sim.options.vismode)
				self._screen.blit(self._font.render('itr: %dk' % (i / 1000), True, (0, 255, 0)), (12, 12))
				self._screen.blit(self._font.render('cur: %.2f MLUPS' % mlups, True, (0, 255, 0)), (12, 24))
				self._screen.blit(self._font.render('avg: %.2f MLUPS' % avg_mlups, True, (0, 255, 0)), (12, 36))

				y = 48
				for info in ret:
					tmp = self._font.render('%s: %.3f' % info, True, (0, 255, 0))
					self._screen.blit(tmp, (12, y))
					y += 12

				pygame.display.flip()

				t_prev = time.time()

			i += 1


import time
import pygame
import numpy
import sim
import sys

pygame.init()
pygame.surfarray.use_arraytype('numpy')

class Fluid2DVis(object):

	def __init__(self, width, height, lat_w, lat_h):
		self._vismode = 0
		self._font = pygame.font.SysFont('Liberation Mono', 14)
		self._screen = pygame.display.set_mode((width, height),
				pygame.RESIZABLE)
		self.lat_w = lat_w
		self.lat_h = lat_h

		self._velocity = True
		self._drawing = False
		self._draw_type = 1

	def _visualize(self, geo_map, vx, vy, rho):
		height, width = vx.shape
		srf = pygame.Surface((width, height))

		if self._vismode == 0:
			drw = numpy.sqrt(vx*vx + vy*vy) / 0.1 * 255
		elif self._vismode == 1:
			drw = numpy.abs(vx) / 0.1 * 255
		elif self._vismode == 2:
			drw = numpy.abs(vy) / 0.1 * 255
		elif self._vismode == 3:
			t = numpy.abs(rho - 1.00)
			drw	= t / numpy.max(t) * 255

		# Rotate the field to the correct position.
		drw = numpy.rot90(drw.astype(numpy.uint8), 3)
		a = pygame.surfarray.pixels3d(srf)

		# Draw the walls.
		b = numpy.rot90(geo_map == sim.GEO_WALL, 3)
		a[b] = (0, 0, 255)

		# Draw the data field for all sites which are not marked as a wall.
		b = numpy.logical_not(b)
		drw = drw.reshape((width, height, 1)) * numpy.uint8([1, 1, 0])
		a[b] = drw[b]

		# Unlock the surface and put the picture on screen.
		del a
		pygame.transform.scale(srf, self._screen.get_size(), self._screen)

		# Draw the velocity field
		if self._velocity:
			max_v = 0.1
			scale = max(height, width)/10 / max_v * 2

			sw, sh = self._screen.get_size()
			for i in range(1, 11):
				for j in range(1, 11):
					ox = sw*i/11
					oy = sh - sh*j/11

					pygame.draw.line(self._screen, (255, 0, 0),
									 (ox, oy),
									 (ox + vx[height*j/11][width*i/11] * scale,
									  oy - vy[height*j/11][width*i/11] * scale))

	def _get_loc(self, event):
		x = event.pos[0] * self.lat_w / self._screen.get_width()
		y = self.lat_h-1 - (event.pos[1] * self.lat_h / self._screen.get_height())
		return min(max(x, 0), self.lat_w-1), min(max(y, 0), self.lat_h-1)

	def _draw_wall(self, update_map, geo_map, event):
		x, y = self._get_loc(event)
		geo_map[y][x] = self._draw_type == 1 and sim.GEO_WALL or sim.GEO_FLUID
		update_map()

	def _process_events(self, update_map, geo_map):
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()
			elif event.type == pygame.VIDEORESIZE:
				self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
			elif event.type == pygame.MOUSEBUTTONUP:
				self._draw_type = event.button
				self._draw_wall(update_map, geo_map, event)
				self._drawing = False
			elif event.type == pygame.MOUSEBUTTONDOWN:
				self._draw_type = event.button
				self._draw_wall(update_map, geo_map, event)
				self._drawing = True
			elif event.type == pygame.MOUSEMOTION:
				if self._drawing:
					self._draw_wall(update_map, geo_map, event)
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
				elif event.key == pygame.K_q:
					sys.exit()

	def main(self, lbm_sim):
		i = 0
		t_prev = time.time()
		while 1:
			self._process_events(lbm_sim.update_map, lbm_sim.geo_map)
			lbm_sim.sim_step(i)

			if i % lbm_sim.options.every == 0:
				t_now = time.time()
				mlups = float(lbm_sim.options.every) * self.lat_w * self.lat_h * 1e-6 / (t_now - t_prev)
				t_prev = t_now

				self._visualize(lbm_sim.geo_map, lbm_sim.vx, lbm_sim.vy, lbm_sim.rho)
				perf = self._font.render('%.2f MLUPS' % mlups, True, (0, 255, 0))
				self._screen.blit(perf, (12, 12))
				pygame.display.flip()

				t_prev = time.time()
				print t_prev - t_now

			i += 1


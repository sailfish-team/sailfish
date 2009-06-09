import time
import pygame
import numpy
import sim

def init(options):
	global font, screen

	# Clear the screen.
	pygame.init()
	pygame.surfarray.use_arraytype('numpy')
	font = pygame.font.SysFont('Liberation Mono', 14)
	screen = pygame.display.set_mode((options.scr_w, options.scr_h), pygame.RESIZABLE)
	screen.fill((0,0,0))

def visualize(screen, map, vx, vy, rho):
	height, width = vx.shape
	srf = pygame.Surface((width, height))

	drw = numpy.sqrt(vx*vx + vy*vy) / 0.1 * 255
	drw = drw.astype(numpy.int32)

	a = pygame.surfarray.pixels3d(srf)
	b = numpy.rot90(map == sim.GEO_WALL, 3)

	a[b] = (0,0,255)
	b = numpy.logical_not(b)

	drw = numpy.rot90(drw, 3).reshape((height, width, 1)) * numpy.int32([1,1,0])
	a[b] = drw[b]

	del a
	pygame.transform.scale(srf, screen.get_size(), screen)

def get_loc(event, screen, options):
	x = event.pos[0] * options.lat_w / screen.get_width()
	y = options.lat_h-1 - (event.pos[1] * options.lat_h / screen.get_height())
	return min(max(x, 0), options.lat_w-1), min(max(y, 0), options.lat_h-1)

def main(options, update_map, sim_step, map, vx, vy, rho):
	i = 0
	t_prev = time.time()
	drawing = False
	draw_type = 1

	global font, screen

	while 1:
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				sys.exit()
			elif event.type == pygame.VIDEORESIZE:
				screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
			elif event.type == pygame.MOUSEBUTTONUP:
				x, y = get_loc(event, screen, options)
				draw_type = event.button
				map[y][x] = draw_type == 1 and sim.GEO_WALL or sim.GEO_FLUID
				drawing = False
				update_map()
			elif event.type == pygame.MOUSEBUTTONDOWN:
				x, y = get_loc(event, screen, options)
				draw_type = event.button
				map[y][x] = draw_type == 1 and sim.GEO_WALL or sim.GEO_FLUID
				drawing = True
				update_map()
			elif event.type == pygame.MOUSEMOTION:
				if drawing:
					x, y = get_loc(event, screen, options)
					map[y][x] = draw_type == 1 and sim.GEO_WALL or sim.GEO_FLUID
					update_map()

		sim_step(i)

		if i % 100 == 0:
			t_now = time.time()
			mlups = 100.0 * options.lat_w * options.lat_h * 1e-6 / (t_now - t_prev)
			t_prev = t_now

			visualize(screen, map, vx, vy, rho)
			perf = font.render('%.2f MLUPS' % mlups, True, (0,255,0))
			screen.blit(perf, (12, 12))
			pygame.display.flip()

			t_prev = time.time()
			print t_prev - t_now

		i += 1


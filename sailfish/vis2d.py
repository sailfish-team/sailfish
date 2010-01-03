import math
import numpy
import pygame

import os
import sys
import time

from sailfish import geo
from sailfish import sym

from scipy import signal

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
    drw = numpy.abs(drw)
    drw = drw.reshape((width, height, 1)) * numpy.float32([1.0, 1.0, 1.0])
    drw[:,:,2] = 1.0
    drw[:,:,1] = 1.0
    drw = hsv_to_rgb(drw) * 255.0
    return drw.astype(numpy.uint8)

def _vis_std(drw, width, height):
    drw = numpy.abs(drw)
    return (drw.reshape((width, height, 1)) * 255.0).astype(numpy.uint8) * numpy.uint8([1,1,0])

def _vis_2col(drw, width, height):
    drw = ((drw*(drw>0).astype(int)).reshape((width, height, 1)) * numpy.uint8([255, 0, 0])
        - ( drw*(drw<0).astype(int)).reshape((width, height, 1)) * numpy.uint8([0, 0, 255]))
    drw[drw>255] = 255
    drw[drw<-255] = -255
    return drw.astype(numpy.uint8)

def _vis_rgb1(drw, width, height):
    """This is the default color palette from gnuplot."""
    drw = numpy.abs(drw)
    r = numpy.sqrt(drw)
    g = numpy.power(drw, 3)
    b = numpy.sin(drw * math.pi)

    return (numpy.dstack([r,g,b]) * 250.0).astype(numpy.uint8)

def gauss_kernel(size, sizey=None):
    """Return a normalized 2D gauss kernel array for convolutions"""
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = numpy.mgrid[-size:size+1, -sizey:sizey+1]
    g = numpy.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

vis_map = {
    'std': _vis_std,
    'rgb1': _vis_rgb1,
    'hsv': _vis_hsv,
    '2col': _vis_2col,
    }

class Fluid2DVis(object):

    def __init__(self, sim, width, height, lat_w, lat_h):
        self._vismode = 0
        self._convolve = False
        self._font = pygame.font.SysFont('Liberation Mono', 14)
        self._screen = pygame.display.set_mode((width, height),
                pygame.RESIZABLE)
        self.lat_w = lat_w
        self.lat_h = lat_h

        self._show_info = True
        self._tracers = False
        self._velocity = False
        self._drawing = False
        self._paused = False
        self._draw_type = 1
        self._reset()
        self.sim = sim

        pygame.key.set_repeat(100,50)

    def _reset(self):
        self._maxv = 0.000001
        self._vscale  = 0.005

    @property
    def velocity_norm(self):
        return self.sim.geo.mask_array_by_fluid(numpy.sqrt(self.sim.vx*self.sim.vx + self.sim.vy*self.sim.vy))

    @property
    def vx(self):
        return self.sim.vx

    @property
    def vy(self):
        return self.sim.vy

    @property
    def density(self):
        return self.sim.rho

    @property
    def geo_map(self):
        return self.sim.geo.map

    def _visualize(self, tx, ty, vismode):
        height, width = self.vx.shape
        srf = pygame.Surface((width, height))

        maxv = numpy.max(self.velocity_norm)
        ret = []

        # Record the highest velocity seen to this moment.
        if self._maxv is None or maxv > self._maxv:
            self._maxv = maxv

        ret.append('max_v: %.3f' % maxv)
        ret.append('rho_avg: %.3f' % numpy.average(self.density))

        b = (self.sim.geo._decode_node_type(self.geo_map) == geo.LBMGeo.NODE_WALL)

        if self._vismode == 0:
            drw = self.velocity_norm / self._maxv
        elif self._vismode == 1:
            drw = self.vx / self._maxv
        elif self._vismode == 2:
            drw = self.vy / self._maxv
        elif self._vismode == 3:
            mrho = numpy.ma.array(self.density, mask=(b))
            rho_min = numpy.min(mrho)
            rho_max = numpy.max(mrho)
            drw = ((self.density - rho_min) / (rho_max - rho_min))
        elif self._vismode == 4:
            self.curl_v = (numpy.hstack( (self.vy[:,1:]-self.vy[:,:-1],numpy.zeros((height,1))))
                    - numpy.vstack( (self.vx[1:,:]-self.vx[:-1,:],numpy.zeros((1,width)))))
            drw = -(self.curl_v) / self._vscale
            if self._convolve:
                g = gauss_kernel(2, sizey=2)
                drw = signal.convolve(drw,g, mode='same')

        # Rotate the field to the correct position.
        drw = numpy.rot90(drw.astype(numpy.float32), 3)
        a = pygame.surfarray.pixels3d(srf)
        b = numpy.rot90(b, 3)
        # Draw the walls.
        a[b] = (255, 255, 255)

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
                                     (ox + self.vx[height*j/vfsp][width*i/vfsp] * scale,
                                      oy - self.vy[height*j/vfsp][width*i/vfsp] * scale))

        self._draw_tracers(tx, ty, sw, sh, width, height)
        return ret

    def _draw_tracers(self, tx, ty, sw, sh, width, height):
        # Draw the tracer particles
        if self._tracers:
            for x, y in zip(tx, ty):
                pygame.draw.circle(self._screen, (0, 255, 255), (int(x * sw / width), int(sh - y * sh / height)), 2)

    def _get_loc(self, event):
        x = event.pos[0] * self.lat_w / self._screen.get_width()
        y = self.lat_h-1 - (event.pos[1] * self.lat_h / self._screen.get_height())
        return min(max(x, 0), self.lat_w-1), min(max(y, 0), self.lat_h-1)

    def _draw_wall(self, event):
        x, y = self._get_loc(event)
        self.sim.geo.set_geo((x, y),
                self._draw_type == 1 and geo.LBMGeo.NODE_WALL or geo.LBMGeo.NODE_FLUID,
                update=True)

    def _process_misc_event(self, event):
        """A function to make it possible to process additional events in subclasses."""
        pass

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(event.size, pygame.RESIZABLE)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._draw_type = event.button
                self._draw_wall(event)
                self._drawing = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._draw_type = event.button
                self._draw_wall(event)
                self._drawing = True
            elif event.type == pygame.MOUSEMOTION:
                if self._drawing:
                    self._draw_wall(event)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_0:
                    self._vismode = 0
                elif event.key == pygame.K_1:
                    self._vismode = 1
                elif event.key == pygame.K_2:
                    self._vismode = 2
                elif event.key == pygame.K_3:
                    self._vismode = 3
                elif event.key == pygame.K_4:
                    self._vismode = 4
                elif event.key == pygame.K_v:
                    self._velocity = not self._velocity
                elif event.key == pygame.K_t:
                    self._tracers = not self._tracers
                elif event.key == pygame.K_c:
                    self._convolve = not self._convolve
                elif event.key == pygame.K_p:
                    self._paused = not self._paused
                    if self._paused:
                        print 'Simulation paused @ iter = %d.' % self.sim.iter_
                elif event.key == pygame.K_q:
                    sys.exit()
                elif event.key == pygame.K_r:
                    self._reset()
                    self.sim.geo.reset()
                elif event.key == pygame.K_s:
                    i = 0

                    while os.path.exists('%s_%05d.png' % (self.sim.filename, i)):
                        i += 1
                        if i > 99999:
                            break

                    fname = '%s_%05d.png' % (self.sim.filename, i)
                    if os.path.exists(fname):
                        print 'Could not create screenshot.'

                    pygame.image.save(self._screen, fname)
                    print 'Saved %s.' % fname
                elif event.key == pygame.K_i:
                    self._show_info = not self._show_info
                elif event.key == pygame.K_COMMA:
                    if self._vismode == 4:
                        self._vscale /= 1.1
                    else:
                        self._maxv /= 1.1
                elif event.key == pygame.K_PERIOD:
                    if self._vismode == 4:
                        self._vscale *= 1.1
                    else:
                        self._maxv *= 1.1

            self._process_misc_event(event)

    def main(self):
        t_prev = time.time()
        avg_mlups = 0.0

        while 1:
            self._process_events()

            if self._paused:
                continue

            i = self.sim.iter_
            self.sim.sim_step(self._tracers)

            if i % self.sim.options.every == 0 and i:
                avg_mlups, mlups = self.sim.get_mlups(time.time() - t_prev)

                ret = self._visualize(self.sim.tracer_x, self.sim.tracer_y, self.sim.options.vismode)

                if self._show_info:
                    self._screen.blit(self._font.render('itr: %dk' % (i / 1000), True, (0, 255, 0)), (12, 12))
                    self._screen.blit(self._font.render('tim: %.4f' % self.sim.time, True, (0, 255, 0)), (12, 24))
                    self._screen.blit(self._font.render('c/a: %.2f / %.2f MLUPS' % (mlups, avg_mlups), True, (0, 255, 0)), (12, 36))

                    y = 48
                    for info in ret:
                        tmp = self._font.render(info, True, (0, 255, 0))
                        self._screen.blit(tmp, (12, y))
                        y += 12

                pygame.display.flip()
                t_prev = time.time()

class Fluid3DVisCutplane(Fluid2DVis):

    def __init__(self, sim, shape, scr_scale):
        Fluid2DVis.__init__(self, sim, int(shape[0] * scr_scale), int(shape[1] * scr_scale), shape[0], shape[1])
        self.shape = shape
        self._scr_scale = scr_scale
        self._cut_dim = 2
        self._cut_pos = [self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2]
        self._reset_display()

    @property
    def _slice_args(self):
        args = []

        for i in range(2, -1, -1):
            if i == self._cut_dim:
                args.append(self._cut_pos[self._cut_dim])
            else:
                args.append(slice(None))
        return args

    @property
    def velocity_norm(self):
        # FIXME: This should be masked by fluid.
        return numpy.sqrt(self.vx*self.vx + self.vy*self.vy + self.vz*self.vz)

    @property
    def vx(self):
        return self.sim.velocity[self._dims[0]][self._slice_args]

    @property
    def vy(self):
        return self.sim.velocity[self._dims[1]][self._slice_args]

    @property
    def vz(self):
        return self.sim.velocity[self._cut_dim][self._slice_args]

    @property
    def density(self):
        return self.sim.rho[self._slice_args]

    @property
    def geo_map(self):
        return self.sim.geo.map[self._slice_args]

    def _2d_to_3d_loc(self, x, y):
        """Translate a location on the currently visible slice into a location in the 3D lattice.

        Args:
          x, y: coordinates on the current 2D slice

        Returns:
          x, y, z: coordinates in the simulation domain
        """
        loc = [0,0,0]
        loc[self._cut_dim] = self._cut_pos[self._cut_dim]
        loc[self._dims[0]] = x
        loc[self._dims[1]] = y
        return loc

    def _reset_display(self):
        dims = set([0,1,2])
        dims.remove(self._cut_dim)
        dims = sorted(list(dims))

        self._dims = dims
        self._screen = pygame.display.set_mode((int(self.shape[dims[0]] * self._scr_scale),
                int(self.shape[dims[1]] * self._scr_scale)), pygame.RESIZABLE)

        # For compatibility with other functions for 2D.
        self.lat_w = self.shape[dims[0]]
        self.lat_h = self.shape[dims[1]]

    def _process_misc_event(self, event):
        if event.type == pygame.KEYDOWN:
            # Select the axis normal to the cutplane.
            if event.key == pygame.K_x:
                self._cut_dim = 0
                self._reset_display()
            elif event.key == pygame.K_y:
                self._cut_dim = 1
                self._reset_display()
            elif event.key == pygame.K_z:
                self._cut_dim = 2
                self._reset_display()
            # Move the cutplane along the selected axis.
            elif event.key == pygame.K_QUOTE:
                if self._cut_pos[self._cut_dim] < self.shape[self._cut_dim]-1:
                    self._cut_pos[self._cut_dim] += 1
            elif event.key == pygame.K_SEMICOLON:
                if self._cut_pos[self._cut_dim] > 0:
                    self._cut_pos[self._cut_dim] -= 1

    def _visualize(self, tx, ty, vismode):
        ret = Fluid2DVis._visualize(self, tx, ty, vismode)
        dim_names = ('X', 'Y', 'Z')
        ret.append('cut {0} @ {1}'.format(dim_names[self._cut_dim], self._cut_pos[self._cut_dim]))
        return ret

    def _draw_tracers(self, tx, ty, sw, sh, width, height):
        pass

    def _draw_wall(self, event):
        x, y, z = self._2d_to_3d_loc(*self._get_loc(event))
        self.sim.geo.set_geo((x, y, z),
                self._draw_type == 1 and geo.LBMGeo.NODE_WALL or geo.LBMGeo.NODE_FLUID,
                update=True)


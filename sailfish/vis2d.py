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

def _cmap_hsv(drw):
    drw = drw.reshape((drw.shape[0], drw.shape[1], 1)) * numpy.float32([1.0, 1.0, 1.0])
    drw[:,:,2] = 1.0
    drw[:,:,1] = 1.0
    drw = hsv_to_rgb(drw) * 255.0
    return drw.astype(numpy.uint8)

def _cmap_std(drw):
    return (drw.reshape((drw.shape[0], drw.shape[1], 1)) * 255.0).astype(numpy.uint8) * numpy.uint8([1,1,0])

def _cmap_2col(drw):
    drw = ((drw*(drw>0).astype(int)).reshape((drw.shape[0], drw.shape[1], 1)) * numpy.uint8([255, 0, 0])
        - ( drw*(drw<0).astype(int)).reshape((drw.shape[0], drw.shape[1], 1)) * numpy.uint8([0, 0, 255]))
    drw[drw>255] = 255
    drw[drw<-255] = -255
    return drw.astype(numpy.uint8)

def _cmap_rgb1(drw):
    """Default color palette from gnuplot."""
    r = numpy.sqrt(drw)
    g = numpy.power(drw, 3)
    b = numpy.sin(drw * math.pi)

    return (numpy.dstack([r,g,b]) * 250.0).astype(numpy.uint8)

def _cmap_bin_red_blue(a, b):
    """Two fields, mapped to the red and blue components, respectively."""
    g = a.copy()
    g[:] = 0.0
    return (numpy.dstack([a,g,b]) * 255.0).astype(numpy.uint8)

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

cmaps = {
    1: {
        'std': _cmap_std,
        'rgb1': _cmap_rgb1,
        'hsv': _cmap_hsv,
        '2col': _cmap_2col,
        },
    2: {
        'rb': _cmap_bin_red_blue,
        }
    }

class Fluid2DVis(object):

    _color_unused = (128, 128, 128)
    _color_wall = (255, 255, 255)

    VIS_LINEAR = 0
    VIS_FLUCTUATION = 1
    VIS_TYPES = [VIS_LINEAR, VIS_FLUCTUATION]

    def __init__(self, sim, width, height, depth, lat_nx, lat_ny, scale):
        # If the size of the window has not been explicitly defined, automatically adjust it
        # based on the size of the grid,
        if width == 0:
            width = int(lat_nx * scale)

        if height == 0:
            height = int(lat_ny * scale)

        self.depth = depth
        self._visfield = 0
        self._vistype = self.VIS_LINEAR
        self._cmap = [None, 'std', 'rb']
        self._cmap_scale_lock = False
        self._convolve = False
        self._font = pygame.font.SysFont('Liberation Mono', 14)
        self.set_mode(width, height)
        self.lat_nx = lat_nx
        self.lat_ny = lat_ny

        self._show_info = True
        self._tracers = False
        self._velocity = False
        self._drawing = False
        self._paused = False
        self._draw_type = 1
        self.sim = sim
        self._reset()

        pygame.key.set_repeat(100,50)
        pygame.display.set_caption('Sailfish v0.1-alpha1')

    def set_mode(self, width, height):
        if self.depth != 0:
            self._screen = pygame.display.set_mode((width, height),
                    pygame.RESIZABLE, self.depth)
        else:
            self._screen = pygame.display.set_mode((width, height),
                    pygame.RESIZABLE)

    def _reset(self):
        self._cmap_scale = [1.0] * self.sim.num_fields

    @property
    def vx(self):
        return self.sim.vx

    @property
    def vy(self):
        return self.sim.vy

    @property
    def field(self):
        return self.sim.vis_fields[self._visfield]

    @property
    def geo_map(self):
        return self.sim.geo.map

    def get_field_vals(self, field):
        v = []
        for f in field.vals:
            v.append(f())
        return v

    def _visualize(self, tx, ty):
        height, width = self.lat_ny, self.lat_nx
        srf = pygame.Surface((width, height))
        ret = []

        dec_map = self.sim.geo._decode_node_type(self.geo_map)
        wall_map = (dec_map == self.sim.geo.NODE_WALL)
        unused_map = (dec_map == self.sim.geo.NODE_UNUSED)

        field = self.field
        ret.append('%s' % field.name)
        fs = []

        for i, fv in enumerate(self.get_field_vals(field)):
            fv = numpy.ma.array(fv, mask=(numpy.logical_or(wall_map, unused_map)))

            if self._vistype == self.VIS_LINEAR:

                # Scale the field eithr manually, or automatically.
                if self._cmap_scale_lock:
                    a = self._cmap_scale[self._visfield]
                    if field.negative:
                        rng = (-a, a)
                    else:
                        rng = (0.0, a)
                else:
                    if field.ranges is not None:
                        rng = field.ranges[i]
                    else:
                        rng = (numpy.min(fv), numpy.max(fv))

                # If negative values are allowed, map the field to
                # [-1;1], otherwise map it to [0;1]
                if field.negative:
                    fv[fv < 0] /= -rng[0]
                    fv[fv > 0] /= rng[1]
                    fv[fv > 1.0] = 1.0
                    fv[fv < -1.0] = -1.0
                else:
                    fv = (fv - rng[0]) / (rng[1] - rng[0])
                    fv[fv > 1.0] = 1.0

                fs.append(fv)

            elif self._vistype == self.VIS_FLUCTUATION:
                max_ = numpy.max(fv)
                min_ = numpy.min(fv)
                avg_ = numpy.average(fv)
                fs.append((fv - avg_) / (max_ - min_))

        if self._convolve:
            g = gauss_kernel(2, sizey=2)
            fs = map(lambda x: signal.convolve(x, g, mode='same'), fs)

        srf2 = self._draw_field(fs, srf, wall_map, unused_map, width, height)
        pygame.transform.scale(srf2, self._screen.get_size(), self._screen)
        sw, sh = self._screen.get_size()

        # Draw the velocity field
        if self._velocity:
            maxv = numpy.max(numpy.sqrt(numpy.square(self.vx) + numpy.square(self.vy)))
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

    def _draw_field(self, fields, srf, wall_map, unused_map, width, height):
        fv = []

        # Rotate the field to the correct position.
        for field in fields:
            fv.append(numpy.rot90(field.astype(numpy.float32), 3))

        a = pygame.surfarray.pixels3d(srf)

        wall_map = numpy.rot90(wall_map, 3)
        unused_map = numpy.rot90(unused_map, 3)

        # Draw the walls.
        a[wall_map] = self._color_wall
        a[unused_map] = self._color_unused

        n = len(fields)
        fluid_map = numpy.logical_not(numpy.logical_or(wall_map, unused_map))
        field = cmaps[n][self._cmap[n]](*fv)
        a[fluid_map] = field[fluid_map]

        # Unlock the surface and put the picture on screen.
        del a
        return srf

    def _draw_tracers(self, tx, ty, sw, sh, width, height):
        # Draw the tracer particles
        if self._tracers:
            for x, y in zip(tx, ty):
                pygame.draw.circle(self._screen, (0, 255, 255), (int(x * sw / width), int(sh - y * sh / height)), 2)

    def _get_loc(self, event):
        x = event.pos[0] * self.lat_nx / self._screen.get_width()
        y = self.lat_ny-1 - (event.pos[1] * self.lat_ny / self._screen.get_height())
        return min(max(x, 0), self.lat_nx-1), min(max(y, 0), self.lat_ny-1)

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
                self.set_mode(*event.size)
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
                if event.key == pygame.K_MINUS:
                    self._visfield -= 1
                    self._visfield %= self.sim.num_fields
                elif event.key == pygame.K_EQUALS:
                    self._visfield += 1
                    self._visfield %= self.sim.num_fields
                elif event.key == pygame.K_LEFTBRACKET:
                    n = len(self.field.vals)
                    idx = cmaps[n].keys().index(self._cmap[n]) - 1
                    idx %= len(cmaps[n].keys())
                    self._cmap[n] = cmaps[n].keys()[idx]
                elif event.key == pygame.K_RIGHTBRACKET:
                    n = len(self.field.vals)
                    idx = cmaps[n].keys().index(self._cmap[n]) + 1
                    idx %= len(cmaps[n].keys())
                    self._cmap[n] = cmaps[n].keys()[idx]
                elif event.key == pygame.K_m:
                    self._cmap_scale_lock = not self._cmap_scale_lock
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
                    self._cmap_scale[self._visfield] = self._cmap_scale[self._visfield] / 1.1
                elif event.key == pygame.K_PERIOD:
                    self._cmap_scale[self._visfield] = self._cmap_scale[self._visfield] * 1.1

            self._process_misc_event(event)

    def _update_display(self, i, avg_mlups, mlups):
        ret = self._visualize(self.sim.tracer_x, self.sim.tracer_y)

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

    def main(self):
        self._reset()
        t_prev = time.time()
        avg_mlups = 0.0
        mlups = 0.0

        while 1:
            self._process_events()
            i = self.sim.iter_

            if self._paused:
                self._update_display(i, avg_mlups, mlups)
                pygame.time.wait(50)
                continue

            self.sim.sim_step(self._tracers)

            if i % self.sim.options.every == 0 and i:
                avg_mlups, mlups = self.sim.get_mlups(time.time() - t_prev)
                self._update_display(i, avg_mlups, mlups)
                t_prev = time.time()

class Fluid3DVisCutplane(Fluid2DVis):

    def __init__(self, sim, shape, depth, scr_scale):
        Fluid2DVis.__init__(self, sim, int(shape[0] * scr_scale), int(shape[1] *
            scr_scale), depth, shape[0], shape[1], scr_scale)
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

    def get_field_vals(self, field):
        v = []
        for f in field.vals:
            a = f()[self._slice_args]
            v.append(a)
        return v

    @property
    def vx(self):
        return self.sim.velocity[self._dims[0]][self._slice_args]

    @property
    def vy(self):
        return self.sim.velocity[self._dims[1]][self._slice_args]

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
        self.set_mode(int(self.shape[dims[0]] * self._scr_scale),
                int(self.shape[dims[1]] * self._scr_scale))

        # For compatibility with other functions for 2D.
        self.lat_nx = self.shape[dims[0]]
        self.lat_ny = self.shape[dims[1]]

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

    def _visualize(self, tx, ty):
        ret = Fluid2DVis._visualize(self, tx, ty)
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


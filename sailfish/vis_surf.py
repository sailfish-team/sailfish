import math
import os
import sys
import time

import numpy
import pygame

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import numpymodule

from sailfish import geo
from sailfish import sym

pygame.init()
pygame.surfarray.use_arraytype('numpy')

def _GL_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, 1.0*width/height, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

def _GL_init():
    glShadeModel(GL_SMOOTH)
    glClearColor(0.0, 0.0, 0.0, 0.0)
    glClearDepth(1.0)
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LEQUAL)
    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)

class FluidSurfaceVis(object):
    display_flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
    display_depth = 24

    def __init__(self, sim, width, height, lat_nx, lat_ny):
        self._font = pygame.font.SysFont('Liberation Mono', 14)
        self._screen = pygame.display.set_mode((width, height), self.display_flags,
                self.display_depth)

        _GL_resize(width, height)
        _GL_init()

        self.lat_nx = lat_nx
        self.lat_ny = lat_ny
        self.sim = sim

        m = min(lat_nx, lat_ny)
        mx = lat_nx / m
        my = lat_ny / m

        self.mesh_x, self.mesh_y = numpy.mgrid[
                -my:my:complex(0, lat_ny),
                -mx:mx:complex(0, lat_nx)]

        self.mesh_x = self.mesh_x.astype(numpy.float32)
        self.mesh_y = self.mesh_y.astype(numpy.float32)
        self.mesh_n = (self.lat_nx-1) * (self.lat_ny-1) * 4

        self._polygon_mode = 0
        self._show_info = True
        self._velocity = False
        self._paused = False
        self._reset()

        pygame.key.set_repeat(100,50)

        from lbm import __version__ as version
        pygame.display.set_caption('Sailfish v%s' % version)

    def _reset(self):
        self._maxv = 0.000001

    @property
    def velocity_norm(self):
        return self.sim.geo.mask_array_by_fluid(
                numpy.sqrt(numpy.add(numpy.square(self.vx),
                    numpy.square(self.vy))))

    @property
    def vx(self):
        return self.sim.vx

    @property
    def vy(self):
        return self.sim.vy

    @property
    def height(self):
        return self.sim.rho

    @property
    def geo_map(self):
        return self.sim.geo.map

    def _gl_arrays(self):
        mesh_z = self.height.astype(numpy.float32)

        a = numpy.dstack((self.mesh_x, self.mesh_y, mesh_z))
        b = numpy.roll(a, -1, 0)
        c = numpy.roll(b, -1, 1)
        d = numpy.roll(a, -1, 1)

        a = numpy.delete(a, -1, 0)
        a = numpy.delete(a, -1, 1)
        b = numpy.delete(b, -1, 0)
        b = numpy.delete(b, -1, 1)
        c = numpy.delete(c, -1, 0)
        c = numpy.delete(c, -1, 1)
        d = numpy.delete(d, -1, 0)
        d = numpy.delete(d, -1, 1)

        res = numpy.dstack((a, b, c, d))
        vtx = numpy.ravel(res)

        col = numpy.zeros((3, self.mesh_n), dtype=numpy.float32)
        min_ = numpy.min(mesh_z)
        max_ = numpy.max(mesh_z)

        min_ = 0.8
        max_ = 1.2

        tmp = vtx[2::3]
        col[1,:] = (tmp[:]-min_)/(max_-min_)
        col = numpy.ravel(numpy.transpose(col))

        vtx.shape = (self.mesh_n, 3)
        col.shape = (self.mesh_n, 3)
        return vtx, col

    def _visualize(self):
        height, width = self.vx.shape
        srf = pygame.Surface((width, height))

        ret = []

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(0.0, -0.5, -4.0)
        glRotatef(-60.0, 1.0, 0.0, 0.0)
        glRotatef(90.0, 0.0, 0.0, 1.0)

        glEnableClientState(GL_COLOR_ARRAY)
        glEnableClientState(GL_VERTEX_ARRAY)

        vertices, colors = self._gl_arrays()
        glVertexPointerf(vertices)
        glColorPointerf(colors)
        glDrawArrays(GL_QUADS, 0, len(vertices))

        return ret

    def _process_events(self):
        while True:
            event = pygame.event.poll()
            if event.type == pygame.NOEVENT:
                break
            elif event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                self._screen = pygame.display.set_mode(event.size,
                        self.display_flags, self.display_depth)
                _GL_resize(*event.size)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_v:
                    self._velocity = not self._velocity
                elif event.key == pygame.K_p:
                    self._paused = not self._paused
                    if self._paused:
                        print 'Simulation paused @ iter = %d.' % self.sim.iter_
                elif event.key == pygame.K_q:
                    sys.exit()
                elif event.key == pygame.K_r:
                    self._reset()
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
                    self._maxv /= 1.1
                elif event.key == pygame.K_PERIOD:
                    self._maxv *= 1.1
                elif event.key == pygame.K_w:
                    modes = [GL_FILL, GL_LINE, GL_POINT]
                    self._polygon_mode += 1
                    self._polygon_mode %= 3
                    glPolygonMode(GL_FRONT_AND_BACK, modes[self._polygon_mode]);

    def main(self):
        t_prev = time.time()
        avg_mlups = 0.0

        while 1:
            self._process_events()

            if self._paused:
                continue

            i = self.sim.iter_
            self.sim.sim_step(False)

            if i % self.sim.options.every == 0 and i:
                avg_mlups, mlups = self.sim.get_mlups(time.time() - t_prev)
                self._visualize()
                pygame.display.flip()
                t_prev = time.time()

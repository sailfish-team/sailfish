import math
import os
import sys
import time

import numpy
import pygame

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.arrays import numpymodule

has_ropengl = True
try:
    import ropengl
except ImportError:
    has_ropengl = False

from sailfish import geo
from sailfish import sym
from sailfish import vis

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

def _vis_rgb(v):
    r = numpy.sqrt(v)
    g = numpy.power(v, 3)
    b = numpy.sin(v * math.pi)
    return numpy.vstack((r,g,b))

def _vis_scl(v):
    col = numpy.zeros((3, len(v)), dtype=numpy.float32)
    col[1,:] = v
    return col

colormaps = {
    'rgb': _vis_rgb,
    'scl': _vis_scl,
    }

class FluidSurfaceVis(vis.FluidVis):

    def __init__(self, sim, width, height, depth, lat_nx, lat_ny):
        super(FluidSurfaceVis, self).__init__()
        self._font = pygame.font.SysFont('Liberation Mono', 14)
        self.depth = depth
        self.set_mode(width, height)

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

        self._zoom = -4.0
        self._angle_x = -60.0
        self._angle_z = 90.0

        self._polygon_mode = 0
        self._show_info = True
        self._velocity = False
        self._paused = False
        self._reset()

        self._colormap = colormaps.keys()[0]
        self._minh = -0.1
        self._maxh = 0.1

        pygame.key.set_repeat(100,50)

        from lbm import __version__ as version
        pygame.display.set_caption('Sailfish v%s' % version)

    def _reset(self):
        self._minh = 10000000.0
        self._maxh = -10000000.0

    def set_mode(self, width, height):
        display_flags = pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE
        if self.depth > 0:
            self._screen = pygame.display.set_mode((width, height), display_flags,
                self.depth)
        else:
            self._screen = pygame.display.set_mode((width, height), display_flags)

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
        mesh_z = self.height.astype(numpy.float32) - 1.0

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

        min_ = numpy.min(mesh_z)
        max_ = numpy.max(mesh_z)

        self._minh = min(self._minh, numpy.min(mesh_z))
        self._maxh = max(self._maxh, numpy.max(mesh_z))

        col = colormaps[self._colormap]((vtx[2::3]-self._minh)/(self._maxh-self._minh))
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

        glTranslatef(0.0, 0.0, self._zoom)
        glRotatef(self._angle_x, 1.0, 0.0, 0.0)
        glRotatef(self._angle_z, 0.0, 0.0, 1.0)

        vertices, colors = self._gl_arrays()

        if not has_ropengl:
            glEnableClientState(GL_COLOR_ARRAY)
            glEnableClientState(GL_VERTEX_ARRAY)

            glVertexPointerf(vertices)
            glColorPointerf(colors)
            glDrawArrays(GL_QUADS, 0, len(vertices))
        else:
            ropengl.glVertexColorArray(vertices, colors)

        return ret

    def _process_events(self):
        while True:
            event = pygame.event.poll()
            if event.type == pygame.NOEVENT:
                break
            elif event.type == pygame.QUIT:
                sys.exit()
            elif event.type == pygame.VIDEORESIZE:
                self.set_mode(*event.size)
                _GL_resize(*event.size)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 5:
                    self._zoom -= 0.05
                elif event.button == 4:
                    self._zoom += 0.05
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
                elif event.key == pygame.K_UP:
                    if event.mod & pygame.KMOD_SHIFT:
                        self._angle_x += 3.0
                    else:
                        self._angle_x += 1.0
                elif event.key == pygame.K_DOWN:
                    if event.mod & pygame.KMOD_SHIFT:
                        self._angle_x -= 3.0
                    else:
                        self._angle_x -= 1.0
                elif event.key == pygame.K_LEFT:
                    if event.mod & pygame.KMOD_SHIFT:
                        self._angle_z -= 3.0
                    else:
                        self._angle_z -= 1.0
                elif event.key == pygame.K_RIGHT:
                    if event.mod & pygame.KMOD_SHIFT:
                        self._angle_z += 3.0
                    else:
                        self._angle_z += 1.0
                elif event.key == pygame.K_c:
                    idx = colormaps.keys().index(self._colormap) + 1
                    idx %= len(colormaps.keys())
                    self._colormap = colormaps.keys()[idx]

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

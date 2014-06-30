"""pygame visualization backend."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import math
import os

import numpy as np
import pygame

from sailfish import node_type, util, vis

def _font_name():
    import platform
    if platform.system() == 'Windows':
        return 'Courier New'
    else:
        return 'Liberation Mono'

def _hsv_to_rgb(a):
    t = a[:,:,0]*6.0
    i = t.astype(np.uint8)
    f = t - np.floor(t)

    v = a[:,:,2]

    o = np.ones_like(a[:,:,0])
    p = v * (o - a[:,:,1])
    q = v * (o - a[:,:,1]*f)
    t = v * (o - a[:,:,1]*(o - f))

    i = np.mod(i, 6)
    sh = i.shape
    i = i.reshape(sh[0], sh[1], 1) * np.uint8([1,1,1])

    choices = [np.dstack((v, t, p)),
               np.dstack((q, v, p)),
               np.dstack((p, v, t)),
               np.dstack((p, q, v)),
               np.dstack((t, p, v)),
               np.dstack((v, p, q))]

    return np.choose(i, choices)

def symmetric(f):
    f.symmetric = True
    return f

def _cmap_hsv(drw):
    drw = drw.reshape((drw.shape[0], drw.shape[1], 1)) * np.float32([1.0, 1.0, 1.0])
    drw[:,:,2] = 1.0
    drw[:,:,1] = 1.0
    drw = _hsv_to_rgb(drw) * 255.0
    return drw.astype(np.uint8)

def _cmap_std(drw):
    return (drw.reshape((drw.shape[0], drw.shape[1], 1)) * 255.0).astype(np.uint8) * np.uint8([1,1,0])

@symmetric
def _cmap_2col(drw):
    drw = ((drw*(drw>0).astype(int)).reshape((drw.shape[0], drw.shape[1], 1)) * np.uint8([255, 0, 0])
        - ( drw*(drw<0).astype(int)).reshape((drw.shape[0], drw.shape[1], 1)) * np.uint8([0, 0, 255]))
    drw[drw>255] = 255
    drw[drw<-255] = -255
    return drw.astype(np.uint8)

def _cmap_rgb1(drw):
    """Default color palette from gnuplot."""
    r = np.sqrt(drw)
    g = np.power(drw, 3)
    b = np.sin(drw * math.pi)

    return (np.dstack([r,g,b]) * 250.0).astype(np.uint8)

def _cmap_bin_red_blue(a, b):
    """Two fields, mapped to the red and blue components, respectively."""
    g = a.copy()
    g[:] = 0.0
    return (np.dstack([a,g,b]) * 255.0).astype(np.uint8)

def _gauss_kernel(size, sizey=None):
    """Return a normalized 2D gauss kernel array for convolutions"""
    size = int(size)
    if not sizey:
        sizey = size
    else:
        sizey = int(sizey)
    x, y = np.mgrid[-size:size+1, -sizey:sizey+1]
    g = np.exp(-(x**2/float(size) + y**2/float(sizey)))
    return g / g.sum()

def _emboss_field(fv, a):
    # Based on the code posted on
    # http://stackoverflow.com/questions/2034037/image-embossing-in-python-with-pil-adding-depth-azimuth-etc
    azi = np.pi/8.
    ele = np.pi/16.
    dep = 0.81
    grad_x, grad_y = np.gradient(255 * fv * dep)

    gd = np.cos(ele) # length of projection of ray on ground plane
    dx = gd * np.cos(azi)
    dy = gd * np.sin(azi)
    dz = np.sin(ele)
    # finding the unit normal vectors for the image
    len_ = np.sqrt(np.square(grad_x) + np.square(grad_y) + 1.)
    a2 = 255 * (dx*grad_x + dy*grad_y + dz) / len_
    a2 = a2.clip(0,255).astype('int')
    w = 0.5

    a[:,:,0] = (w*a[:,:,0] + (1-w)*a2)
    a[:,:,1] = (w*a[:,:,1] + (1-w)*a2)
    a[:,:,2] = (w*a[:,:,2] + (1-w)*a2)

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

# TODO(michalj): Restore the option to manually impart velocity on the fluid.
# TODO(michalj): Restore support for drawing walls.
# TODO(michalj): Restore support for tracers.
class Fluid2DVis(vis.FluidVis):
    _color_unused = (128, 128, 128)
    _color_wall = (255, 255, 255)

    VIS_LINEAR = 0
    VIS_FLUCTUATION = 1
    VIS_TYPES = [VIS_LINEAR, VIS_FLUCTUATION]

    name = 'pygame'
    dims = [2]

    @classmethod
    def add_options(cls, group):
        group.add_argument('--scr_w', help='screen width', type=int, default=0)
        group.add_argument('--scr_h', help='screen height', type=int, default=0)
        group.add_argument('--scr_scale', help='screen scale', type=float, default=3.0)
        group.add_argument('--scr_depth', help='screen color depth', type=int,
                           default=24)

    def __init__(self, config, subdomains, quit_event, sim_quit_event,
                 vis_config, geo_queues=None):
        pygame.init()
        pygame.font.init()
        pygame.surfarray.use_arraytype('numpy')

        super(Fluid2DVis, self).__init__()
        self.config = config
        self.config.logger.info("Initializating pygame 2D vis. engine.")
        self._quit_event = quit_event
        self._sim_quit_event = sim_quit_event
        self._vis_config = vis_config
        self._geo_queues = geo_queues
        self._subdomains = subdomains

        # This will default the visualization to the first component of the
        # velocity field.
        self._vis_config.field = 1

        # Whether a single subdomain is visualized.
        self._single_subdomain = True

        self._font_color = (0, 255, 0)
        self._show_info = True
        self._show_walls = True
        self._velocity = False
        self._draw_type = 1
        self._vistype = self.VIS_LINEAR
        self._cmap = [None, 'std', 'rb']
        self._font = pygame.font.SysFont(_font_name(), 14)
        self._drawing = False

        self.resize()

        pygame.key.set_repeat(100,50)
        pygame.display.set_caption('Sailfish')

    def set_mode(self, width, height):
        if self.config.scr_depth != 0:
            self._screen = pygame.display.set_mode((width, height),
                    pygame.RESIZABLE, self.config.scr_depth)
        else:
            self._screen = pygame.display.set_mode((width, height),
                    pygame.RESIZABLE)

    def resize(self):
        width, height = self.size
        width = int(width * self.config.scr_scale)
        height = int(height * self.config.scr_scale)
        self.set_mode(width, height)

    @property
    def size(self):
        return self._subdomains[self._vis_config.subdomain].size

    def get_field_vals(self, field):
        v = []
        for f in field.vals:
            v.append(f())
        return v

    def _visualize(self):
        width, height = self.size
        srf = pygame.Surface((width, height))
        subdomain = self._subdomains[self._vis_config.subdomain]
        ret = ['subdomain {0}'.format(subdomain.id),
               self._vis_config.field_name]

        srf2, f_min, f_max = self._draw_field(srf, width, height, subdomain)
        pygame.transform.scale(srf2, self._screen.get_size(), self._screen)

        ret.append('[%.2f:%.2f]' % (f_min, f_max))

        # TODO(michalj): Add support for vector fields.
        # TODO(michalj): Add support for tracer particles.
        return ret

    def _get_geo_map(self, width, height, subdomain):
        geo_map = np.zeros((height, width), dtype=np.uint8)
        geo_map.reshape(width * height)[:] = subdomain.vis_geo_buffer[:]
        return geo_map

    def _draw_geometry(self, tg_buffer, width, height, subdomain):
        geo_map = np.rot90(self._get_geo_map(width, height, subdomain), 3)
        dry_types = geo_map.dtype.type(node_type.get_dry_node_type_ids())
        tg_buffer[util.in_anyd_fast(geo_map, dry_types)] = self._color_wall

    def _get_field(self, width, height, subdomain):
        # FIXME(michalj): This is horribly inefficient.  We should only recreate
        # the array if the data has been updated.
        tmp = np.zeros((height, width), dtype=np.float32)
        tmp.reshape(width * height)[:] = subdomain.vis_buffer[:]
        return tmp

    def _draw_field(self, srf, width, height, subdomain):
        a = pygame.surfarray.pixels3d(srf)
        tmp = self._get_field(width, height, subdomain)
        v_max = np.nanmax(tmp)
        v_min = np.nanmin(tmp)

        cmap = cmaps[1][self._cmap[1]]
        if hasattr(cmap, 'symmetric'):
            v_absmax = max(v_max, abs(v_min))
            tmp[:] /= v_absmax
        else:
            tmp[:] = (tmp - v_min) / (v_max - v_min)
            tmp[tmp > 1.0] = 1.0
            tmp = np.abs(tmp)

        tmp = np.rot90(tmp, 3)

        # TODO(michalj): Add support for multi-component fields.
        vis_field = cmaps[1][self._cmap[1]](tmp)
        a[:] = vis_field[:]

        self._draw_geometry(a, width, height, subdomain)

        # Unlock the surface and put the picture on screen.
        del a
        return srf, v_min, v_max

    def _get_loc(self, event, subdomain):
        x = event.pos[0] * subdomain.nx / self._screen.get_width()
        y = subdomain.ny - 1 - (event.pos[1] * subdomain.ny / self._screen.get_height())
        return min(max(x, 0), subdomain.nx-1), min(max(y, 0), subdomain.ny-1)

    def _process_misc_event(self, event):
        """A function to make it possible to process additional events in subclasses."""
        pass

    def _draw_wall(self, event):
        subdomain = self._subdomains[self._vis_config.subdomain]
        x, y = self._get_loc(event, subdomain)
        self._geo_queues[self._vis_config.subdomain].put(((x, y), self._draw_type == 1))

    def _process_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._sim_quit_event.set()
            elif event.type == pygame.VIDEORESIZE:
                self.set_mode(*event.size)
            elif event.type == pygame.MOUSEBUTTONUP:
                self._draw_type = event.button
                self._draw_wall(event)
                self._drawing = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._draw_type = event.button
                self._drawing = True
                self._draw_wall(event)
            elif event.type == pygame.MOUSEMOTION:
                if self._drawing:
                    self._draw_wall(event)
            elif event.type == pygame.KEYDOWN:
                # Previous field.
                if event.key == pygame.K_MINUS:
                    new_field = self._vis_config.field - 1
                    new_field %= self._vis_config.fields
                    self._vis_config.field = new_field
                # Next field.
                elif event.key == pygame.K_EQUALS:
                    new_field = self._vis_config.field + 1
                    new_field %= self._vis_config.fields
                    self._vis_config.field = new_field
                # Previous subdomain.
                elif event.key == pygame.K_j:
                    new_subdomain = self._vis_config.subdomain - 1
                    new_subdomain %= len(self._subdomains)
                    self._vis_config.subdomain = new_subdomain
                    self.resize()
                # Next subdomain.
                elif event.key == pygame.K_k:
                    new_subdomain = self._vis_config.subdomain + 1
                    new_subdomain %= len(self._subdomains)
                    self._vis_config.subdomain = new_subdomain
                    self.resize()
                elif event.key == pygame.K_LEFTBRACKET:
                    #n = len(self.field.vals)
                    n = 1
                    idx = cmaps[n].keys().index(self._cmap[n]) - 1
                    idx %= len(cmaps[n].keys())
                    self._cmap[n] = cmaps[n].keys()[idx]
                elif event.key == pygame.K_RIGHTBRACKET:
                    #n = len(self.field.vals)
                    n = 1
                    idx = cmaps[n].keys().index(self._cmap[n]) + 1
                    idx %= len(cmaps[n].keys())
                    self._cmap[n] = cmaps[n].keys()[idx]
#                elif event.key == pygame.K_v:
#                    self._velocity = not self._velocity
                elif event.key == pygame.K_q:
                    self._sim_quit_event.set()
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

            self._process_misc_event(event)

    def _update_display(self):
        curr_iter = self._vis_config.iteration
        if curr_iter < 0:
            self._screen.blit(
                    self._font.render('Waiting for simulation startup...',
                    True, self._font_color), (12, 12))
        else:
            ret = self._visualize()

            if self._show_info:
                self._screen.blit(self._font.render('itr: %dk' % (curr_iter /
                    1000), True, (0, 255, 0)), (12, 12))

                y = 28
                for info in ret:
                    tmp = self._font.render(info, True, (0, 255, 0))
                    self._screen.blit(tmp, (12, y))
                    y += 14

                for info in self.display_infos:
                    tmp = self._font.render(info(), True, (0, 255, 0))
                    self._screen.blit(tmp, (12, y))
                    y += 14

        pygame.display.flip()


    def run(self):
        while True:
            self._process_events()
            self._update_display()
#            pygame.time.wait(50)

            if self._quit_event.is_set():
                break

engine=Fluid2DVis

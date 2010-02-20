#!/usr/bin/python -u

import sys
import os
import numpy
import glob
import pygame

from sailfish import lbm
from sailfish import vis2d
from sailfish import geo
from scipy import signal

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

from lbm_poiseuille import LBMGeoPoiseuille

def load_image(name, colorkey=None):
    fullname = os.path.join('', name)
    try:
        image = pygame.image.load(fullname)
    except pygame.error, message:
        print 'Cannot load image:', fullname
        raise SystemExit, message
    image = image.convert()
    if colorkey is not None:
        if colorkey is -1:
            colorkey = image.get_at((0,0))
        image.set_colorkey(colorkey, RLEACCEL)
    return image, image.get_rect()

class Fluid2DVisPresentation(vis2d.Fluid2DVis):
    """Fluid2DVis with presentation tricks."""

    def __init__(self, *args):
        vis2d.Fluid2DVis.__init__(self, *args)
        self.im_number = 0
        self.im = load_image('slides/sailfish-0.png')
        self.im_maxnumber = len(glob.glob('slides/sailfish-*.png'))
        self._set_wall_from_image()
        self._show_info = False
        pygame.display.set_caption('Sailfish v0.1-alpha1 (presentation mode)')

    def _reset(self):
        self._maxv = 0.04
        self._vscale = 0.005
        self._vismode = 4

    def _draw_field(self, field, srf, b, unused_map, vismode, width, height):
        # Rotate the field to the correct position.
        field = numpy.rot90(field.astype(numpy.float32), 3)
        a = pygame.surfarray.pixels3d(srf)
        b = numpy.rot90(b, 3)

        # Draw the data field for all sites which are not marked as a wall.
        b = numpy.logical_not(b)
        field = vis2d.vis_map[vismode](field, width, height)
        a[b] = field[b]

        # Unlock the surface and put the picture on screen.
        del a

        srf2 = pygame.transform.scale(srf, self._screen.get_size())
        srf2.set_alpha(256)
        #        self._screen.blit(srf2, (0,0))

        im2 = pygame.transform.scale(self.im[0], self._screen.get_size())
        im2.set_colorkey(0)
        srf2.blit(im2, (0,0))
        return srf2

    def _process_misc_event(self,event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_w:
                self._set_wall_from_image()
            elif event.key == pygame.K_f:
                pygame.display.toggle_fullscreen()
            elif event.key == pygame.K_DOWN or event.key == pygame.K_UP:
                if event.key == pygame.K_DOWN:
                    self.im_number -= 1
                else:
                    self.im_number += 1

                if self.im_number < 0:
                    self.im_number = (self.im_maxnumber-1)
                if self.im_number > (self.im_maxnumber-1):
                    self.im_number = 0

                self.im = load_image('slides/sailfish-%d.png' % self.im_number)
                self._reset()
                self.sim.geo.reset()
                self._set_wall_from_image()

    def _set_wall_from_image(self):
        """Set walls based on data from a png file."""
        srf2 = pygame.transform.scale(self.im[0], (self.lat_nx, self.lat_ny))
        a = pygame.surfarray.pixels3d(srf2)
        a = a.sum(axis=-1)>0
        self.sim.geo.set_geo_from_bool_array(numpy.flipud(a.transpose()), update=True)

class LPresSim(lbm.FluidLBMSim):
    filename = 'Sailfish_Presentation'

    def __init__(self, geo_class, args=sys.argv[1:], defaults=None):
        opts = []
        opts.append(optparse.make_option('--horizontal', dest='horizontal', action='store_true', default=True, help='use horizontal channel'))
        opts.append(optparse.make_option('--stationary', dest='stationary', action='store_true', default=True, help='start with the correct velocity profile in the whole simulation domain'))
        opts.append(optparse.make_option('--drive', dest='drive', type='choice', choices=['force', 'pressure'], default='force'))

        if defaults is not None:
            defaults_ = defaults
        else:
            defaults_ = {'max_iters': 500000, 'visc': 0.1, 'lat_nx': 320,
                    'lat_ny': 240, 'verbose': True, 'vismode': '2col',
                    'every': 400, 'model': 'mrt','visc':0.001,'scr_scale':1.0}

        lbm.FluidLBMSim.__init__(self, geo_class, options=opts, args=args, defaults=defaults_)

    def _init_vis_2d(self):
        self.vis = Fluid2DVisPresentation(self, self.options.scr_w, self.options.scr_h,
                                          self.options.scr_depth, self.options.lat_nx, self.options.lat_ny,
                                          self.options.scr_scale)

if __name__ == '__main__':
    sim = LPresSim(LBMGeoPoiseuille)
    sim.run()

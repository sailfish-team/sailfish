#!/usr/bin/python -u

import os
import sys
import numpy
import math
import matplotlib
import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from sailfish import geo
from sailfish import sym

MAX_ITERS = 10000
POINTS = 30

defaults = {
        'stationary': True,
        'batch': True,
        'quiet': True,
        'verbose': False,
        'lat_nx': 64,
        'lat_ny': 64,
    }

def run_test(bc, drive, precision, model, grid, name):
    xvec = []
    yvec = []
    basepath = os.path.join('regtest/results', name, grid, model, drive, precision)
    profpath = os.path.join(basepath, 'profiles')

    if not os.path.exists(profpath):
        os.makedirs(profpath)

    f = open(os.path.join(basepath, '%s.dat' % bc), 'w')

    defaults['grid'] = grid
    defaults['model'] = model
    defaults['drive'] = drive
    if drive == 'pressure':
        defaults['bc_pressure'] = bc
    else:
        defaults['bc_wall'] = bc
    defaults['precision'] = precision

    print '* Testing "%s" for visc' % bc,
    i = 0

    for visc in numpy.logspace(-3, -1, num=POINTS):
        print '%f ' % visc,

        iters = int(1000 / visc)
        xvec.append(visc)

        defaults['visc'] = visc
        defaults['max_iters'] = iters

        sim = LTestPoiSim([], defaults)
        sim.run()

        yvec.append(sim.result)

        prof_sim = sim.get_profile()
        prof_th = sim.geo.get_velocity_profile(fluid_only=True)

        print >>f, visc, sim.result

        plt.gca().yaxis.grid(True)
        plt.gca().xaxis.grid(True)
        plt.gca().xaxis.grid(True, which='minor')
        plt.gca().yaxis.grid(True, which='minor')
        plt.gca().set_xbound(0, len(prof_sim)-1)

        f2 = open(os.path.join(profpath, '%s-profile%d.dat' % (bc, i)), 'w')
        for j in range(0, len(prof_sim)):
            print >>f2, prof_sim[j], prof_th[j]
        f2.close()

        plt.clf()
        plt.plot(prof_th - prof_sim, 'bo-')
        plt.title('visc = %f' % xvec[i])
        plt.savefig(os.path.join(profpath, '%s-profile%d.pdf' % (bc, i)), format='pdf')

        plt.clf()
        plt.cla()

        i += 1

    print

    f.close()

    plt.clf()
    plt.cla()

    plt.semilogx(xvec, yvec, 'bo-')
    plt.gca().yaxis.grid(True)
    plt.gca().yaxis.grid(True, which='minor')
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.ylabel('max velocity / theoretical max velocity - 1')
    plt.xlabel('viscosity')
    plt.savefig(os.path.join(basepath, '%s.pdf' % bc), format='pdf')

parser = OptionParser()
parser.add_option('--precision', dest='precision', help='precision (single, double)', type='choice', choices=['single', 'double'], default='single')
parser.add_option('--drive', dest='drive', help='drive', type='choice', choices=['force', 'pressure'], default='force')
parser.add_option('--model', dest='model', help='model', type='choice', choices=['mrt', 'bgk'], default='bgk')
parser.add_option('--grid', dest='grid', help='grid', type='string', default='')
parser.add_option('--dim', dest='dim', help='dimensionality', type='choice', choices=['2','3'], default='2')
parser.add_option('--bc', dest='bc', help='boundary conditions to test (comma separated', type='string', default='')
(options, args) = parser.parse_args()

if options.dim == '2':
    from examples.lbm_poiseuille import LPoiSim, LBMGeoPoiseuille
    name = 'poiseuille'
else:
    from examples.lbm_poiseuille_3d import LPoiSim, LBMGeoPoiseuille
    defaults['along_y'] = True
    defaults['along_z'] = False
    defaults['along_x'] = False
    defaults['lat_nz'] = 64
    name = 'poiseuille3d'

if options.grid:
    grid = options.grid
else:
    if options.dim == '2':
        grid = 'D2Q9'
    else:
        grid = 'D3Q13'

if options.drive == 'force':
    geo_type = geo.LBMGeo.NODE_WALL
else:
    geo_type = geo.LBMGeo.NODE_PRESSURE

if options.bc:
    bcs = filter(lambda x: geo_type in geo.get_bc(x).supported_types, options.bc.split(','))
else:
    bcs = [x.name for x in geo.SUPPORTED_BCS if geo_type in x.supported_types]

class LTestPoiSim(LPoiSim):
    def __init__(self, args, defaults):
        super(LTestPoiSim, self).__init__(LBMGeoPoiseuille, args, defaults)
        self.clear_hooks()
        self.add_iter_hook(self.options.max_iters-1, self.save_output)

    def save_output(self):
#       self.result = (numpy.max(self.vy[16,1:self.geo.lat_nx-1]) / max(self.geo.get_velocity_profile())) - 1.0
        self.res_maxv = numpy.max(self.geo.mask_array_by_fluid(self.vy))
        self.th_maxv = max(self.geo.get_velocity_profile())

        self.result = self.res_maxv / self.th_maxv - 1.0

print 'Running tests for %s, %s' % (options.precision, options.drive)

for bc in bcs:
    run_test(bc, options.drive, options.precision, options.model, grid, name)

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

from examples.lbm_sphere_force_3d import LSphereSim, LBMGeoSphere

MAX_ITERS = 50000

defaults = {
        'batch': True,
        'verbose': True,
        'max_iters': MAX_ITERS,
    }

def run_test(precision, model, grid, name):
    xvec = []
    yvec = []
    yvec2 = []
    basepath = os.path.join('regtest/results', name, grid, model, precision)

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    bc = 'fullbb'

    f = open(os.path.join(basepath, '%s.dat' % bc), 'w')

    defaults['grid'] = grid
    defaults['model'] = model
    defaults['bc_wall'] = bc
    defaults['precision'] = precision

    for re in [1, 10, 50, 100, 200, 300, 400]:
        defaults['re'] = re
        print 'Testing for Re = %d' % re

        sim = LSphereSim(LBMGeoSphere, defaults)
        sim.run()

        dc = math.fsum(sim.coeffs)/len(sim.coeffs)
        dct = sim.drag_theo()

        xvec.append(re)
        yvec.append(dc)
        yvec2.append(dct)

        print >>f, dc, dct

    print

    f.close()

    plt.clf()
    plt.cla()

    plt.loglog(xvec, yvec, 'bo-')
    plt.loglog(xvec, yvec2, 'ro-')
    plt.gca().yaxis.grid(True)
    plt.gca().yaxis.grid(True, which='minor')
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.ylabel('c_d')
    plt.xlabel('Re')
    plt.savefig(os.path.join(basepath, '%s.pdf' % bc), format='pdf')

parser = OptionParser()
parser.add_option('--precision', dest='precision', help='precision (single, double)', type='choice', choices=['single', 'double'], default='single')
parser.add_option('--model', dest='model', help='model', type='choice', choices=['mrt', 'bgk'], default='bgk')
parser.add_option('--grid', dest='grid', help='grid', type='string', default='')
(options, args) = parser.parse_args()

geo_type = geo.LBMGeo3D.NODE_VELOCITY

if options.grid:
    grid = options.grid
else:
    grid = 'D3Q13'

print 'Running tests for %s' % (options.precision)

run_test(options.precision, options.model, grid, 'drag_coeffcient')

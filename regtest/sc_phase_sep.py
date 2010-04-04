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

MAX_ITERS = 100000
POINTS = 26 * 3 + 1

defaults = {
        'batch': True,
        'quiet': True,
        'verbose': False,
        'every': 1000,
    }

def run_test(name, precision):
    xvec = []
    minvec = []
    maxvec = []
    ordvec = []
    basepath = os.path.join('regtest/results', name)

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    f = open(os.path.join(basepath, '%s.dat' % precision), 'w')

    for g in numpy.linspace(3, 5.6, num=POINTS):

        print '%f ' % g,

        xvec.append(g)

        defaults['G'] = g
        defaults['max_iters'] = MAX_ITERS

        sim = SCSim(GeoSC, defaults)
        sim.run()

        minvec.append(sim._stats[0])
        maxvec.append(sim._stats[1])
        ordvec.append(sim._stats[2])

        print >>f, g, sim._stats[0], sim._stats[1], sim._stats[2]

    print

    f.close()

    plt.clf()
    plt.cla()

    plt.plot(xvec, minvec, 'b.-', label='min rho')
    plt.plot(xvec, maxvec, 'r.-', label='max rho')
    plt.plot(xvec, ordvec, 'k.-', label='order parameter')
    plt.gca().yaxis.grid(True)
    plt.gca().yaxis.grid(True, which='minor')
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.ylabel('rho / order parameter')
    plt.xlabel('g')
    plt.legend(loc='upper left')
    plt.savefig(os.path.join(basepath, '%s.pdf' % precision), format='pdf')

parser = OptionParser()
parser.add_option('--precision', dest='precision', help='precision (single, double)', type='choice', choices=['single', 'double'], default='single')
(options, args) = parser.parse_args()

from examples.sc_phase_separation import SCSim, GeoSC
run_test('sc_phase_separation', options.precision)

#!/usr/bin/python -u

"""Phase separation in the single fluid Shan-Chen model.

This test runs a phase separation simulation with multiple values of the
Shan-Chen coupling constant and verifies that spontaneous spinodal
decomposition takes place around G = 4.0.
"""

import argparse
import os
import shutil
import tempfile

import matplotlib
import numpy as np

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from examples.sc_phase_separation import SCSim
from sailfish.geo import LBGeometry2D
from sailfish.controller import LBSimulationController

POINTS = 26 * 3 + 1


def run_test(name, precision):
    minvec = []
    maxvec = []
    ordvec = []
    basepath = os.path.join('regtest/results', name)

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    xvec = np.linspace(3, 5.6, num=POINTS)
    f = open(os.path.join(basepath, '%s.dat' % precision), 'w')
    output = os.path.join(tmpdir, 'phase_sep')

    for g in xvec:
        print ' {0}'.format(g),
        defaults = {
                'quiet': True,
                'verbose': False,
                'every': 1000,
                'max_iters': 100000,
                'output': output,
                'G': g,
            }

        LBSimulationController(SCSim, LBGeometry2D,
                default_config=defaults).run(ignore_cmdline=True)

        data = np.load('{0}_blk0_{1}.npz'.format(output, 100000))
        rho = data['rho']
        avg = np.average(rho)
        order = np.sqrt(np.average(np.square(rho - avg))) / avg

        minvec.append(np.min(rho))
        maxvec.append(np.max(rho))
        ordvec.append(order)
        print >>f, g, minvec[-1], maxvec[-1], ordvec[-1]

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


parser = argparse.ArgumentParser()
parser.add_argument('--precision', type=str, default='single',
        choices=['single', 'double'])
args, remaining = parser.parse_known_args()
tmpdir = tempfile.mkdtemp()
run_test('sc_phase_separation', args.precision)
shutil.rmtree(tmpdir)

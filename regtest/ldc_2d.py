#!/usr/bin/python

"""Compares Sailfish results for the 2D lid driven cavity test case with
numerical results from the literature. Numerical solutions of 2-D steady
incompressible driven cavity flow at high Reynolds numbers E. Erturk; T. C. Corke
and C. Gokcol with numerical calculation of Navier-Stokes equations;
grid size: 601x601."""

import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import os
import shutil
import tempfile

from examples.ldc_2d import LDCBlock, LDCSim
from sailfish.controller import LBSimulationController
from sailfish import io

from utils.merge_subdomains import merge_subdomains

tmpdir = tempfile.mkdtemp()

max_iters = [500000, 1000000, 2000000, 3000000, 4000000, 4500000, 4500000, 5500000, 5500000, 6000000]
reynolds = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 21000]


class TestLDCSim(LDCSim):

    @classmethod
    def update_defaults(cls, defaults):
        LDCBlock.max_v = 0.05
        LDCSim.update_defaults(defaults)
        defaults.update({
            'quiet': True,
            'lat_nx': 512,
            'lat_ny': 512,
            'output': os.path.join(tmpdir, 'result')})

    @classmethod
    def modify_config(cls, config):
        config.visc = (config.lat_nx - 2) * LDCBlock.max_v / config.re
        config.every = config.max_iters

    @classmethod
    def add_options(cls, group, dim):
        LDCSim.add_options(group, dim)
        group.add_argument('--re', help='Reynolds number', type=int, default=0)


def save_output(basepath, max_iters):
    merged = merge_subdomains(os.path.join(tmpdir, 'result'),
            io.filename_iter_digits(max_iters), max_iters, save=False)

    rho = merged['rho']
    lat_ny, lat_nx = rho.shape

    vx = merged['v'][0]
    vy = merged['v'][1]

    nxh = lat_nx / 2
    nyh = lat_ny / 2

    res_vx = (vx[:, nxh] + vx[:, nxh-1]) / 2 / LDCBlock.max_v
    res_vy = (vy[nyh, :] + vy[nyh-1, :]) / 2 / LDCBlock.max_v

    plt.plot(res_vx, np.linspace(-1.0, 1.0, lat_ny), label='Sailfish')
    plt.plot(np.linspace(-1.0, 1.0, lat_nx), res_vy, label='Sailfish')

    np.savetxt(os.path.join(basepath, 'vx.dat'), res_vx)
    np.savetxt(os.path.join(basepath, 'vy.dat'), res_vy)


def run_test(name, re, max_iters, i):
    print 'Testing Re = %s' % re

    basepath = os.path.join('results', name, 're%s' % re)
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    ctrl = LBSimulationController(TestLDCSim,
                                  default_config={'re': re, 'max_iters': max_iters})
    ctrl.run(ignore_cmdline=True)
    horiz = np.loadtxt('ldc_golden/vx2d', skiprows=4)
    vert = np.loadtxt('ldc_golden/vy2d', skiprows=4)

    plt.plot(horiz[:, 0] * 2 - 1, horiz[:, i], '.', label='Paper')
    plt.plot(vert[:, i], 2 * (vert[:, 0] - 0.5), '.', label='Paper')
    save_output(basepath, max_iters)
    plt.legend(loc='lower right')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.gca().yaxis.grid(True, which='minor')

    plt.title('2D Lid Driven Cavity, Re = %s' % re)
    print os.path.join(basepath, 'results.pdf')
    plt.savefig(os.path.join(basepath, 'results.pdf'), format='pdf')

    plt.clf()
    plt.cla()
    plt.show()


for i, (re, max_it) in enumerate(zip(reynolds, max_iters)):
    run_test('ldc2d', re, max_it, i+1)

shutil.rmtree(tmpdir)

#!/usr/bin/python -u

import os
import shutil
import tempfile

import numpy as np
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt

from sailfish.controller import LBSimulationController
from examples import poiseuille
from examples import poiseuille_3d
from sailfish import io


POINTS = 30


class TestPoiseuille2D(poiseuille.PoiseuilleSim):
    @classmethod
    def update_defaults(cls, defaults):
        poiseuille.PoiseuilleSim.update_defaults(defaults)
        defaults.update({
            'stationary': True,
            'horizontal': True,
            'every': 100,
            'quiet': True,
            # Use an odd number of nodes here so that the largest
            # velocity is attained exactly at a node.
            'lat_nx': 127,
            'lat_ny': 128})


def run_test_2d():
    xvec = np.logspace(-3, -1, num=POINTS)
    yvec = np.zeros(POINTS, dtype=np.float64)
    tmpdir = tempfile.mkdtemp()

    summary_path = 'regtest/results/poiseuille/summary.png'
    profile_path = 'regtest/results/poiseuille'

    for i, visc in enumerate(xvec):
        print '%f ' % visc,

        max_iters = int(100 / visc)
        base_path = os.path.join(tmpdir, 'visc{0}'.format(i))

        defaults = {}
        defaults['visc'] = visc
        defaults['max_iters'] = max_iters
        defaults['output'] = base_path

        ctrl = LBSimulationController(TestPoiseuille2D, default_config=defaults)
        ctrl.run()

        digits = io.filename_iter_digits(max_iters)

        final_iter = 100 * (max_iters / 100)
        fname = io.filename(base_path, digits, 0, final_iter)
        res = np.load(fname)

        vx = res['v'][0]
        nyw = vx.shape[1] / 2
        hy = np.mgrid[0:vx.shape[0]]

        profile_sim = vx[:,nyw]
        profile_th = TestPoiseuille2D.subdomain.velocity_profile(ctrl.config, hy)

        yvec[i] = np.nanmax(profile_sim) / np.nanmax(profile_th) - 1.0

        plt.gca().yaxis.grid(True)
        plt.gca().xaxis.grid(True)
        plt.gca().xaxis.grid(True, which='minor')
        plt.gca().yaxis.grid(True, which='minor')
        plt.plot((profile_th[1:-1] - profile_sim[1:-1]) / np.nanmax(profile_th), 'b.-')
        plt.gca().set_xbound(0, len(profile_sim)-2)
        plt.title('(theoretical - simulation) / theo; visc = %f' % visc)
        plt.savefig(os.path.join(profile_path, 'profile{0:02d}.png'.format(i)),
                format='png')

        plt.clf()
        plt.cla()

    plt.clf()
    plt.cla()

    plt.semilogx(xvec, yvec * 100, 'b.-')
    plt.gca().yaxis.grid(True)
    plt.gca().yaxis.grid(True, which='minor')
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.ylabel('max velocity / theoretical max velocity - 1 [%]')
    plt.xlabel('viscosity')
    plt.savefig(summary_path, format='png')

    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    run_test_2d()

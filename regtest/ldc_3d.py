#!/usr/bin/python -u

import os
import numpy as np
import matplotlib
from optparse import OptionParser

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from examples.lbm_ldc_3d import LBMGeoLDC, LDCSim
from sailfish import geo

MAX_ITERS = 50000

def run_test(bc, precision, model, grid, name):
    xvec = []
    yvec = []
    basepath = os.path.join('regtest/results', name, grid, model, precision)

    if not os.path.exists(basepath):
        os.makedirs(basepath)

    defaults = {'grid': grid, 'model': model, 'precision': precision,
            'max_iters': MAX_ITERS, 'batch': True, 'quiet': True, 'verbose':
            False}
    sim = LTestLDCSim(defaults)
    sim.run()

    horiz = np.loadtxt('regtest/ldc_golden/re400_horiz', skiprows=1)
    vert = np.loadtxt('regtest/ldc_golden/re400_vert', skiprows=1)

    plt.plot(2 * (horiz[:,0] - 0.5), -2 * (horiz[:,1] - 0.5), label='Sheu, Tsai paper')
    plt.plot(2 * (vert[:,0] - 0.5), -2 * (vert[:,1] - 0.5), label='Sheu, Tsai paper')

    plt.plot(sim.res_vx, np.linspace(-1.0, 1.0, sim.options.lat_nz), label='Sailfish')
    plt.plot(np.linspace(-1.0, 1.0, sim.options.lat_nx), sim.res_vz, label='Sailfish')

    np.savetxt(os.path.join(basepath, 're400_vx.dat'), sim.res_vx)
    np.savetxt(os.path.join(basepath, 're400_vz.dat'), sim.res_vz)

    plt.legend(loc='lower right')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.gca().yaxis.grid(True, which='minor')

    plt.title('Lid Driven Cavity, Re = 400')
    plt.savefig(os.path.join(basepath, 're400-%s.pdf' % bc), format='pdf')

    plt.clf()
    plt.cla()

parser = OptionParser()
parser.add_option('--precision', dest='precision', help='precision (single, double)', type='choice', choices=['single', 'double'], default='single')
parser.add_option('--model', dest='model', help='model', type='choice', choices=['mrt', 'bgk'], default='bgk')
parser.add_option('--grid', dest='grid', help='grid', type='string', default='')
parser.add_option('--bc', dest='bc', help='boundary conditions to test (comma separated', type='string', default='')
(options, args) = parser.parse_args()

name = 'ldc3d'

if options.grid:
    grid = options.grid
else:
    grid = 'D3Q19'

geo_type = geo.LBMGeo.NODE_VELOCITY

if options.bc:
    bcs = filter(lambda x: geo_type in geo.get_bc(x).supported_types, options.bc.split(','))
else:
    bcs = [x.name for x in geo.SUPPORTED_BCS if geo_type in x.supported_types]

class LTestLDCSim(LDCSim):
    def __init__(self, defaults):
        super(LTestLDCSim, self).__init__(LBMGeoLDC, defaults=defaults)
        self.clear_hooks()
        self.add_iter_hook(self.options.max_iters-1, self.save_output)

    def save_output(self):
        nxh = self.options.lat_nx/2
        nyh = self.options.lat_ny/2
        nzh = self.options.lat_nz/2

        self.res_vx = (self.vx[:, nyh, nxh] + self.vx[:, nyh-1, nxh-1]) / 2 / self.geo.max_v
        self.res_vz = (self.vz[nzh, nyh, :] + self.vz[nzh-1, nyh-1, :]) / 2 / self.geo.max_v


for bc in bcs:
    print 'Running test for "%s".' % bc
    run_test(bc, options.precision, options.model, grid, name)


#!/usr/bin/python 

import numpy as np
import matplotlib

import math
matplotlib.use('cairo')
import matplotlib.pyplot as plt
from optparse import OptionParser
import os
import shutil
import tempfile

from examples.lbm_ldc_multi_3d import LDCGeometry, LDCBlock, LDCSim
from sailfish.controller import LBSimulationController
from sailfish import geo
from sailfish import geo_block

tmpdir = tempfile.mkdtemp()
MAX_ITERS = 100
LAT_NX = 128
LAT_NY = 128
LAT_NZ = 128
output = ''

name = 'ldc3d'


class TestLDCSim(LDCSim):  

    @classmethod
    def update_defaults(cls, defaults):
        LDCBlock.max_v = 0.05
        LDCSim.update_defaults(defaults)
        defaults.update({
            'max_iters': MAX_ITERS,
            'lat_nx': LAT_NX,
            'lat_ny': LAT_NY,
            'lat_nz': LAT_NZ,
            'output': os.path.join(tmpdir,'result')})        
	
    @classmethod
    def modify_config(cls, config):
        config.visc = (config.lat_nx-2) * LDCBlock.max_v / config.re
        config.every = config.max_iters - 1
	
        # Protection in the event of max_iters changes from the command line.
        global MAX_ITERS
        MAX_ITERS = config.max_iters

    @classmethod
    def add_options(cls, group, dim):
        LDCSim.add_options(group, dim)
        group.add_argument('--re', dest="re", help = 'Reynolds number', type=int, default=400)


def save_output(basepath):
    name_digits = str(int(math.log10(MAX_ITERS)) + 1)
    opt = ('%s_blk0_%0' + name_digits+ 'd'+'.npz') % (tmpdir+"/result", MAX_ITERS-1)
    href = np.load(opt)

    hrho = href['rho']
    lat_nz, lat_ny, lat_nx = hrho.shape

    vx = href['v'][0]
    vy = href['v'][1]
    vz = href['v'][2] 

    nxh = lat_nx/2
    nyh = lat_ny/2
    nzh = lat_nz/2

    res_vx = (vx[:, nyh, nxh] + vx[:, nyh-1, nxh-1]) / 2 / LDCBlock.max_v
    res_vz = (vz[nzh, nyh, :] + vz[nzh-1, nyh-1, :]) / 2 / LDCBlock.max_v
    
    plt.plot(res_vx, np.linspace(-1.0, 1.0, lat_nz), label='Sailfish')
    plt.plot(np.linspace(-1.0, 1.0, lat_nx), res_vz, label='Sailfish')

    np.savetxt(os.path.join(basepath, 're400_vx.dat'), res_vx)
    np.savetxt(os.path.join(basepath, 're400_vz.dat'), res_vz)

		
def run_test(name):   
    basepath = os.path.join('results', name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    ctrl = LBSimulationController(TestLDCSim, LDCGeometry)     
    ctrl.run()
    horiz = np.loadtxt('ldc_golden/re400_horiz', skiprows=1)
    vert = np.loadtxt('ldc_golden/re400_vert', skiprows=1)
    
    plt.plot(2 * (horiz[:,0] - 0.5), -2 * (horiz[:,1] - 0.5), label='Sheu, Tsai paper')
    plt.plot(2 * (vert[:,0] - 0.5), -2 * (vert[:,1] - 0.5), label='Sheu, Tsai paper')
    save_output(basepath)
    plt.legend(loc='lower right')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.gca().yaxis.grid(True, which='minor')

    plt.title('Lid Driven Cavity, Re = 400')
    print os.path.join(basepath, 're400.pdf' )
    plt.savefig(os.path.join(basepath, 're400.pdf' ), format='pdf')
	
    plt.clf()
    plt.cla()
    plt.show()
    shutil.rmtree(tmpdir)
 
   
run_test(name)

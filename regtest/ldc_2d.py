#!/usr/bin/python 

"""Compares Sailfish results for the 2D lid driven cavity test case with 
numerical results from the literature. Numerical solutions of 2-D steady 
incompressible driven cavity ow at high Reynolds numbers E. Erturk; T. C. Corke
and C. Gokcol with numerical calculation of Navier-Stokes equations;  
grid size: 601X601 """

import numpy as np
import matplotlib

import math
matplotlib.use('cairo')
import matplotlib.pyplot as plt
from optparse import OptionParser
import os
import shutil
import tempfile

from examples.lbm_ldc_multi import LDCBlock, LDCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import SubdomainSpec2D

tmpdir = tempfile.mkdtemp()

BLOCKS = 1
LAT_NX = 512
LAT_NY = 512
output = ''
RE = 1000
MAX_ITERS = 500000

max_iters = [500000, 1000000, 2000000, 3000000, 4000000, 4500000, 4500000, 5500000, 5500000, 6000000]
reynolds = [1000, 2500, 5000, 7500, 10000, 12500, 15000, 17500, 20000, 21000]
name = 'ldc2d'


class TestLDCGeometry(LBGeometry2D):

    def blocks(self, n=None):
        blocks = []
        bps = self.config.blocks
        yq = self.gy / bps
        yd = self.gy % bps

        for k in range(0, bps):
            ysize = yq
            if k == bps - 1:
                ysize += yd
            blocks.append(SubdomainSpec2D((0, k * yq), 
                                          (self.gx, ysize)))
        return blocks



class TestLDCSim(LDCSim):  

    @classmethod
    def update_defaults(cls, defaults):
        LDCBlock.max_v = 0.05
        LDCSim.update_defaults(defaults)
        defaults.update({
            'max_iters': MAX_ITERS,
            'lat_nx': LAT_NX,
            'lat_ny': LAT_NY,
            'output': os.path.join(tmpdir, 'result')})        
	
    @classmethod
    def modify_config(cls, config):
        print config.re
        config.visc = (config.lat_nx - 2) * LDCBlock.max_v / config.re
        config.every = config.max_iters - 1
	
        # Protection in the event of max_iters changes from the command line.
        global MAX_ITERS, BLOCKS, LAT_NX, LAT_NY
        MAX_ITERS = config.max_iters
        BLOCKS = config.blocks
        LAT_NX = config.lat_nx
        LAT_NY = config.lat_ny

    @classmethod
    def add_options(cls, group, dim):
        LDCSim.add_options(group, dim)
        group.add_argument('--re', dest="re", help='Reynolds number', type=int, default=RE)


def save_output(basepath):
    name_digits = str(int(math.log10(MAX_ITERS)) + 1)
    opt = ('%s_blk0_%0' + name_digits+ 'd' + '.npz') % (os.path.join(tmpdir, 'result'), MAX_ITERS - 1)
    href = np.load(opt)
    hrho = href['rho']

    vx = href['v'][0]
    vy = href['v'][1]

    for i in range(BLOCKS-1):
        opt = ('%s_blk%s_' + '%0' + name_digits + 'd'+'.npz') % (tmpdir+
					"/result", str(i+1), MAX_ITERS-1)
        print opt
        href = np.load(opt)
        hrho_p = href['rho']     
        vx_p = href['v'][0]
        vy_p = href['v'][1]
        vx = np.vstack([vx, vx_p])
        vy = np.vstack([vy, vy_p])
    nxh = LAT_NX / 2
    nyh = LAT_NY / 2

    res_vx = (vx[:, nxh] + vx[:, nxh-1]) / 2 / LDCBlock.max_v
    res_vy = (vy[nyh, :] + vy[nyh-1, :]) / 2 / LDCBlock.max_v
    
    plt.plot(res_vx, np.linspace(-1.0, 1.0, LAT_NY), label='Sailfish')
    plt.plot(np.linspace(-1.0, 1.0, LAT_NX), res_vy, label='Sailfish')

    np.savetxt(os.path.join(basepath, 'vx.dat'), res_vx)
    np.savetxt(os.path.join(basepath, 'vy.dat'), res_vy)

		
def run_test(name, i):   
    global RE 
    RE = reynolds[i]
    global MAX_ITERS
    MAX_ITERS = max_iters[i]
    basepath = os.path.join('results', name, 're%s' % RE)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    
    ctrl = LBSimulationController(TestLDCSim, TestLDCGeometry)  
    ctrl.run()
    horiz = np.loadtxt('ldc_golden/vx2d', skiprows=4)
    vert = np.loadtxt('ldc_golden/vy2d', skiprows=4)
    
    plt.plot(horiz[:, 0] * 2 - 1, horiz[:, i+1], label='Paper')
    plt.plot(vert[:, i+1], 2 * (vert[:, 0] - 0.5), label='Paper')
    save_output(basepath)
    plt.legend(loc='lower right')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.gca().yaxis.grid(True, which='minor')

    plt.title('Lid Driven Cavity, Re = %s' % RE)
    print os.path.join(basepath, 'results.pdf')
    plt.savefig(os.path.join(basepath, 'results.pdf'), format='pdf')
	
    plt.clf()
    plt.cla()
    plt.show()
    
 
for i in range(10):
    run_test(name, i)
shutil.rmtree(tmpdir)

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

from examples.lbm_ldc_multi import LDCGeometry, LDCBlock, LDCSim
from sailfish.controller import LBSimulationController
from sailfish import geo
from sailfish import geo_block

tmpdir = tempfile.mkdtemp()
MAX_ITERS = 1000000
LAT_NX = 512
LAT_NY = 512
output = ''
RE = 1000

reynolds = [1000, 2500, 5000]
name = 'ldc2d'


class TestLDCSim(LDCSim):  

    @classmethod
    def update_defaults(cls, defaults):
        LDCBlock.max_v = 0.05
        LDCSim.update_defaults(defaults)
        defaults.update({
            'max_iters': MAX_ITERS,
            'lat_nx': LAT_NX,
            'lat_ny': LAT_NY,
            'output': os.path.join(tmpdir,'result')})        
	
    @classmethod
    def modify_config(cls, config):
        print config.re
        config.visc = (config.lat_nx-2) * LDCBlock.max_v / config.re
        config.every = config.max_iters - 1
	
        # Protection in the event of max_iters changes from the command line.
        global MAX_ITERS
        MAX_ITERS = config.max_iters

    @classmethod
    def add_options(cls, group, dim):
        LDCSim.add_options(group, dim)
        group.add_argument('--re', dest="re", help = 'Reynolds number', type=int, default=RE)


def save_output(basepath):
    name_digits = str(int(math.log10(MAX_ITERS)) + 1)
    opt = ('%s_blk0_%0' + name_digits+ 'd'+'.npz') % (tmpdir+"/result", MAX_ITERS-1)
    href = np.load(opt)

    hrho = href['rho']
    lat_ny, lat_nx = hrho.shape

    vx = href['v'][0]
    vy = href['v'][1]

    nxh = lat_nx / 2
    nyh = lat_ny / 2

    res_vx = (vx[:, nxh] + vx[:, nxh-1]) / 2 / LDCBlock.max_v
    res_vy = (vy[nyh, :] + vy[nyh-1, :]) / 2 / LDCBlock.max_v
    
    plt.plot(res_vx, np.linspace(-1.0, 1.0, lat_ny) , label='Sailfish')
    plt.plot(np.linspace(-1.0, 1.0, lat_nx), res_vy, label='Sailfish ')

    np.savetxt(os.path.join(basepath, 'vx.dat'), res_vx)
    np.savetxt(os.path.join(basepath, 'vy.dat'), res_vy)

		
def run_test(name, i):   
    global RE 
    RE = reynolds[i]
    basepath = os.path.join('results', name, 're%s' % RE)
    if not os.path.exists(basepath):
        os.makedirs(basepath)
    
    ctrl = LBSimulationController(TestLDCSim, LDCGeometry)  
    ctrl.run()
    horiz = np.loadtxt('ldc_golden/vx2d', skiprows=1)
    vert = np.loadtxt('ldc_golden/vy2d', skiprows=1)
    
    plt.plot(horiz[:,0]*2-1, horiz[:,i+1]  , label='Paper')
    plt.plot(vert[:,i+1],2 * (vert[:,0] - 0.5) , label='Paper')
    save_output(basepath)
    plt.legend(loc='lower right')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.gca().yaxis.grid(True, which='minor')

    plt.title('Lid Driven Cavity, Re = %s'% RE)
    print os.path.join(basepath, 'results.pdf')
    plt.savefig(os.path.join(basepath,'results.pdf'), format='pdf')
	
    plt.clf()
    plt.cla()
    plt.show()
    
 
for i in range(3):
    run_test(name, i)
shutil.rmtree(tmpdir)

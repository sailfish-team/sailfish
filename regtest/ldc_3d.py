#!/usr/bin/python 

import os
import numpy as np
import matplotlib
from optparse import OptionParser
import tempfile
import shutil
import math

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from examples.lbm_ldc_multi_3d import LDCGeometry, LDCBlock, LDCSim
from sailfish.controller import LBSimulationController
from sailfish import geo
from sailfish import geo_block



tmpdir = tempfile.mkdtemp()
MAX_ITERS = 50000
LAT_NX=128
LAT_NY=128
output = ''


name = 'ldc3d'

    

class LTestLDCSim(LDCSim):
    
    def save_output(self):
        nxh = self.config.lat_nx/2
        nyh = self.config.lat_ny/2
        nzh = self.config.lat_nz/2

        self.res_vx = (self.vx[:, nyh, nxh] + self.vx[:, nyh-1, nxh-1]) / 2 / self.LDCBlock.max_v
        self.res_vz = (self.vz[nzh, nyh, :] + self.vz[nzh-1, nyh-1, :]) / 2 / self.LDCBlock.max_v

    @classmethod
    def update_defaults(cls, defaults):
    LDCBlock.max_v=0.05
	defaults.update({
            'max_iters': MAX_ITERS,
	    'lat_nx':LAT_NX ,
	    'lat_ny':LAT_NY ,
            'output': tmpdir+"/wynik"})
	
        LDCSim.update_defaults(defaults)
	
    @classmethod
    def modify_config(cls, config):
        config.visc = (config.lat_nx-2)*LDCBlock.max_v/config.re
	config.every=config.max_iters-1

    @classmethod
    def add_options(cls, group, dim):
        LDCSim.add_options(group, dim)
        group.add_argument('--re', dest="re", help='Reynolds number', type=int, default=400)


class LTestSimulationController(LBSimulationController): 
    res_vx =""
    res_vz =""

    def save_output(self):
        nxh = self.config.lat_nx/2
        nyh = self.config.lat_ny/2
        nzh = self.config.lat_nz/2
	dig=str(int(math.log10(self.config.max_iters)) + 1)
	opt = ('%s_blk0_%0' + str(int(math.log10(self.config.max_iters)) + 1) + 'd'+'.npz') % (self.config.output, self.config.max_iters-1)
	href = np.load(opt)

        hrho = href['rho']
        vx  = href['v'][0]
        vy  = href['v'][1]
        vz  = href['v'][2] 

        self.res_vx = (vx[:, nyh, nxh] + vx[:, nyh-1, nxh-1]) / 2 / LDCBlock.max_v
        self.res_vz = (vz[nzh, nyh, :] + vz[nzh-1, nyh-1, :]) / 2 / LDCBlock.max_v

		
def run_test(name):   
    xvec = []
    yvec = []
    basepath = os.path.join('results', name)

    if not os.path.exists(basepath):
        os.makedirs(basepath)

	
    ctrl=LTestSimulationController(LTestLDCSim, LDCGeometry)     
    ctrl.run()
    horiz = np.loadtxt('ldc_golden/re400_horiz', skiprows=1)
    vert = np.loadtxt('ldc_golden/re400_vert', skiprows=1)
    
    plt.plot(2 * (horiz[:,0] - 0.5), -2 * (horiz[:,1] - 0.5), label='Sheu, Tsai paper')
    plt.plot(2 * (vert[:,0] - 0.5), -2 * (vert[:,1] - 0.5), label='Sheu, Tsai paper')

    ctrl.save_output()
    
    plt.plot(ctrl.res_vx, np.linspace(-1.0, 1.0, ctrl.config.lat_nz), label='Sailfish')
    plt.plot(np.linspace(-1.0, 1.0, ctrl.config.lat_nx), ctrl.res_vz, label='Sailfish')

    np.savetxt(os.path.join(basepath, 're400_vx.dat'), ctrl.res_vx)
    np.savetxt(os.path.join(basepath, 're400_vz.dat'), ctrl.res_vz)

    plt.legend(loc='lower right')
    plt.gca().yaxis.grid(True)
    plt.gca().xaxis.grid(True)
    plt.gca().xaxis.grid(True, which='minor')
    plt.gca().yaxis.grid(True, which='minor')

    plt.title('Lid Driven Cavity, Re = 400')
    print os.path.join(basepath, 're400-%s.pdf' )
    plt.savefig(os.path.join(basepath, 're400-%s.pdf' ), format='pdf')
	
    plt.clf()
    plt.cla()
    plt.show()

run_test(name)

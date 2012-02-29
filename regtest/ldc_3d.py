#!/usr/bin/python

import numpy as np
import matplotlib

matplotlib.use('cairo')
import matplotlib.pyplot as plt
import os
import shutil
import tempfile

<<<<<<< HEAD
from examples.lbm_ldc_multi_3d import LDCBlock, LDCSim
=======
from examples.ldc_3d import LDCGeometry, LDCBlock, LDCSim
from sailfish import io
>>>>>>> upstream/multigpu
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry3D
from sailfish.geo_block import SubdomainSpec3D

tmpdir = tempfile.mkdtemp()
MAX_ITERS = 100
LAT_NX = 164
LAT_NY = 164
LAT_NZ = 164
BLOCKS = 1
output = ''

name = 'ldc3d'


class TestLDCGeometry(LBGeometry3D):

    def blocks(self, n=None):
        blocks = []
        bps = self.config.blocks
        zq = self.gz / bps
        zd = self.gz % bps

        for k in range(0, bps):
            zsize = zq
            if k == bps - 1:
                zsize += zd
            blocks.append(SubdomainSpec3D((0, 0, k * zq), 
                                          (self.gx, self.gy, zsize)))
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
            'lat_nz': LAT_NZ,
            'output': os.path.join(tmpdir, 'result')})

    @classmethod
    def modify_config(cls, config):
        config.visc = (config.lat_nx-2) * LDCBlock.max_v / config.re
        config.every = config.max_iters

        # Protection in the event of max_iters changes from the command line.
        global MAX_ITERS, BLOCKS, LAT_NX, LAT_NY, LAT_NZ
        MAX_ITERS = config.max_iters
        BLOCKS = config.blocks
        LAT_NX = config.lat_nx
        LAT_NY = config.lat_ny
        LAT_NZ = config.lat_nz


    @classmethod
    def add_options(cls, group, dim):
        LDCSim.add_options(group, dim)
        group.add_argument('--re', dest="re", help = 'Reynolds number', type=int, default=400)


def save_output(basepath):
<<<<<<< HEAD
    name_digits = str(int(math.log10(MAX_ITERS)) + 1)
    opt = ('%s_blk0_%0' + name_digits+ 'd'+'.npz') % (tmpdir+"/result", MAX_ITERS-1)
    href = np.load(opt)
    hrho = href['rho']     
    vx = href['v'][0]
    vy = href['v'][1]
    vz = href['v'][2]
=======
    res = np.load(io.filename(os.path.join(tmpdir, 'result'),
        io.filename_iter_digits(MAX_ITERS), 0, MAX_ITERS))

    rho = res['rho']
    lat_nz, lat_ny, lat_nx = rho.shape

    vx = res['v'][0]
    vy = res['v'][1]
    vz = res['v'][2]
>>>>>>> upstream/multigpu

    for i in range(BLOCKS-1):
        opt = ('%s_blk%s_' + '%0' + name_digits + 'd'+'.npz') % (tmpdir+"/result", str(i+1), MAX_ITERS-1)
        href = np.load(opt)
        hrho_p = href['rho']     
        vx_p = href['v'][0]
        vy_p = href['v'][1]
        vz_p = href['v'][2]
        vx = np.vstack([vx, vx_p])
        vy = np.vstack([vy, vy_p])
        vz = np.vstack([vz, vz_p])
   
    nxh = LAT_NX / 2
    nyh = LAT_NY / 2
    nzh = LAT_NZ / 2

    res_vx = (vx[:, nyh, nxh] + vx[:, nyh-1, nxh-1]) / 2 / LDCBlock.max_v
    res_vz = (vz[nzh, nyh, :] + vz[nzh-1, nyh-1, :]) / 2 / LDCBlock.max_v

    np.savetxt(os.path.join(basepath, 're400_vx.dat'), res_vx)
    np.savetxt(os.path.join(basepath, 're400_vz.dat'), res_vz)

    plt.plot(res_vx, np.linspace(-1.0, 1.0, LAT_NZ), label='Sailfish')
    plt.plot(np.linspace(-1.0, 1.0, LAT_NX), res_vz, label='Sailfish')




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

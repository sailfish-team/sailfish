#!/usr/bin/python

import math
import numpy as np
from sailfish.geo import LBGeometry2D
from sailfish.subdomain import SubdomainSpec2D, Subdomain2D
from sailfish.node_type import NTFullBBWall, NTEquilibriumVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBFluidSim
from sailfish.util import logpoints,linpoints

class LDCGeometry(LBGeometry2D):
    def subdomains(self, n=None):
        subdomains = []
        bps = int(math.sqrt(self.config.ldc_subdomains))

        # Special case.
        if self.config.ldc_subdomains == 3:
            w1 = self.gx / 2
            w2 = self.gx - w1
            h1 = self.gy / 2
            h2 = self.gy - h1

            subdomains.append(SubdomainSpec2D((0, 0), (w1, h1)))
            subdomains.append(SubdomainSpec2D((0, h1), (w1, h2)))
            subdomains.append(SubdomainSpec2D((w1, 0), (w2, self.gy)))
            return subdomains

        if bps**2 != self.config.ldc_subdomains:
            print ('Only configurations with '
                    'square-of-interger numbers of subdomains are supported. '
                    'Falling back to {0} x {0} subdomains.'.format(bps))

        yq = self.gy / bps
        ydiff = self.gy % bps
        xq = self.gx / bps
        xdiff = self.gx % bps

        for i in range(0, bps):
            xsize = xq
            if i == bps - 1:
                xsize += xdiff

            for j in range(0, bps):
                ysize = yq
                if j == bps - 1:
                    ysize += ydiff

                subdomains.append(SubdomainSpec2D((i * xq, j * yq), (xsize, ysize)))

        return subdomains


class LDCBlock(Subdomain2D):
    """2D Lid-driven cavity geometry."""

    max_v = 0.10

    def boundary_conditions(self, hx, hy):
        wall_bc = NTFullBBWall
        velocity_bc = NTEquilibriumVelocity

        lor = np.logical_or
        land = np.logical_and
        lnot = np.logical_not

        wall_map = land(lor(lor(hx == self.gx-1, hx == 0), hy == 0),
                        lnot(hy == self.gy-1))
        self.set_node(hy == self.gy-1, velocity_bc((self.max_v, 0.0)))
        self.set_node(wall_map, wall_bc)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vx[hy == self.gy-1] = self.max_v


class LDCSim(LBFluidSim):
    subdomain = LDCBlock





    
    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 1024/8,
            'lat_ny': 1024/8,
            'max_iters':200000,
            'every':100,
            'visc': 0.06011, # overwritten in after step
            'model':'mrt',
            })

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)

        group.add_argument('--ldc_subdomains', type=int, default=1, help='number of blocks to use')


    u_norm_table=[]
    def after_step(self,runner):

        from sailfish.sym import relaxation_time
        import pycuda.driver as cuda

        every_n=1000 #self.config.every/1

########### before each u field processing
        if self.iteration%(every_n-1) == 0:
            self.need_sync_flag=True
        else:
            self.need_sync_flag=False

############## Calculate and save the norm of valocity field ######
        if self.iteration==every_n:

            self.u_old = np.sqrt( runner._vector_fields[0][0]**2+runner._vector_fields[0][1]**2 )


        if self.iteration%every_n == 0 and self.iteration > every_n:

            u=np.sqrt( runner._vector_fields[0][0]**2+runner._vector_fields[0][1]**2 )
            du_norma=np.linalg.norm(u-self.u_old)/(self.config.lat_nx*self.config.lat_ny)
            u_norma=np.linalg.norm(u/(self.config.lat_nx*self.config.lat_ny) )
            self.u_old=u.copy()
            self.u_norm_table.append( (self.iteration,du_norma,u_norma) )


        if self.iteration==self.config.max_iters-1:

            u_norm_table_np=np.array(self.u_norm_table)
            np.savez('unorm',it=u_norm_table_np[:,0],du_norma=u_norm_table_np[:,1],u_norma=u_norm_table_np[:,2])





                
            
if __name__ == '__main__':
    ctrl = LBSimulationController(LDCSim, LDCGeometry)
    ctrl.run()

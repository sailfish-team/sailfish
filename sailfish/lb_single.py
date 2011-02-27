"""Classes for single fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPLv3'

import numpy as np

from sailfish import sym
from sailfish.lb_base import LBSim

class GridError(Exception):
    pass

class LBFluidSim(LBSim):
    kernel_file = "single_fluid.mako"

    @classmethod
    def add_options(cls, group, dim):
        LBSim.add_options(group, dim)

        group.add_argument('--visc', type=float, default=1.0, help='numerical viscosity')
        group.add_argument('--incompressible',
                action='store_true', default=False,
                help='use the incompressible model of Luo and He')
        group.add_argument('--model', help='LB model to use',
                type=str, choices=['bgk', 'mrt'],
                default='bgk')
        group.add_argument('--subgrid', default='none', type=str,
                choices=['none', 'les-smagorinsky'],
                help='subgrid model to use')
        group.add_argument('--smagorinsky_const',
                help='Smagorinsky constant', type=float, default=0.03)

        grids = [x.__name__ for x in sym.KNOWN_GRIDS if x.dim == dim]
        group.add_argument('--grid', help='LB grid', type=str,
                choices=grids, default=grids[0])

    def __init__(self, config):
        LBSim.__init__(self, config)

        self.grids = []
        for x in sym.KNOWN_GRIDS:
            if x.__name__ == config.grid:
                self.grids.append(x)
                break

        if not self.grids:
            raise GridError('Invalid grid selected: {}'.format(config.grid))

        self.equilibrium, self.equilibrium_vars = sym.bgk_equilibrium(self.grids[0])

    @property
    def grid(self):
        return self.grids[0]

    def update_context(self, ctx):
        ctx['tau'] = (6.0 * self.config.visc + 1.0)/2.0
        ctx['visc'] = self.config.visc
        ctx['model'] = self.config.model
        ctx['loc_names'] = ['gx', 'gy', 'gz']
        ctx['simtype'] = 'lbm'
        ctx['grid'] = self.grids[0]
        ctx['grids'] = self.grids
        ctx['bgk_equilibrium'] = self.equilibrium
        ctx['bgk_equilibrium_vars'] = self.equilibrium_vars

        ctx['forces'] = {}
        ctx['force_couplings'] = {}
        ctx['force_for_eq'] = {}
        ctx['image_fields'] = set()

    def initial_conditions(self, runner):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_v = runner.gpu_field(self.v)
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)

        args1 = [gpu_dist1a] + gpu_v + [gpu_rho]
        args2 = [gpu_dist1b] + gpu_v + [gpu_rho]

        runner.exec_kernel('SetInitialConditions', args1, 'P'*len(args1))
        runner.exec_kernel('SetInitialConditions', args2, 'P'*len(args2))

    def get_compute_kernels(self, runner, full_output):
        """
        Args:
          full_output: if True, returns kernels that prepare fields for
              visualization or saving into a file
        """
        gpu_rho = runner.gpu_field(self.rho)
        gpu_v = runner.gpu_field(self.v)
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_map = runner.gpu_geo_map()

        args1 = [gpu_map, gpu_dist1a, gpu_dist1b, gpu_rho] + gpu_v
        args2 = [gpu_map, gpu_dist1b, gpu_dist1b, gpu_rho] + gpu_v

        if full_output:
            args1 += [np.uint32(1)]
            args2 += [np.uint32(1)]
        else:
            args1 += [np.uint32(0)]
            args2 += [np.uint32(0)]

        kernels = []
        kernels.append(runner.get_kernel(
                'CollideAndPropagate', args1, 'P'*(len(args1)-1)+'i'))
        kernels.append(runner.get_kernel(
                'CollideAndPropagate', args2, 'P'*(len(args2)-1)+'i'))
        return kernels

    def init_fields(self, runner):
        self.rho = runner.make_scalar_field(name='rho')
        self.v = runner.make_vector_field(name='v')
        self.vx, self.vy = self.v

# TODO(michalj): Port the single-phase Shan-Chen class.

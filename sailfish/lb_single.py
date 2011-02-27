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
          full_output: if True, returns kernels for

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


####################################################
# OLD STUFF BELOW THIS LINE

""""

    @property
    def sim_info(self):
        ret = lbm.LBMSim.sim_info.fget(self)
        ret['incompressible'] = self.incompressible
        ret['model'] = self.lbm_model
        ret['bc_wall'] = self.options.bc_wall
        ret['bc_slip'] = self.options.bc_slip
        ret['bc_velocity'] = self.options.bc_velocity
        ret['bc_pressure'] = self.options.bc_pressure

        if hasattr(self.geo, 'get_reynolds'):
            ret['Re'] = self.geo.get_reynolds(self.options.visc)

        return ret

    def __init__(self, geo_class, options=[], args=None, defaults=None):
        lbm.LBMSim.__init__(self, geo_class, options, args, defaults)
        self._set_grid(self.options.grid)

        # If the model has not been explicitly specified by the user, try to automatically
        # select a working model.
        if 'model' not in self.options.specified and (defaults is None or 'model' not in defaults.keys()):
            self._set_model(self.options.model, 'mrt', 'bgk')
        else:
            self._set_model(self.options.model)

        self.num_tracers = self.options.tracers
        self.incompressible = self.options.incompressible
        self.equilibrium, self.equilibrium_vars = sym.bgk_equilibrium(self.grid)

    def _init_fields(self, need_dist):
        super(FluidLBMSim, self)._init_fields(need_dist)
        self.vis.add_field(self.rho, 'density', True)

    def _update_ctx(self, ctx):
        ctx['incompressible'] = self.incompressible
        ctx['bc_wall'] = self.options.bc_wall
        ctx['bc_slip'] = self.options.bc_slip

        if self.geo.has_velocity_nodes:
            ctx['bc_velocity'] = self.options.bc_velocity
        else:
            ctx['bc_velocity'] = None

        if self.geo.has_pressure_nodes:
            ctx['bc_pressure'] = self.options.bc_pressure
        else:
            ctx['bc_pressure'] = None

        ctx['bc_wall_'] = geo.get_bc(self.options.bc_wall)
        ctx['bc_slip_'] = geo.get_bc(self.options.bc_slip)
        ctx['bc_velocity_'] = geo.get_bc(self.options.bc_velocity)
        ctx['bc_pressure_'] = geo.get_bc(self.options.bc_pressure)
        ctx['simtype'] = 'fluid'
        ctx['subgrid'] = self.options.subgrid
        ctx['smagorinsky_const'] = self.options.smagorinsky_const



class ShanChenSingle(FluidLBSim):
    @classmethod
    def add_options(cls, group):
        FluidLBSim.add_options(group)
        group.add_argument('--G', type=float, default=1.0,
            help='Shan-Chen interaction strength')

    @property
    def constants(self):
        return [('SCG', self.options.G)]


    def __init__(self, geo_class, options=[], args=None, defaults=None):
        super(ShanChenSingle, self).__init__(geo_class, options, args, defaults)
        self.add_force_coupling(0, 0, 'SCG')
        self.add_nonlocal_field(0)


    def _init_compute_fields(self):
        super(ShanChenSingle, self)._init_compute_fields()
        self.img_rho = self.bind_nonlocal_field(self.gpu_rho, 0)

    def _init_compute_kernels(self):
        # Kernel arguments.
        args_tracer2 = [self.gpu_dist1a, self.geo.gpu_map] + self.gpu_tracer_loc
        args_tracer1 = [self.gpu_dist1b, self.geo.gpu_map] + self.gpu_tracer_loc
        args1 = ([self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_rho] + self.gpu_velocity +
                 [np.uint32(0), self.gpu_rho])
        args2 = ([self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_rho] + self.gpu_velocity +
                 [np.uint32(0), self.gpu_rho])

        # Special argument list for the case where macroscopic quantities data is to be
        # saved in global memory, i.e. a visualization step.
        args1v = ([self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_rho] + self.gpu_velocity +
                  [np.uint32(1), self.gpu_rho])
        args2v = ([self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_rho] + self.gpu_velocity +
                  [np.uint32(1), self.gpu_rho])

        macro_args1 = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_rho]
        macro_args2 = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_rho]

        k_block_size = self._kernel_block_size()
        cnp_name = 'CollideAndPropagate'
        macro_name = 'PrepareMacroFields'

        fields = [self.img_rho]

        kern_cnp1 = self.backend.get_kernel(self.mod, cnp_name,
                    args=args1,
                    args_format='P'*(len(args1)-2)+'iP',
                    block=k_block_size, fields=fields)
        kern_cnp2 = self.backend.get_kernel(self.mod, cnp_name,
                    args=args2,
                    args_format='P'*(len(args2)-2)+'iP',
                    block=k_block_size, fields=fields)
        kern_cnp1s = self.backend.get_kernel(self.mod, cnp_name,
                    args=args1v,
                    args_format='P'*(len(args1v)-2)+'iP',
                    block=k_block_size, fields=fields)
        kern_cnp2s = self.backend.get_kernel(self.mod, cnp_name,
                    args=args2v,
                    args_format='P'*(len(args2v)-2)+'iP',
                    block=k_block_size, fields=fields)
        kern_trac1 = self.backend.get_kernel(self.mod,
                    'LBMUpdateTracerParticles',
                    args=args_tracer1,
                    args_format='P'*len(args_tracer1),
                    block=(1,))
        kern_trac2 = self.backend.get_kernel(self.mod,
                    'LBMUpdateTracerParticles',
                    args=args_tracer2,
                    args_format='P'*len(args_tracer2),
                    block=(1,))
        kern_mac1 = self.backend.get_kernel(self.mod, macro_name,
                         args=macro_args1, args_format='P'*len(macro_args1),
                         block=k_block_size)
        kern_mac2 = self.backend.get_kernel(self.mod, macro_name,
                         args=macro_args2, args_format='P'*len(macro_args2),
                         block=k_block_size)

        # Map: iteration parity -> kernel arguments to use.
        self.kern_map = {
            0: (kern_cnp1, kern_cnp1s, kern_trac1, kern_mac1),
            1: (kern_cnp2, kern_cnp2s, kern_trac2, kern_mac2),
        }

        if self.grid.dim == 2:
            self.kern_grid_size = (self.arr_nx/self.options.block_size, self.arr_ny)
        else:
            self.kern_grid_size = (self.arr_nx/self.options.block_size * self.arr_ny, self.arr_nz)

    def _lbm_step(self, get_data, **kwargs):
        kerns = self.kern_map[self.iter_ & 1]

        self.backend.run_kernel(kerns[3], self.kern_grid_size)
        self.backend.sync()

        if get_data:
            self.backend.run_kernel(kerns[1], self.kern_grid_size)
            if kwargs.get('tracers'):
                self.backend.run_kernel(kerns[2], (self.num_tracers,))
                self.hostsync_tracers()
            self.hostsync_velocity()
            self.hostsync_density()
        else:
            self.backend.run_kernel(kerns[0], self.kern_grid_size)
            if kwargs.get('tracers'):
                self.backend.run_kernel(kerns[2], (self.num_tracers,))

    def _update_ctx(self, ctx):
        super(ShanChenSingle, self)._update_ctx(ctx)
        ctx['simtype'] = 'shan-chen'
        ctx['sc_pseudopotential'] = 'sc_ppot_exp'

"""

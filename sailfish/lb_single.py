"""Classes for single fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import numpy

from sailfish import lbm, geo, sym

class FluidLBMSim(lbm.LBMSim):
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

    def _add_options(self, parser, lb_group):
        grids = [x.__name__ for x in sym.KNOWN_GRIDS if x.dim == self.geo_class.dim]
        default_grid = grids[0]

        lb_group.add_option('--model', dest='model', help='LBE model to use', type='choice', choices=['bgk', 'mrt'], action='store', default='bgk')
        lb_group.add_option('--incompressible', dest='incompressible', help='whether to use the incompressible model of Luo and He', action='store_true', default=False)
        lb_group.add_option('--grid', dest='grid', help='grid type to use', type='choice', choices=grids, default=default_grid)
        lb_group.add_option('--bc_wall', dest='bc_wall', help='boundary condition implementation to use for wall nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_WALL in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='fullbb')
        lb_group.add_option('--bc_slip', dest='bc_slip', help='boundary condition implementation to use for slip nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_SLIP in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='slipbb')
        lb_group.add_option('--bc_velocity', dest='bc_velocity', help='boundary condition implementation to use for velocity nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_VELOCITY in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='equilibrium')
        lb_group.add_option('--bc_pressure', dest='bc_pressure', help='boundary condition implementation to use for pressure nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_PRESSURE in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='equilibrium')
        lb_group.add_option('--subgrid', dest='subgrid', help='subgrid model to use', type='choice',
                choices=['none', 'les-smagorinsky'], default='none')
        lb_group.add_option('--smagorinsky_const', dest='smagorinsky_const', help='Smagorinsky constant', type='float', action='store', default=0.03)

        return []


class ShanChenSingle(FluidLBMSim):
    @property
    def constants(self):
        return [('SCG', self.options.G)]

    def __init__(self, geo_class, options=[], args=None, defaults=None):
        super(ShanChenSingle, self).__init__(geo_class, options, args, defaults)
        self.add_force_coupling(0, 0, 'SCG')
        self.add_nonlocal_field(0)

    def _add_options(self, parser, lb_group):
        super(ShanChenSingle, self)._add_options(parser, lb_group)

        lb_group.add_option('--G', dest='G',
            help='Shan-Chen interaction strength', action='store', type='float',
            default=1.0)
        return None

    def _init_compute_fields(self):
        super(ShanChenSingle, self)._init_compute_fields()
        self.img_rho = self.bind_nonlocal_field(self.gpu_rho, 0)

    def _init_compute_kernels(self):
        # Kernel arguments.
        args_tracer2 = [self.gpu_dist1a, self.geo.gpu_map] + self.gpu_tracer_loc
        args_tracer1 = [self.gpu_dist1b, self.geo.gpu_map] + self.gpu_tracer_loc
        args1 = ([self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_rho] + self.gpu_velocity +
                 [numpy.uint32(0), self.gpu_rho])
        args2 = ([self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_rho] + self.gpu_velocity +
                 [numpy.uint32(0), self.gpu_rho])

        # Special argument list for the case where macroscopic quantities data is to be
        # saved in global memory, i.e. a visualization step.
        args1v = ([self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_rho] + self.gpu_velocity +
                  [numpy.uint32(1), self.gpu_rho])
        args2v = ([self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_rho] + self.gpu_velocity +
                  [numpy.uint32(1), self.gpu_rho])

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

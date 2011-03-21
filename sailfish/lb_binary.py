"""Classes for binary fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import numpy

from sailfish import lbm, lb_single, sym

class BinaryFluidBase(lb_single.FluidLBMSim):
    kernel_file = 'binary_fluid.mako'

    def __init__(self, geo_class, options=[], args=None, defaults=None):
        super(BinaryFluidBase, self).__init__(geo_class, options, args, defaults)
        self._prepare_symbols()
        self.add_nonlocal_field(0)
        self.add_nonlocal_field(1)

    def curr_dists(self):
        if self.iter_ & 1:
            return [self.gpu_dist1b, self.gpu_dist2b]
        else:
            return [self.gpu_dist1a, self.gpu_dist2a]

    def _prepare_symbols(self):
        self.S.alias('phi', self.S.g1m0)

    def _init_fields(self, need_dist):
        lbm.LBMSim._init_fields(self, need_dist)
        self.phi = self.make_field('phi', True)

        if need_dist:
            self.dist2 = self.make_dist(self.grid)

    def _init_compute_fields(self):
        super(BinaryFluidBase, self)._init_compute_fields()
        self.gpu_phi = self.backend.alloc_buf(like=self.phi)
        self.gpu_mom0.append(self.gpu_phi)

        if not self._ic_fields:
            self.gpu_dist2a = self.backend.alloc_buf(like=self.dist2)
            self.gpu_dist2b = self.backend.alloc_buf(like=self.dist2)
        else:
            self.gpu_dist2a = self.backend.alloc_buf(size=self.get_dist_bytes(self.grid), wrap_in_array=False)
            self.gpu_dist2b = self.backend.alloc_buf(size=self.get_dist_bytes(self.grid), wrap_in_array=False)

        self.img_rho = self.bind_nonlocal_field(self.gpu_rho, 0)
        self.img_phi = self.bind_nonlocal_field(self.gpu_phi, 1)

    def _init_compute_kernels(self):
        cnp_args1n = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_dist2a,
                      self.gpu_dist2b, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [numpy.uint32(0)]
        cnp_args1s = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_dist2a,
                      self.gpu_dist2b, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [numpy.uint32(1)]
        cnp_args2n = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_dist2b,
                      self.gpu_dist2a, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [numpy.uint32(0)]
        cnp_args2s = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_dist2b,
                      self.gpu_dist2a, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [numpy.uint32(1)]

        macro_args1 = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist2a, self.gpu_rho, self.gpu_phi]
        macro_args2 = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist2b, self.gpu_rho, self.gpu_phi]

        k_block_size = self._kernel_block_size()
        cnp_name = 'CollideAndPropagate'
        macro_name = 'PrepareMacroFields'
        fields = [self.img_rho, self.img_phi]

        kern_cnp1n = self.backend.get_kernel(self.mod, cnp_name,
                         args=cnp_args1n, args_format='P'*(len(cnp_args1n)-1)+'i',
                         block=k_block_size, fields=fields)
        kern_cnp1s = self.backend.get_kernel(self.mod, cnp_name,
                         args=cnp_args1s, args_format='P'*(len(cnp_args1n)-1)+'i',
                         block=k_block_size, fields=fields)
        kern_cnp2n = self.backend.get_kernel(self.mod, cnp_name,
                         args=cnp_args2n, args_format='P'*(len(cnp_args1n)-1)+'i',
                         block=k_block_size, fields=fields)
        kern_cnp2s = self.backend.get_kernel(self.mod, cnp_name,
                         args=cnp_args2s, args_format='P'*(len(cnp_args1n)-1)+'i',
                         block=k_block_size, fields=fields)
        kern_mac1 = self.backend.get_kernel(self.mod, macro_name,
                         args=macro_args1, args_format='P'*len(macro_args1),
                         block=k_block_size)
        kern_mac2 = self.backend.get_kernel(self.mod, macro_name,
                         args=macro_args2, args_format='P'*len(macro_args2),
                         block=k_block_size)

        # For occupancy analysis in performance tests.
        self._lb_kernel = kern_cnp1n

        # Map: iteration parity -> kernel arguments to use.
        self.kern_map = {
            0: (kern_mac1, kern_cnp1n, kern_cnp1s),
            1: (kern_mac2, kern_cnp2n, kern_cnp2s),
        }

        if self.grid.dim == 2:
            self.kern_grid_size = (self.arr_nx/self.options.block_size, self.arr_ny)
        else:
            self.kern_grid_size = (self.arr_nx/self.options.block_size * self.arr_ny, self.arr_nz)

    def _init_compute_ic(self):
        if not self._ic_fields:
            # Nothing to do, the initial distributions have already been
            # set and copied to the GPU in _init_compute_fields.
            return

        args1 = [self.gpu_dist1a, self.gpu_dist2a] + self.gpu_velocity + [self.gpu_rho, self.gpu_phi]
        args2 = [self.gpu_dist1b, self.gpu_dist2b] + self.gpu_velocity + [self.gpu_rho, self.gpu_phi]

        kern1 = self.backend.get_kernel(self.mod, 'SetInitialConditions',
                    args=args1,
                    args_format='P'*len(args1),
                    block=self._kernel_block_size())

        kern2 = self.backend.get_kernel(self.mod, 'SetInitialConditions',
                    args=args2,
                    args_format='P'*len(args2),
                    block=self._kernel_block_size())

        self.backend.run_kernel(kern1, self.kern_grid_size)
        self.backend.run_kernel(kern2, self.kern_grid_size)
        self.backend.sync()

    def _lbm_step(self, get_data, **kwargs):
        kerns = self.kern_map[self.iter_ & 1]

        self.backend.run_kernel(kerns[0], self.kern_grid_size)
        self.backend.sync()

        if get_data:
            self.backend.run_kernel(kerns[2], self.kern_grid_size)
            self.backend.sync()
            self.hostsync_velocity()
            self.hostsync_density()
            self.backend.from_buf(self.gpu_phi)
            self.backend.sync()
        else:
            self.backend.run_kernel(kerns[1], self.kern_grid_size)

class BinaryFluidFreeEnergy(BinaryFluidBase):
    @property
    def constants(self):
        return [('Gamma', self.options.Gamma), ('A', self.options.A), ('kappa', self.options.kappa),
                ('tau_a', self.options.tau_a), ('tau_b', self.options.tau_b)]

    def __init__(self, geo_class, options=[], args=None, defaults=None):
        super(BinaryFluidFreeEnergy, self).__init__(geo_class, options, args, defaults)
        self.equilibrium, self.equilibrium_vars = sym.free_energy_binary_liquid_equilibrium(self)

    def _add_options(self, parser, lb_group):
        super(BinaryFluidFreeEnergy, self)._add_options(parser, lb_group)

        lb_group.add_option('--bc_wall_grad_phase', dest='bc_wall_grad_phase',
            type='float', default=0.0, help='gradient of the phase field at '
            'the wall; this determines the wetting properties')
        lb_group.add_option('--bc_wall_grad_order', dest='bc_wall_grad_order', type='int',
            default=2, help='order of the gradient stencil used for the '
            'wetting boundary condition at the walls; valid values are 1 and 2')
        lb_group.add_option('--Gamma', dest='Gamma',
            help='Gamma parameter', action='store', type='float',
            default=0.5)
        lb_group.add_option('--kappa', dest='kappa',
            help='kappa parameter', action='store', type='float',
            default=0.5)
        lb_group.add_option('--A', dest='A',
            help='A parameter', action='store', type='float',
            default=0.5)
        lb_group.add_option('--tau_phi', dest='tau_phi', help='relaxation time for the phi field',
                            action='store', type='float', default=1.0)
        lb_group.add_option('--tau_a', dest='tau_a', help='relaxation time for the A component',
                            action='store', type='float', default=1.0)
        lb_group.add_option('--tau_b', dest='tau_b', help='relaxation time for the B component',
                            action='store', type='float', default=1.0)
        return None

    def _update_ctx(self, ctx):
        super(BinaryFluidFreeEnergy, self)._update_ctx(ctx)
        ctx['grids'] = [self.grid, self.grid]
        ctx['tau_phi'] = self.options.tau_phi
        ctx['simtype'] = 'free-energy'
        ctx['bc_wall_grad_phase'] = self.options.bc_wall_grad_phase
        ctx['bc_wall_grad_order'] = self.options.bc_wall_grad_order

    def _prepare_symbols(self):
        """Additional symbols and coefficients for the free-energy binary liquid model."""
        super(BinaryFluidFreeEnergy, self)._prepare_symbols()
        from sympy import Symbol, Rational

        self.S.Gamma = Symbol('Gamma')
        self.S.kappa = Symbol('kappa')
        self.S.A = Symbol('A')
        self.S.alias('lap0', self.S.g0d2m0)
        self.S.alias('lap1', self.S.g1d2m0)
        self.S.make_vector('grad0', self.grid.dim, self.S.g0d1m0x, self.S.g0d1m0y, self.S.g0d1m0z)
        self.S.make_vector('grad1', self.grid.dim, self.S.g1d1m0x, self.S.g1d1m0y, self.S.g1d1m0z)

        if self.grid.dim == 3:
            self.S.wxy = [x[0]*x[1]*Rational(1, 4) for x in sym.D3Q19.basis[1:]]
            self.S.wyz = [x[1]*x[2]*Rational(1, 4) for x in sym.D3Q19.basis[1:]]
            self.S.wxz = [x[0]*x[2]*Rational(1, 4) for x in sym.D3Q19.basis[1:]]
            self.S.wi = []
            self.S.wxx = []
            self.S.wyy = []
            self.S.wzz = []

            for x in sym.D3Q19.basis[1:]:
                if x.dot(x) == 1:
                    self.S.wi.append(Rational(1, 6))

                    if abs(x[0]) == 1:
                        self.S.wxx.append(Rational(5, 12))
                    else:
                        self.S.wxx.append(-Rational(1, 3))

                    if abs(x[1]) == 1:
                        self.S.wyy.append(Rational(5, 12))
                    else:
                        self.S.wyy.append(-Rational(1, 3))

                    if abs(x[2]) == 1:
                        self.S.wzz.append(Rational(5, 12))
                    else:
                        self.S.wzz.append(-Rational(1, 3))

                elif x.dot(x) == 2:
                    self.S.wi.append(Rational(1, 12))

                    if abs(x[0]) == 1:
                        self.S.wxx.append(-Rational(1, 24))
                    else:
                        self.S.wxx.append(Rational(1, 12))

                    if abs(x[1]) == 1:
                        self.S.wyy.append(-Rational(1, 24))
                    else:
                        self.S.wyy.append(Rational(1, 12))

                    if abs(x[2]) == 1:
                        self.S.wzz.append(-Rational(1, 24))
                    else:
                        self.S.wzz.append(Rational(1, 12))
        else:
            self.S.wxy = [x[0]*x[1]*Rational(1, 4) for x in sym.D2Q9.basis[1:]]
            self.S.wyz = [0] * 9
            self.S.wxz = [0] * 9
            self.S.wzz = [0] * 9
            self.S.wi = []
            self.S.wxx = []
            self.S.wyy = []

            for x in sym.D2Q9.basis[1:]:
                if x.dot(x) == 1:
                    self.S.wi.append(Rational(1, 3))

                    if abs(x[0]) == 1:
                        self.S.wxx.append(Rational(1, 3))
                    else:
                        self.S.wxx.append(-Rational(1, 6))

                    if abs(x[1]) == 1:
                        self.S.wyy.append(Rational(1, 3))
                    else:
                        self.S.wyy.append(-Rational(1, 6))
                else:
                    self.S.wi.append(Rational(1, 12))
                    self.S.wxx.append(-Rational(1, 24))
                    self.S.wyy.append(-Rational(1, 24))


    def _init_fields(self, need_dist):
        super(BinaryFluidFreeEnergy, self)._init_fields(need_dist)
        self.vis.add_field((lambda: self.rho + self.phi, lambda: self.rho - self.phi), 'density')


class ShanChenBinary(BinaryFluidBase):
    @property
    def constants(self):
        return [('SCG', self.options.G)]

    def __init__(self, geo_class, options=[], args=None, defaults=None):
        super(ShanChenBinary, self).__init__(geo_class, options, args, defaults)
        self.equilibrium, self.equilibrium_vars = sym.bgk_equilibrium(self.grid)
        eq2, _ = sym.bgk_equilibrium(self.grid, self.S.phi, self.S.phi)
        self.equilibrium.append(eq2[0])
        self.add_force_coupling(0, 1, 'SCG')

    def _init_fields(self, need_dist):
        super(ShanChenBinary, self)._init_fields(need_dist)
        self.vis.add_field((lambda: self.rho, lambda: self.phi), 'density')

    def _add_options(self, parser, lb_group):
        super(ShanChenBinary, self)._add_options(parser, lb_group)

        lb_group.add_option('--G', dest='G',
            help='Shan-Chen interaction strength', action='store', type='float',
            default=1.0)
        lb_group.add_option('--tau_phi', dest='tau_phi', help='relaxation time for the phi field',
                            action='store', type='float', default=1.0)
        return None

    def _update_ctx(self, ctx):
        super(ShanChenBinary, self)._update_ctx(ctx)
        ctx['grids'] = [self.grid, self.grid]
        ctx['tau_phi'] = self.options.tau_phi
        ctx['simtype'] = 'shan-chen'
        ctx['sc_pseudopotential'] = 'sc_ppot_lin'

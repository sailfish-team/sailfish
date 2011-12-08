"""Classes for binary fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import numpy as np
from sailfish import sym, util
from sailfish.lb_base import LBSim


class LBBinaryFluidBase(LBSim):
    """Simulates a binary fluid."""

    kernel_file = 'binary_fluid.mako'

    def __init__(self, config):
        super(LBBinaryFluidBase, self).__init__(config)
        grid = util.get_grid_from_config(config)

        if grid is None:
            raise util.GridError('Invalid grid selected: {0}'.format(config.grid))

        self.grids = [grid, grid]
        self._prepare_symbols()

    @property
    def grid(self):
        """Grid with the highest connectivity (Q)."""
        return self.grids[0]

    @classmethod
    def add_options(cls, group, dim):
        LBSim.add_options(group, dim)
        group.add_argument('--tau_phi', type=float, default=1.0,
                help='relaxation time for the phase field')

        grids = [x.__name__ for x in sym.KNOWN_GRIDS if x.dim == dim]
        group.add_argument('--grid', help='LB grid', type=str,
                choices=grids, default=grids[0])

    def init_fields(self, runner):
        self.phi = runner.make_scalar_field(name='phi', async=True)
        self.rho = runner.make_scalar_field(name='rho', async=True)
        self.v = runner.make_vector_field(name='v', async=True)

        if self.grid.dim == 2:
            self.vx, self.vy = self.v
            runner.add_visualization_field(
                    lambda: np.square(self.vx) + np.square(self.vy),
                    name='v^2')
        else:
            self.vx, self.vy, self.vz = self.v
            runner.add_visualization_field(
                    lambda: np.square(self.vx) + np.square(self.vy) +
                    np.square(self.vz), name='v^2')

    # FIXME
    def get_pbc_kernels(self, runner):
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)

        kernels = []
        for i in range(0, 3):
            kernels.append(runner.get_kernel(
                'ApplyPeriodicBoundaryConditions', [gpu_dist1a, np.uint32(i)], 'Pi'))
        for i in range(0, 3):
            kernels.append(runner.get_kernel(
                'ApplyPeriodicBoundaryConditions', [gpu_dist1b, np.uint32(i)], 'Pi'))

        return kernels

    # FIXME
    def get_compute_kernels(self, runner, full_output, bulk):
        cnp_args1n = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_dist2a,
                      self.gpu_dist2b, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [np.uint32(0)]
        cnp_args1s = [self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_dist2a,
                      self.gpu_dist2b, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [np.uint32(1)]
        cnp_args2n = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_dist2b,
                      self.gpu_dist2a, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [np.uint32(0)]
        cnp_args2s = [self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_dist2b,
                      self.gpu_dist2a, self.gpu_rho, self.gpu_phi] + self.gpu_velocity + [np.uint32(1)]

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

    def initial_conditions(self, runner):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_phi = runner.gpu_field(self.phi)
        gpu_v = runner.gpu_field(self.v)

        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)

        args1 = [gpu_dist1a, gpu_dist2a] + gpu_v + [gpu_rho, gpu_phi]
        args2 = [gpu_dist1b, gpu_dist2b] + gpu_v + [gpu_rho, gpu_phi]

        runner.exec_kernel('SetInitialConditions', args1, 'P'*len(args1))
        runner.exec_kernel('SetInitialConditions', args2, 'P'*len(args2))

    def update_context(self, ctx):
        super(LBBinaryFluidBase, self).update_context(ctx)
        ctx['tau_phi'] = self.config.tau_phi
        ctx['bgk_equilibrium'] = self.equilibrium
        ctx['bgk_equilibrium_vars'] = self.equilibrium_vars

    # FIXME
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

    def _prepare_symbols(self):
        self.S.alias('phi', self.S.g1m0)


class LBBinaryFluidFreeEnergy(LBBinaryFluidBase):

    def __init__(self, config):
        super(LBBinaryFluidFreeEnergy, self).__init__(config)
        self.equilibrium, self.equilibrium_vars = sym.free_energy_binary_liquid_equilibrium(self)

    @property
    def constants(self):
        return [('Gamma', self.options.Gamma), ('A', self.options.A), ('kappa', self.options.kappa),
                ('tau_a', self.options.tau_a), ('tau_b', self.options.tau_b)]

    @classmethod
    def add_options(cls, group, dim):
        LBBinaryFluidBase.add_options(group, dim)

        group.add_argument('--bc_wall_grad_phase', type=float, default=0.0,
                help='gradient of the phase field at the wall; '
                    'this determines the wetting properties')
        group.add_argument('--bc_wall_grad_order', type=int, default=2,
                choices=[1, 2],
                help='order of the gradient stencil used for the '
                    'wetting boundary condition at the walls; valid values are 1 and 2')
        group.add_argument('--Gamma', type=float, default=0.5, help='Gamma parameter')
        group.add_argument('--kappa', type=float, default=0.5, help='kappa parameter')
        group.add_argument('--A', type=float, default=0.5, help='A parameter')
        group.add_argument('--tau_a', type=float, default=1.0,
                help='relaxation time for the A component')
        group.add_argument('--tau_b', type=float, default=1.0,
                help='relaxation time for the B component')

    def update_context(self, ctx):
        super(LBBinaryFluidFreeEnergy, self).update_context(ctx)
        ctx['simtype'] = 'free-energy'
        ctx['bc_wall_grad_phase'] = self.config.bc_wall_grad_phase
        ctx['bc_wall_grad_order'] = self.config.bc_wall_grad_order

    def _prepare_symbols(self):
        """Creates additional symbols and coefficients for the free-energy binary liquid model."""
        super(LBBinaryFluidFreeEnergy, self)._prepare_symbols()
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

    # FIXME
    #def _init_fields(self, need_dist):
    #    super(BinaryFluidFreeEnergy, self)._init_fields(need_dist)
    #    self.vis.add_field((lambda: self.rho + self.phi, lambda: self.rho - self.phi), 'density')


class LBShanChenBinary(LBBinaryFluidBase):

    def __init__(self, config):
        super(LBShanChenBinary, self).__init__(config)
        self.equilibrium, self.equilibrium_vars = sym.bgk_equilibrium(self.grid)
        eq2, _ = sym.bgk_equilibrium(self.grid, self.S.phi, self.S.phi)
        self.equilibrium.append(eq2[0])
        self.add_force_coupling(0, 1, 'SCG')

    @property
    def constants(self):
        return [('SCG', self.config.G)]

    @classmethod
    def add_options(cls, group, dim):
        LBBinaryFluidBase.add_options(group, dim)

        group.add_argument('--visc', type=float, default=1.0, help='numerical viscosity')
        group.add_argument('--G', type=float, default=1.0,
                help='Shan-Chen interaction strenght constant')

    def update_context(self, ctx):
        super(LBShanChenBinary, self).update_context(ctx)
        ctx['simtype'] = 'shan-chen'
        ctx['sc_pseudopotential'] = 'sc_ppot_lin'
        ctx['tau'] = (6.0 * self.config.visc + 1.0)/2.0
        ctx['visc'] = self.config.visc

"""Classes for binary fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import numpy as np
from sailfish import block_runner, sym, util
from sailfish.lb_base import LBSim, ScalarField, VectorField


class LBBinaryFluidBase(LBSim):
    """Base class for binary fluid simulations."""

    subdomain_runner = block_runnner.NNBlockRunner
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

    @classmethod
    def visualization_fields(cls, dim):
        if dim == 2:
            return [ScalarField('v^2',
                    expr=lambda f: np.square(f['vx']) + np.square(f['vy']))]
        else:
            return [ScalarField('v^2',
                    expr=lambda f: np.square(f['vx']) + np.square(f['vy']) +
                        np.square(f['vz']))]

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

    def get_compute_kernels(self, runner, full_output, bulk):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_phi = runner.gpu_field(self.phi)
        gpu_v = runner.gpu_field(self.v)
        gpu_map = runner.gpu_geo_map()

        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)

        args1 = [gpu_map, gpu_dist1a, gpu_dist1b, gpu_dist2a, gpu_dist2b,
                gpu_rho, gpu_phi] + gpu_v
        args2 = [gpu_map, gpu_dist1b, gpu_dist1a, gpu_dist2b, gpu_dist2a,
                gpu_rho, gpu_phi] + gpu_v

        options = 0
        if full_output:
            options |= 1
        if bulk:
            options |= 2

        args1.append(np.uint32(options))
        args2.append(np.uint32(options))

        macro_args1 = [gpu_map, gpu_dist1a, gpu_dist2a, gpu_rho, gpu_phi]
        macro_args2 = [gpu_map, gpu_dist1b, gpu_dist2b, gpu_rho, gpu_phi]

        macro_kernels = [
            runner.get_kernel('PrepareMacroFields', macro_args1,
                'P' * len(macro_args1)),
            runner.get_kernel('PrepareMacroFields', macro_args2,
                'P' * len(macro_args2))]

        sim_kernels = [
            runner.get_kernel('CollideAndPropagate', args1,
                'P'*(len(args1)-1)+'i')),
            runner.get_kernel('CollideAndPropagate', args2,
                'P'*(len(args2)-1)+'i'))]
        return zip(macro_kernels, sim_kernels)

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
    """Binary fluid mixture using the free-energy model."""

    def __init__(self, config):
        super(LBBinaryFluidFreeEnergy, self).__init__(config)
        self.equilibrium, self.equilibrium_vars = sym.free_energy_binary_liquid_equilibrium(self)

    def constants(self):
        ret = super(LBBinaryFluidFreeEnergy, self).constants()
        ret['Gamma'] = self.options.Gamma
        ret['A'] = self.options.A
        ret['kappa'] = self.options.kappa
        ret['tau_a'] = self.options.tau_a
        ret['tau_b'] = self.options.tau_b
        return ret

    @classmethod
    def fields(cls):
        return [ScalarField('rho'), ScalarField('phi', need_nn=True), VectorField('v')]

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


class LBBinaryFluidShanChen(LBBinaryFluidBase):
    """Binary fluid mixture using the Shan-Chen model."""

    def __init__(self, config):
        super(LBBinaryFluidShanChen, self).__init__(config)
        self.equilibrium, self.equilibrium_vars = sym.bgk_equilibrium(self.grid)
        eq2, _ = sym.bgk_equilibrium(self.grid, self.S.phi, self.S.phi)
        self.equilibrium.append(eq2[0])
        self.add_force_coupling(0, 1, 'SCG')

    def constants(self):
        ret = super(LBBinaryFluidShanChen, self).constants()
        ret['SCG'] = self.config.G
        return ret

    @classmethod
    def fields(cls):
        return [ScalarField('rho', need_nn=True),
                ScalarField('phi', need_nn=True),
                VectorField('v')]

    @classmethod
    def add_options(cls, group, dim):
        LBBinaryFluidBase.add_options(group, dim)

        group.add_argument('--visc', type=float, default=1.0, help='numerical viscosity')
        group.add_argument('--G', type=float, default=1.0,
                help='Shan-Chen interaction strenght constant')

    def update_context(self, ctx):
        super(LBBinaryFluidShanChen, self).update_context(ctx)
        ctx['simtype'] = 'shan-chen'
        ctx['sc_pseudopotential'] = 'sc_ppot_lin'
        ctx['tau'] = (6.0 * self.config.visc + 1.0)/2.0
        ctx['visc'] = self.config.visc

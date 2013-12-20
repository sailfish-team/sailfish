"""Classes for binary fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict
from functools import partial
import numpy as np
from sailfish import subdomain_runner, sym, sym_equilibrium
from sailfish.lb_base import LBSim, LBForcedSim, ScalarField, VectorField, KernelPair
from sailfish.lb_single import MacroKernels


class LBBinaryFluidBase(LBSim):
    """Base class for binary fluid simulations."""

    subdomain_runner = subdomain_runner.NNSubdomainRunner
    kernel_file = 'binary_fluid.mako'
    nonlocality = 1

    def __init__(self, config):
        super(LBBinaryFluidBase, self).__init__(config)
        self.grids.append(self.grid)
        self._prepare_symbols()

    @classmethod
    def add_options(cls, group, dim):
        LBSim.add_options(group, dim)
        group.add_argument('--tau_phi', type=float, default=1.0,
                help='relaxation time for the phase field')

    @classmethod
    def visualization_fields(cls, dim):
        if dim == 2:
            return [ScalarField('v^2',
                    expr=lambda f: np.square(f['vx']) + np.square(f['vy']))]
        else:
            return [ScalarField('v^2',
                    expr=lambda f: np.square(f['vx']) + np.square(f['vy']) +
                        np.square(f['vz']))]

    def get_pbc_kernels(self, runner):
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)

        # grid type (primary, secondary) -> axis -> kernels
        dist_kernels = defaultdict(lambda: defaultdict(list))
        macro_kernels = defaultdict(lambda: defaultdict(list))

        for i in range(0, 3):
            dist_kernels[0][i] = [
                    runner.get_kernel(
                        'ApplyPeriodicBoundaryConditions', [gpu_dist1a,
                            np.uint32(i)], 'Pi'),
                    runner.get_kernel(
                        'ApplyPeriodicBoundaryConditions', [gpu_dist2a,
                            np.uint32(i)], 'Pi')]

            for field_pair in self._scalar_fields:
                if not field_pair.abstract.need_nn:
                    continue
                macro_kernels[0][i].append(
                        runner.get_kernel('ApplyMacroPeriodicBoundaryConditions',
                            [runner.gpu_field(field_pair.buffer), np.uint32(i)], 'Pi'))

        if self.config.access_pattern == 'AB':
            gpu_dist1 = gpu_dist1b
            gpu_dist2 = gpu_dist2b
            kernel = 'ApplyPeriodicBoundaryConditions'
        else:
            gpu_dist1 = gpu_dist1a
            gpu_dist2 = gpu_dist2a
            kernel = 'ApplyPeriodicBoundaryConditionsWithSwap'

        for i in range(0, 3):
            dist_kernels[1][i] = [
                    runner.get_kernel(kernel, [gpu_dist1, np.uint32(i)], 'Pi'),
                    runner.get_kernel(kernel, [gpu_dist2, np.uint32(i)], 'Pi')]

            # This is the same as above -- for macroscopic fields, there is no
            # distinction between primary and secondary buffers.
            for field_pair in self._scalar_fields:
                if not field_pair.abstract.need_nn:
                    continue
                macro_kernels[1][i].append(
                        runner.get_kernel('ApplyMacroPeriodicBoundaryConditions',
                            [runner.gpu_field(field_pair.buffer), np.uint32(i)], 'Pi'))

        ret = MacroKernels(macro=macro_kernels, distributions=dist_kernels)
        return ret

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
        if self.config.access_pattern == 'AB':
            runner.exec_kernel('SetInitialConditions', args2, 'P'*len(args2))

    def update_context(self, ctx):
        super(LBBinaryFluidBase, self).update_context(ctx)
        ctx['tau_phi'] = self.config.tau_phi
        ctx['model'] = 'bgk'

    def _prepare_symbols(self):
        self.S.alias('phi', self.S.g1m0)


class LBBinaryFluidFreeEnergy(LBBinaryFluidBase):
    """Binary fluid mixture using the free-energy model."""

    equilibria = (sym_equilibrium.free_energy_equilibrium_fluid,
                  sym_equilibrium.free_energy_equilibrium_order_param)

    def constants(self):
        ret = super(LBBinaryFluidFreeEnergy, self).constants()
        ret['Gamma'] = self.config.Gamma
        ret['A'] = self.config.A
        ret['kappa'] = self.config.kappa
        ret['tau_a'] = self.config.tau_a
        ret['tau_b'] = self.config.tau_b
        return ret

    @classmethod
    def fields(cls):
        return [ScalarField('rho'), ScalarField('phi', need_nn=True),
                VectorField('v'), ScalarField('phi_laplacian')]

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
        group.add_argument('--model', type=str, choices=['bgk', 'mrt'],
                default='bgk', help='LB model to use')

    def update_context(self, ctx):
        super(LBBinaryFluidFreeEnergy, self).update_context(ctx)
        ctx['simtype'] = 'free-energy'
        ctx['bc_wall_grad_phase'] = self.config.bc_wall_grad_phase
        ctx['bc_wall_grad_order'] = self.config.bc_wall_grad_order
        ctx['model'] = self.config.model

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

    def get_compute_kernels(self, runner, full_output, bulk):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_phi = runner.gpu_field(self.phi)
        gpu_lap = runner.gpu_field(self.phi_laplacian)
        gpu_v = runner.gpu_field(self.v)
        gpu_map = runner.gpu_geo_map()

        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)

        options = 0
        if full_output:
            options |= 1
        if bulk:
            options |= 2

        if hasattr(self, '_force_term_for_eq') and self._force_term_for_eq.get(1) == 0:
            phi_args = [gpu_rho, gpu_phi]
        else:
            phi_args = [gpu_phi]

        options = np.uint32(options)
        # Primary.
        args1a = ([gpu_map, gpu_dist1a, gpu_dist1b, gpu_rho, gpu_phi] +
                  gpu_v + [gpu_lap, options])
        args1b = ([gpu_map, gpu_dist2a, gpu_dist2b] + phi_args +
                  gpu_v + [gpu_lap, options])
        # Secondary.
        args2a = ([gpu_map, gpu_dist1b, gpu_dist1a, gpu_rho, gpu_phi] +
                  gpu_v + [gpu_lap, options])
        args2b = ([gpu_map, gpu_dist2b, gpu_dist2a] + phi_args +
                  gpu_v + [gpu_lap, options])

        macro_args1 = [gpu_map, gpu_dist1a, gpu_dist2a, gpu_rho, gpu_phi,
                       options]
        macro_args2 = [gpu_map, gpu_dist1b, gpu_dist2b, gpu_rho, gpu_phi,
                       options]

        args_a_signature = 'P' * (len(args1a) - 1) + 'i'
        args_b_signature = 'P' * (len(args1b) - 1) + 'i'
        macro_signature = 'P' * (len(macro_args1) - 1) + 'i'

        if runner.gpu_scratch_space is not None:
            macro_args1.append(runner.gpu_scratch_space)
            macro_args2.append(runner.gpu_scratch_space)
            macro_signature += 'P'

            args1a.append(runner.gpu_scratch_space)
            args2a.append(runner.gpu_scratch_space)
            args1b.append(runner.gpu_scratch_space)
            args2b.append(runner.gpu_scratch_space)
            args_a_signature += 'P'
            args_b_signature += 'P'

        macro = runner.get_kernel('FreeEnergyPrepareMacroFields', macro_args1,
                                  macro_signature,
                                  needs_iteration=self.config.needs_iteration_num)

        if self.config.access_pattern == 'AB':
            macro_secondary = runner.get_kernel('FreeEnergyPrepareMacroFields',
                                                macro_args2,
                                                macro_signature,
                                                needs_iteration=self.config.needs_iteration_num)
            macro_pair = KernelPair(macro, macro_secondary)
        else:
            macro_pair = KernelPair(macro, macro)

        # Note: these two kernels need to be executed in order.
        primary = [
            runner.get_kernel('FreeEnergyCollideAndPropagateFluid', args1a,
                              args_a_signature,
                              needs_iteration=self.config.needs_iteration_num),
            runner.get_kernel('FreeEnergyCollideAndPropagateOrderParam', args1b,
                              args_b_signature,
                              needs_iteration=self.config.needs_iteration_num)
        ]

        if self.config.access_pattern == 'AB':
            secondary = [
                runner.get_kernel('FreeEnergyCollideAndPropagateFluid', args2a,
                                  args_a_signature,
                                  needs_iteration=self.config.needs_iteration_num),
                runner.get_kernel('FreeEnergyCollideAndPropagateOrderParam',
                                  args2b,
                                  args_b_signature,
                                  needs_iteration=self.config.needs_iteration_num)
            ]
            sim_pair = KernelPair(primary, secondary)
        else:
            sim_pair = KernelPair(primary, primary)

        return zip(macro_pair, sim_pair)


class LBBinaryFluidShanChen(LBBinaryFluidBase, LBForcedSim):
    """Binary fluid mixture using the Shan-Chen model."""

    equilibria = (sym_equilibrium.bgk_equilibrium,
                  partial(sym_equilibrium.bgk_equilibrium, rho=sym.S.phi,
                          rho0=sym.S.phi))

    def __init__(self, config):
        super(LBBinaryFluidShanChen, self).__init__(config)
        self.add_force_coupling(0, 0, 'G11')
        self.add_force_coupling(0, 1, 'G12')
        self.add_force_coupling(1, 0, 'G21')
        self.add_force_coupling(1, 1, 'G22')

    def constants(self):
        ret = super(LBBinaryFluidShanChen, self).constants()
        ret['G11'] = self.config.G11
        ret['G12'] = self.config.G12
        ret['G21'] = self.config.G12
        ret['G22'] = self.config.G22
        return ret

    @classmethod
    def fields(cls):
        return [ScalarField('rho', need_nn=True),
                ScalarField('phi', need_nn=True),
                VectorField('v')]

    @classmethod
    def add_options(cls, group, dim):
        LBForcedSim.add_options(group, dim)
        LBBinaryFluidBase.add_options(group, dim)

        group.add_argument('--visc', type=float, default=1.0, help='numerical viscosity')
        group.add_argument('--G11', type=float, default=0.0,
                help='Shan-Chen component 1 self-interaction strength constant')
        group.add_argument('--G12', type=float, default=0.0,
                help='Shan-Chen component 1<->2 interaction strength constant')
        group.add_argument('--G22', type=float, default=0.0,
                help='Shan-Chen component 2 self-interaction strength constant')
        group.add_argument('--sc_potential', type=str,
                choices=sym.SHAN_CHEN_POTENTIALS, default='linear',
                help='Shan-Chen pseudopotential function to use')

    def update_context(self, ctx):
        super(LBBinaryFluidShanChen, self).update_context(ctx)
        ctx['simtype'] = 'shan-chen'
        ctx['sc_potential'] = self.config.sc_potential
        ctx['tau'] = sym.relaxation_time(self.config.visc)
        ctx['visc'] = self.config.visc

    def get_compute_kernels(self, runner, full_output, bulk):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_phi = runner.gpu_field(self.phi)
        gpu_v = runner.gpu_field(self.v)
        gpu_map = runner.gpu_geo_map()

        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)

        options = 0
        if full_output:
            options |= 1
        if bulk:
            options |= 2

        options = np.uint32(options)
        # Primary.
        args1a = ([gpu_map, gpu_dist1a, gpu_dist1b, gpu_rho, gpu_phi] +
                  gpu_v + [options])
        args1b = ([gpu_map, gpu_dist2a, gpu_dist2b, gpu_rho, gpu_phi] +
                  gpu_v + [options])
        # Secondary.
        args2a = ([gpu_map, gpu_dist1b, gpu_dist1a, gpu_rho, gpu_phi] +
                  gpu_v + [options])
        args2b = ([gpu_map, gpu_dist2b, gpu_dist2a, gpu_rho, gpu_phi] +
                  gpu_v + [options])

        macro_args1 = ([gpu_map, gpu_dist1a, gpu_dist2a, gpu_rho, gpu_phi] +
                       gpu_v + [options])
        macro_args2 = ([gpu_map, gpu_dist1b, gpu_dist2b, gpu_rho, gpu_phi] +
                       gpu_v + [options])

        args_a_signature = 'P' * (len(args1a) - 1) + 'i'
        args_b_signature = 'P' * (len(args1b) - 1) + 'i'
        macro_signature = 'P' * (len(macro_args1) - 1) + 'i'

        if runner.gpu_scratch_space is not None:
            macro_args1.append(runner.gpu_scratch_space)
            macro_args2.append(runner.gpu_scratch_space)
            macro_signature += 'P'

            args1a.append(runner.gpu_scratch_space)
            args2a.append(runner.gpu_scratch_space)
            args1b.append(runner.gpu_scratch_space)
            args2b.append(runner.gpu_scratch_space)
            args_a_signature += 'P'
            args_b_signature += 'P'

        macro = runner.get_kernel('ShanChenPrepareMacroFields', macro_args1,
                                  macro_signature,
                                  needs_iteration=self.config.needs_iteration_num)

        if self.config.access_pattern == 'AB':
            macro_secondary = runner.get_kernel('ShanChenPrepareMacroFields', macro_args2,
                                                macro_signature,
                                                needs_iteration=self.config.needs_iteration_num)
            macro_pair = KernelPair(macro, macro_secondary)
        else:
            macro_pair = KernelPair(macro, macro)

        # TODO(michalj): These kernels can actually run in parallel.
        primary = [
            runner.get_kernel('ShanChenCollideAndPropagate0', args1a,
                              args_a_signature,
                              needs_iteration=self.config.needs_iteration_num),
            runner.get_kernel('ShanChenCollideAndPropagate1', args1b,
                              args_b_signature,
                              needs_iteration=self.config.needs_iteration_num)
        ]

        if self.config.access_pattern == 'AB':
            secondary = [
                runner.get_kernel('ShanChenCollideAndPropagate0', args2a,
                                  args_a_signature,
                                  needs_iteration=self.config.needs_iteration_num),
                runner.get_kernel('ShanChenCollideAndPropagate1', args2b,
                                  args_b_signature,
                                  needs_iteration=self.config.needs_iteration_num)
            ]
            sim_pair = KernelPair(primary, secondary)
        else:
            sim_pair = KernelPair(primary, primary)

        return zip(macro_pair, sim_pair)

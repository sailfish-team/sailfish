"""Classes for multi fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict
from functools import partial
from itertools import product
import numpy as np
from sailfish import subdomain_runner, sym, sym_equilibrium
from sailfish.lb_base import LBSim, LBForcedSim, ScalarField, VectorField, KernelPair

class LBMultiFluidBase(LBSim):
    """Base class for multiple fluid simulations."""

    subdomain_runner = subdomain_runner.NNSubdomainRunner
    kernel_file = 'multi_fluid.mako'
    nonlocality = 1

    def __init__(self, config):
        super(LBMultiFluidBase, self).__init__(config)
        for c in range(1, self.config.lat_nc):
            self.grids.append(self.grid)
        self._prepare_symbols()

    @classmethod
    def add_options(cls, group, dim):
        LBSim.add_options(group, dim)

        group.add_argument('--lat_nc', type=int, default=2, help='number of components')

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
        gpu_distsA = {c: runner.gpu_dist(c, 0) for c in range(0, self.config.lat_nc)}
        gpu_distsB = {c: runner.gpu_dist(c, 1) for c in range(0, self.config.lat_nc)}

        # grid type (primary, secondary) -> axis -> kernels
        dist_kernels = defaultdict(lambda: defaultdict(list))
        macro_kernels = defaultdict(lambda: defaultdict(list))
        if self.config.node_addressing == 'indirect':
            args = [runner.gpu_indirect_address()]
            signature = 'PPi'
        else:
            args = []
            signature = 'Pi'

        for i in range(0, 3):
            dist_kernels[0][i] = [ 
                    runner.get_kernel('ApplyPeriodicBoundaryConditions',
                        args + [gpu_distsA[c], np.uint32(i)], signature) 
                    for c in range(0, self.config.lat_nc) ]

            for field_pair in self._scalar_fields:
                if not field_pair.abstract.need_nn:
                    continue
                macro_kernels[0][i].append(
                        runner.get_kernel('ApplyMacroPeriodicBoundaryConditions',
                                          args + [runner.gpu_field(field_pair.buffer), np.uint32(i)],
                                          signature))

        if self.config.access_pattern == 'AB':
            gpu_dists = {c: gpu_distsB[c] for c in range(0, self.config.lat_nc)}
            kernel = 'ApplyPeriodicBoundaryConditions'
        else:
            gpu_dists = {c: gpu_distsA[c] for c in range(0, self.config.lat_nc)}
            kernel = 'ApplyPeriodicBoundaryConditionsWithSwap'

        for i in range(0, 3):
            if self.config.node_addressing == 'indirect':
                args2 = [runner.gpu_indirect_address()]
                sig2 = 'PPi'
            else:
                args2 = []
                sig2 = 'Pi'
            dist_kernels[1][i] = [
                    runner.get_kernel(kernel, args2 + [gpu_dists[c], np.uint32(i)], sig2) 
                    for c in range(0, self.config.lat_nc) ]

            # This is the same as above -- for macroscopic fields, there is no
            # distinction between primary and secondary buffers.
            for field_pair in self._scalar_fields:
                if not field_pair.abstract.need_nn:
                    continue
                macro_kernels[1][i].append(
                        runner.get_kernel('ApplyMacroPeriodicBoundaryConditions',
                                          args + [runner.gpu_field(field_pair.buffer), np.uint32(i)],
                                          signature))

        ret = subdomain_runner.MacroKernels(macro=macro_kernels, distributions=dist_kernels)
        return ret

    def initial_conditions(self, runner):
        gpu_rho = {}
        if 'sc_potential' in self.config.__dict__: #TODO(nlooije): fix this dirty hack
            for c in range(0, self.config.lat_nc):
                gpu_rho[c] = runner.gpu_field(eval('self.g{}m0'.format(c)))
        else:
            gpu_rho[0] = runner.gpu_field(self.rho)
            gpu_rho[1] = runner.gpu_field(self.phi)
        gpu_v = runner.gpu_field(self.v)
        gpu_map = runner.gpu_geo_map()

        gpu_distsA = {c: runner.gpu_dist(c, 0) for c in range(0, self.config.lat_nc)}
        gpu_distsB = {c: runner.gpu_dist(c, 1) for c in range(0, self.config.lat_nc)}

        args1 = ([gpu_map] + [gpu_distsA[c] for c in range(0, self.config.lat_nc)] + 
                    gpu_v + [gpu_rho[c] for c in range(0, self.config.lat_nc)])
        args2 = ([gpu_map] + [gpu_distsB[c] for c in range(0, self.config.lat_nc)] + 
                    gpu_v + [gpu_rho[c] for c in range(0, self.config.lat_nc)])

        if self.config.node_addressing == 'indirect':
            gpu_nodes = runner.gpu_indirect_address()
            args1 = [gpu_nodes] + args1
            args2 = [gpu_nodes] + args2

        runner.exec_kernel('SetInitialConditions', args1, 'P'*len(args1))
        if self.config.access_pattern == 'AB':
            runner.exec_kernel('SetInitialConditions', args2, 'P'*len(args2))

    def update_context(self, ctx):
        super(LBMultiFluidBase, self).update_context(ctx)
        ctx['lat_nc'] = self.config.lat_nc
        ctx['model'] = 'bgk'

    def _prepare_symbols(self):
        pass

class LBBinaryFluidFreeEnergy(LBMultiFluidBase):
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
        ret['tau_phi'] = self.config.tau_phi
        return ret

    @classmethod
    def fields(cls, grids):
        return [ScalarField('rho'), ScalarField('phi', need_nn=True),
                VectorField('v'), ScalarField('phi_laplacian')]

    @classmethod
    def add_options(cls, group, dim):
        LBMultiFluidBase.add_options(group, dim)

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
        group.add_argument('--tau_phi', type=float, default=1.0,
                help='relaxation time for the phase field')
        group.add_argument('--model', type=str, choices=['bgk', 'mrt'],
                default='bgk', help='LB model to use')

    def update_context(self, ctx):
        super(LBBinaryFluidFreeEnergy, self).update_context(ctx)
        ctx['simtype'] = 'free-energy'
        ctx['tau_phi'] = self.config.tau_phi
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

        if self.config.node_addressing == 'indirect':
            gpu_nodes = runner.gpu_indirect_address()
            args1a = [gpu_nodes] + args1a
            args1b = [gpu_nodes] + args1b
            args2a = [gpu_nodes] + args2a
            args2b = [gpu_nodes] + args2b
            macro_args1 = [gpu_nodes] + macro_args1
            macro_args2 = [gpu_nodes] + macro_args2

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


class LBMultiFluidShanChen(LBMultiFluidBase, LBForcedSim):
    """Multiple fluid mixture using the Shan-Chen model."""

    equilibria = tuple(partial(sym_equilibrium.bgk_equilibrium,
                                rho=eval('sym.S.g{}m0'.format(c)),
                                rho_0=eval('sym.S.g{}m0'.format(c))) for c in range(0, 9))

    def __init__(self, config):
        super(LBMultiFluidShanChen, self).__init__(config)
        for c1, c2 in product(range(0, config.lat_nc), repeat=2):
            self.add_force_coupling(c1,c2,'G{}{}'.format(c1,c2))

    def constants(self):
        ret = super(LBMultiFluidShanChen, self).constants()
        for c1, c2 in product(range(0, self.config.lat_nc), repeat=2):
            ret['G{}{}'.format(c1,c2)] = eval(('self.config.G{0}{1}' if c1 <= c2 
                                          else 'self.config.G{1}{0}').format(c1,c2))
        return ret

    @classmethod
    def fields(cls, grids):
        return [ScalarField('g{}m0'.format(grid_num), need_nn=True) 
                for grid_num in range(0, len(grids))] + [VectorField('v')]

    @classmethod
    def add_options(cls, group, dim):
        LBForcedSim.add_options(group, dim)
        LBMultiFluidBase.add_options(group, dim)

        for c1 in range(0, 9):
            group.add_argument('--visc{}'.format(c1), type=float, default=1.0/6.0, 
                    help='numerical viscosity for component {}'.format(c1))
            group.add_argument('--tau{}'.format(c1), type=float, default=1.0,
                    help='relaxation time for component {}'.format(c1))
            for c2 in range(0, 9):
                helpname = ('{} self-interaction'.format(c1) if c1 == c2 
                                else '{}<->{} interaction'.format(c1,c2))
                group.add_argument('--G{}{}'.format(c1,c2), type=float, default=0.0, 
                        help='Shan-Chen component {} strength constant'.format(helpname))
        group.add_argument('--sc_potential', type=str,
                choices=sym.SHAN_CHEN_POTENTIALS, default='linear',
                help='Shan-Chen pseudopotential function to use')

    def update_context(self, ctx):
        super(LBMultiFluidShanChen, self).update_context(ctx)
        ctx['simtype'] = 'shan-chen'
        for c in range(0, self.config.lat_nc):
            ctx['tau{}'.format(c)] =  sym.relaxation_time(eval('self.config.visc{}'.format(c)))
            ctx['visc{}'] = eval('self.config.visc{}'.format(c))
        ctx['sc_potential'] = self.config.sc_potential

    def _prepare_symbols(self):
        super(LBMultiFluidShanChen, self)._prepare_symbols()

    def get_compute_kernels(self, runner, full_output, bulk):
        gpu_rho = {}
        for c in range(0, self.config.lat_nc):
            gpu_rho[c] = runner.gpu_field(eval('self.g{}m0'.format(c)))
        gpu_v = runner.gpu_field(self.v)
        gpu_map = runner.gpu_geo_map()

        gpu_distsA = {c: runner.gpu_dist(c, 0) for c in range(0, self.config.lat_nc)}
        gpu_distsB = {c: runner.gpu_dist(c, 1) for c in range(0, self.config.lat_nc)}

        options = 0
        if full_output:
            options |= 1
        if bulk:
            options |= 2

        options = np.uint32(options)
        args1i = {c1: ([gpu_map] + [gpu_distsA[c1], gpu_distsB[c1]] +       # Primary
                     [gpu_rho[c2] for c2 in range(0, self.config.lat_nc)] + 
                     gpu_v + [options]) for c1 in range(0, self.config.lat_nc)}
        args2i = {c1: ([gpu_map] + [gpu_distsB[c1], gpu_distsA[c1]] +       # Secondary
                     [gpu_rho[c2] for c2 in range(0, self.config.lat_nc)] + 
                     gpu_v + [options]) for c1 in range(0, self.config.lat_nc)}

        macro_args1 = ([gpu_map] + [gpu_distsA[c] for c in range(0, self.config.lat_nc)] + 
                       [gpu_rho[c] for c in range(0, self.config.lat_nc)] + 
                       gpu_v + [options])
        macro_args2 = ([gpu_map] + [gpu_distsB[c] for c in range(0, self.config.lat_nc)] +
                       [gpu_rho[c] for c in range(0, self.config.lat_nc)] + 
                       gpu_v + [options])

        if self.config.node_addressing == 'indirect':
            gpu_nodes = runner.gpu_indirect_address()
            for c in range(0, self.config.lat_nc):
                args1i[c] = [gpu_nodes] + args1i[c]
                args2i[c] = [gpu_nodes] + args2i[c]
            macro_args1 = [gpu_nodes] + macro_args1
            macro_args2 = [gpu_nodes] + macro_args2

        args_i_signature = {c: 'P' * (len(args1i[c]) - 1) + 'i' for c in range(0, self.config.lat_nc)}
        macro_signature = 'P' * (len(macro_args1) - 1) + 'i'

        if runner.gpu_scratch_space is not None:
            macro_args1.append(runner.gpu_scratch_space)
            macro_args2.append(runner.gpu_scratch_space)
            macro_signature += 'P'

            for c in range(0, self.config.lat_nc):
                args1i[c].append(runner.gpu_scratch_space)
                args2i[c].append(runner.gpu_scratch_space)
                args_i_signature[c] += 'P'

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
            runner.get_kernel('ShanChenCollideAndPropagate{}'.format(c), 
                              args1i[c], args_i_signature[c],
                              needs_iteration=self.config.needs_iteration_num)
            for c in range(0, self.config.lat_nc)
        ]

        if self.config.access_pattern == 'AB':
            secondary = [
                runner.get_kernel('ShanChenCollideAndPropagate{}'.format(c), 
                                  args2i[c], args_i_signature[c],
                                  needs_iteration=self.config.needs_iteration_num)
                for c in range(0, self.config.lat_nc)
            ]
            sim_pair = KernelPair(primary, secondary)
        else:
            sim_pair = KernelPair(primary, primary)

        return zip(macro_pair, sim_pair)

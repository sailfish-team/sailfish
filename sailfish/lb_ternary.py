"""Classes for ternary fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict
from functools import partial
import numpy as np
from sailfish import subdomain_runner, sym, sym_equilibrium
from sailfish.lb_base import LBSim, LBForcedSim, ScalarField, VectorField, KernelPair


class LBTernaryFluidBase(LBSim):
    """Base class for ternary fluid simulations."""

    subdomain_runner = subdomain_runner.NNSubdomainRunner
    kernel_file = 'models/lb_ternary_fluid.mako'
    nonlocality = 1

    def __init__(self, config):
        super(LBTernaryFluidBase, self).__init__(config)
        self.grids.append(self.grid)
        self.grids.append(self.grid)
        self._prepare_symbols()

    @classmethod
    def add_options(cls, group, dim):
        LBSim.add_options(group, dim)
        group.add_argument('--tau_phi', type=float, default=1.0,
                help='relaxation time for the phase field 1 ')
        group.add_argument('--tau_theta', type=float, default=1.0,
                help='relaxation time for the phase field 2')

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
        gpu_dist3a = runner.gpu_dist(2, 0)
        gpu_dist3b = runner.gpu_dist(2, 1)

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
                    runner.get_kernel(
                        'ApplyPeriodicBoundaryConditions',
                        args + [gpu_dist1a, np.uint32(i)], signature),
                    runner.get_kernel(
                        'ApplyPeriodicBoundaryConditions',
                        args + [gpu_dist2a, np.uint32(i)], signature),
                    runner.get_kernel(
                        'ApplyPeriodicBoundaryConditions',
                        args + [gpu_dist3a, np.uint32(i)], signature)]

            for field_pair in self._scalar_fields:
                if not field_pair.abstract.need_nn:
                    continue
                macro_kernels[0][i].append(
                        runner.get_kernel('ApplyMacroPeriodicBoundaryConditions',
                                          args + [runner.gpu_field(field_pair.buffer), np.uint32(i)],
                                          signature))

        if self.config.access_pattern == 'AB':
            gpu_dist1 = gpu_dist1b
            gpu_dist2 = gpu_dist2b
            gpu_dist3 = gpu_dist3b
            kernel = 'ApplyPeriodicBoundaryConditions'
        else:
            gpu_dist1 = gpu_dist1a
            gpu_dist2 = gpu_dist2a
            gpu_dist3 = gpu_dist3a
            kernel = 'ApplyPeriodicBoundaryConditionsWithSwap'

        for i in range(0, 3):
            if self.config.node_addressing == 'indirect':
                args2 = [runner.gpu_indirect_address()]
                sig2 = 'PPi'
            else:
                args2 = []
                sig2 = 'Pi'
            dist_kernels[1][i] = [
                    runner.get_kernel(kernel, args2 + [gpu_dist1, np.uint32(i)], sig2),
                    runner.get_kernel(kernel, args2 + [gpu_dist2, np.uint32(i)], sig2),
                    runner.get_kernel(kernel, args2 + [gpu_dist3, np.uint32(i)], sig2)]

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
        gpu_rho = runner.gpu_field(self.rho)
        gpu_phi = runner.gpu_field(self.phi)
        gpu_theta = runner.gpu_field(self.theta)
        gpu_v = runner.gpu_field(self.v)
        gpu_map = runner.gpu_geo_map()

        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)
        gpu_dist3a = runner.gpu_dist(2, 0)
        gpu_dist3b = runner.gpu_dist(2, 1)

        args1 = [gpu_map, gpu_dist1a, gpu_dist2a, gpu_dist3a] + gpu_v + [gpu_rho, gpu_phi, gpu_theta]
        args2 = [gpu_map, gpu_dist1b, gpu_dist2b, gpu_dist3b] + gpu_v + [gpu_rho, gpu_phi, gpu_theta]

        if self.config.node_addressing == 'indirect':
            gpu_nodes = runner.gpu_indirect_address()
            args1 = [gpu_nodes] + args1
            args2 = [gpu_nodes] + args2

        runner.exec_kernel('SetInitialConditions', args1, 'P'*len(args1))
        if self.config.access_pattern == 'AB':
            runner.exec_kernel('SetInitialConditions', args2, 'P'*len(args2))

    def update_context(self, ctx):
        super(LBTernaryFluidBase, self).update_context(ctx)
        ctx['tau_phi'] = self.config.tau_phi
        ctx['tau_theta'] = self.config.tau_theta
        ctx['model'] = 'bgk'

    def _prepare_symbols(self):
        self.S.alias('phi', self.S.g1m0)
        self.S.alias('theta', self.S.g2m0)

class LBTernaryFluidShanChen(LBTernaryFluidBase, LBForcedSim):
    """Ternary fluid mixture using the Shan-Chen model."""

    equilibria = (sym_equilibrium.bgk_equilibrium,
                  partial(sym_equilibrium.bgk_equilibrium, rho=sym.S.phi, rho0=sym.S.phi),
                  partial(sym_equilibrium.bgk_equilibrium, rho=sym.S.theta, rho0=sym.S.theta)
                  )

    def __init__(self, config):
        super(LBTernaryFluidShanChen, self).__init__(config)
        self.add_force_coupling(0, 0, 'G11')
        self.add_force_coupling(0, 1, 'G12')
        self.add_force_coupling(0, 2, 'G13')
        self.add_force_coupling(1, 0, 'G21')
        self.add_force_coupling(1, 1, 'G22')
        self.add_force_coupling(1, 2, 'G23')
        self.add_force_coupling(2, 0, 'G31')
        self.add_force_coupling(2, 1, 'G32')
        self.add_force_coupling(2, 2, 'G33')

    def constants(self):
        ret = super(LBTernaryFluidShanChen, self).constants()
        ret['G11'] = self.config.G11
        ret['G12'] = self.config.G12
        ret['G13'] = self.config.G13
        ret['G21'] = self.config.G12
        ret['G22'] = self.config.G22
        ret['G23'] = self.config.G23
        ret['G31'] = self.config.G13
        ret['G32'] = self.config.G23
        ret['G33'] = self.config.G33
        return ret

    @classmethod
    def fields(cls):
        return [ScalarField('rho', need_nn=True),
                ScalarField('phi', need_nn=True),
                ScalarField('theta', need_nn=True),
                VectorField('v')]

    @classmethod
    def add_options(cls, group, dim):
        LBForcedSim.add_options(group, dim)
        LBTernaryFluidBase.add_options(group, dim)

        group.add_argument('--visc', type=float, default=1.0, help='numerical viscosity')
        group.add_argument('--G11', type=float, default=0.0,
                help='Shan-Chen component 1 self-interaction strength constant')
        group.add_argument('--G12', type=float, default=0.0,
                help='Shan-Chen component 1<->2 interaction strength constant')
        group.add_argument('--G13', type=float, default=0.0,
                help='Shan-Chen component 1<->3 interaction strength constant')
        group.add_argument('--G22', type=float, default=0.0,
                help='Shan-Chen component 2 self-interaction strength constant')
        group.add_argument('--G23', type=float, default=0.0,
                help='Shan-Chen component 2<->3 interaction strength constant')
        group.add_argument('--G33', type=float, default=0.0,
                help='Shan-Chen component 3 self-interaction strength constant')
        group.add_argument('--sc_potential', type=str,
                choices=sym.SHAN_CHEN_POTENTIALS, default='linear',
                help='Shan-Chen pseudopotential function to use')

    def update_context(self, ctx):
        super(LBTernaryFluidShanChen, self).update_context(ctx)
        ctx['simtype'] = 'shan-chen'
        ctx['sc_potential'] = self.config.sc_potential
        ctx['tau'] = sym.relaxation_time(self.config.visc)
        ctx['visc'] = self.config.visc

    def get_compute_kernels(self, runner, full_output, bulk):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_phi = runner.gpu_field(self.phi)
        gpu_theta = runner.gpu_field(self.theta)
        gpu_v = runner.gpu_field(self.v)
        gpu_map = runner.gpu_geo_map()

        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_dist2a = runner.gpu_dist(1, 0)
        gpu_dist2b = runner.gpu_dist(1, 1)
        gpu_dist3a = runner.gpu_dist(2, 0)
        gpu_dist3b = runner.gpu_dist(2, 1)

        options = 0
        if full_output:
            options |= 1
        if bulk:
            options |= 2

        options = np.uint32(options)
        # Primary.
        args1a = ([gpu_map, gpu_dist1a, gpu_dist1b, gpu_rho, gpu_phi, gpu_theta] +
                  gpu_v + [options])
        args1b = ([gpu_map, gpu_dist2a, gpu_dist2b, gpu_rho, gpu_phi, gpu_theta] +
                  gpu_v + [options])
        args1c = ([gpu_map, gpu_dist3a, gpu_dist3b, gpu_rho, gpu_phi, gpu_theta] +
                  gpu_v + [options])
        # Secondary.
        args2a = ([gpu_map, gpu_dist1b, gpu_dist1a, gpu_rho, gpu_phi, gpu_theta] +
                  gpu_v + [options])
        args2b = ([gpu_map, gpu_dist2b, gpu_dist2a, gpu_rho, gpu_phi, gpu_theta] +
                  gpu_v + [options])
        args2c = ([gpu_map, gpu_dist3b, gpu_dist3a, gpu_rho, gpu_phi, gpu_theta] +
                  gpu_v + [options])

        macro_args1 = ([gpu_map, gpu_dist1a, gpu_dist2a, gpu_dist3a, gpu_rho, gpu_phi, gpu_theta] +
                       gpu_v + [options])
        macro_args2 = ([gpu_map, gpu_dist1b, gpu_dist2b, gpu_dist3b, gpu_rho, gpu_phi, gpu_theta] +
                       gpu_v + [options])

        if self.config.node_addressing == 'indirect':
            gpu_nodes = runner.gpu_indirect_address()
            args1a = [gpu_nodes] + args1a
            args1b = [gpu_nodes] + args1b
            args1c = [gpu_nodes] + args1c
            args2a = [gpu_nodes] + args2a
            args2b = [gpu_nodes] + args2b
            args2c = [gpu_nodes] + args2c
            macro_args1 = [gpu_nodes] + macro_args1
            macro_args2 = [gpu_nodes] + macro_args2

        args_a_signature = 'P' * (len(args1a) - 1) + 'i'
        args_b_signature = 'P' * (len(args1b) - 1) + 'i'
        args_c_signature = 'P' * (len(args1c) - 1) + 'i'
        macro_signature = 'P' * (len(macro_args1) - 1) + 'i'

        if runner.gpu_scratch_space is not None:
            macro_args1.append(runner.gpu_scratch_space)
            macro_args2.append(runner.gpu_scratch_space)
            macro_signature += 'P'

            args1a.append(runner.gpu_scratch_space)
            args2a.append(runner.gpu_scratch_space)
            args1b.append(runner.gpu_scratch_space)
            args2b.append(runner.gpu_scratch_space)
            args1c.append(runner.gpu_scratch_space)
            args2C.append(runner.gpu_scratch_space)
            args_a_signature += 'P'
            args_b_signature += 'P'
            args_c_signature += 'P'

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
                              needs_iteration=self.config.needs_iteration_num),
            runner.get_kernel('ShanChenCollideAndPropagate2', args1c,
                              args_c_signature,
                              needs_iteration=self.config.needs_iteration_num)
        ]

        if self.config.access_pattern == 'AB':
            secondary = [
                runner.get_kernel('ShanChenCollideAndPropagate0', args2a,
                                  args_a_signature,
                                  needs_iteration=self.config.needs_iteration_num),
                runner.get_kernel('ShanChenCollideAndPropagate1', args2b,
                                  args_b_signature,
                                  needs_iteration=self.config.needs_iteration_num),
                runner.get_kernel('ShanChenCollideAndPropagate2', args2c,
                                  args_c_signature,
                                  needs_iteration=self.config.needs_iteration_num),
            ]
            sim_pair = KernelPair(primary, secondary)
        else:
            sim_pair = KernelPair(primary, primary)

        return zip(macro_pair, sim_pair)

"""Classes for single fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPLv3'

from collections import defaultdict, namedtuple
import numpy as np

from sailfish import subdomain_runner, sym, sym_equilibrium
from sailfish.lb_base import LBSim, LBForcedSim, ScalarField, VectorField, KernelPair


MacroKernels = namedtuple('MacroKernels', 'distributions macro')


class LBFluidSim(LBSim):
    """Simulates a single fluid."""

    kernel_file = "single_fluid.mako"
    alpha_output = False
    equilibria = sym_equilibrium.bgk_equilibrium,

    @classmethod
    def add_options(cls, group, dim):
        LBSim.add_options(group, dim)

        group.add_argument('--visc', type=float, default=1.0, help='numerical viscosity')
        group.add_argument('--incompressible',
                action='store_true', default=False,
                help='use the incompressible model of Luo and He')
        group.add_argument('--regularized',
                           action='store_true', default=False,
                           help='Apply the regularization procedure prior to '
                           'the collision step.')
        group.add_argument('--entropic_equilibrium',
                           action='store_true', default=False,
                           help='Use the equilibrium in product form instead '
                           'of the standard LBGK equilibrium.')
        group.add_argument('--model', help='LB collision model to use',
                type=str, choices=['bgk', 'mrt', 'elbm'],
                default='bgk')
        group.add_argument('--subgrid', default='none', type=str,
                choices=['none', 'les-smagorinsky'],
                help='subgrid model to use')
        group.add_argument('--smagorinsky_const',
                help='Smagorinsky constant', type=float, default=0.1)

    def update_context(self, ctx):
        super(LBFluidSim, self).update_context(ctx)
        if self.config.model == 'elbm':
            ctx['tau'] = self.config.visc / self.grid.cssq
        else:
            ctx['tau'] = sym.relaxation_time(self.config.visc)
        ctx['visc'] = self.config.visc
        ctx['model'] = self.config.model
        ctx['simtype'] = 'lbm'
        ctx['subgrid'] = self.config.subgrid
        ctx['smagorinsky_const'] = self.config.smagorinsky_const
        ctx['entropy_tolerance'] = 1e-6 if self.config.precision == 'single' else 1e-10
        ctx['alpha_output'] = self.alpha_output
        ctx['regularized'] = self.config.regularized

    def initial_conditions(self, runner):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_v = runner.gpu_field(self.v)
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_map = runner.gpu_geo_map()

        args1 = [gpu_dist1a] + gpu_v + [gpu_rho, gpu_map]
        args2 = [gpu_dist1b] + gpu_v + [gpu_rho, gpu_map]
        if runner.gpu_scratch_space is not None:
            args1.append(runner.gpu_scratch_space)
            args2.append(runner.gpu_scratch_space)

        runner.exec_kernel('SetInitialConditions', args1, 'P'*len(args1))

        if self.config.access_pattern == 'AB':
            runner.exec_kernel('SetInitialConditions', args2, 'P'*len(args2))

    def get_compute_kernels(self, runner, full_output, bulk):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_v = runner.gpu_field(self.v)
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)
        gpu_map = runner.gpu_geo_map()

        args1 = [gpu_map, gpu_dist1a, gpu_dist1b, gpu_rho] + gpu_v
        args2 = [gpu_map, gpu_dist1b, gpu_dist1a, gpu_rho] + gpu_v

        options = 0
        if full_output:
            options |= 1
        if bulk:
            options |= 2

        args1.append(np.uint32(options))
        args2.append(np.uint32(options))

        signature = 'P' * (len(args1) - 1) + 'i'

        if runner.gpu_scratch_space is not None:
            args1.append(runner.gpu_scratch_space)
            args2.append(runner.gpu_scratch_space)
            signature += 'P'

        # Alpha field for the entropic LBM.
        if self.alpha_output:
            args1.append(runner.gpu_field(self.alpha))
            args2.append(runner.gpu_field(self.alpha))
            signature += 'P'

        cnp_primary = runner.get_kernel(
            'CollideAndPropagate', args1, signature,
            needs_iteration=self.config.needs_iteration_num)

        if self.config.access_pattern == 'AB':
            secondary_args = args2 if self.config.access_pattern == 'AB' else args1
            cnp_secondary = runner.get_kernel(
                'CollideAndPropagate', secondary_args, signature,
                needs_iteration=self.config.needs_iteration_num)
            return KernelPair([cnp_primary], [cnp_secondary])
        else:
            return KernelPair([cnp_primary], [cnp_primary])

    def get_pbc_kernels(self, runner):
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)

        # grid type (primary, secondary) -> axis -> kernels
        kernels = defaultdict(lambda: defaultdict(list))

        # One kernel per axis, per grid.  Kernels for 3D are always prepared,
        # and in 2D simulations the kernel for the Z dimension is simply
        # ignored.
        for i in range(0, 3):
            kernels[0][i] = [runner.get_kernel(
                'ApplyPeriodicBoundaryConditions', [gpu_dist1a, np.uint32(i)],
                'Pi')]

        if self.config.access_pattern == 'AB':
            gpu_dist = gpu_dist1b
            kernel = 'ApplyPeriodicBoundaryConditions'
        else:
            gpu_dist = gpu_dist1a
            kernel = 'ApplyPeriodicBoundaryConditionsWithSwap'

        for i in range(0, 3):
            kernels[1][i] = [runner.get_kernel(
                kernel, [gpu_dist, np.uint32(i)],
                'Pi')]

        return kernels

    @classmethod
    def fields(cls):
        return [ScalarField('rho'), VectorField('v')]

    @classmethod
    def visualization_fields(cls, dim):
        if dim == 2:
            return [ScalarField('v^2',
                    expr=lambda f: np.square(f['vx']) + np.square(f['vy']))]
        else:
            return [ScalarField('v^2',
                    expr=lambda f: np.square(f['vx']) + np.square(f['vy']) +
                        np.square(f['vz']))]


class LBEntropicFluidSim(LBFluidSim):
    """LBFluidSim with alpha field tracking.

    The alpha field is 2.0 in areas where the fluid dynamics is fully resolved.
    alpha < 2.0 means that the flow field is smoothened, while alpha > 2.0
    indicates enhancement of flow perturbation."""

    alpha_output = True

    @classmethod
    def modify_config(cls, config):
        config.model = 'elbm'

    @classmethod
    def fields(cls):
        return [ScalarField('rho'), VectorField('v'),
                ScalarField('alpha', init=2.0)]


class LBFreeSurface(LBFluidSim):
    """Free surface lattice Boltzmann model."""

    equilibria = sym_equilibrium.shallow_water_equilibrium,

    @classmethod
    def modify_config(cls, config):
        config.grid = 'D2Q9'
        config.model = 'bgk'

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        group.add_argument('--gravity', type=float, default=0.001,
            help='gravitational acceleration')

    def constants(self):
        ret = super(LBFreeSurface, self).constants()
        ret['gravity'] = self.config.gravity
        return ret


class LBSingleFluidShanChen(LBFluidSim, LBForcedSim):
    """Single-phase Shan-Chen model."""

    nonlocality = 1
    subdomain_runner = subdomain_runner.NNSubdomainRunner

    def __init__(self, config):
        super(LBSingleFluidShanChen, self).__init__(config)
        self.add_force_coupling(0, 0, 'SCG')

    @classmethod
    def add_options(cls, group, dim):
        LBFluidSim.add_options(group, dim)
        LBForcedSim.add_options(group, dim)
        group.add_argument('--G', type=float, default=1.0,
                help='Shan-Chen interaction strength constant')
        group.add_argument('--sc_potential', type=str,
                choices=sym.SHAN_CHEN_POTENTIALS, default='linear',
                help='Shan-Chen pseudopotential function to use')

    @classmethod
    def fields(cls):
        return [ScalarField('rho', need_nn=True), VectorField('v')]

    def constants(self):
        ret = super(LBSingleFluidShanChen, self).constants()
        ret['SCG'] = self.config.G
        return ret

    def update_context(self, ctx):
        super(LBSingleFluidShanChen, self).update_context(ctx)
        ctx['simtype'] = 'shan-chen'
        ctx['sc_potential'] = self.config.sc_potential
        ctx['visc'] = self.config.visc

    def get_pbc_kernels(self, runner):
        dist_kernels = super(LBSingleFluidShanChen, self).get_pbc_kernels(runner)
        macro_kernels = defaultdict(lambda: defaultdict(list))
        for i in range(0, 3):
            for field_pair in self._scalar_fields:
                if not field_pair.abstract.need_nn:
                    continue
                macro_kernels[0][i].append(
                        runner.get_kernel('ApplyMacroPeriodicBoundaryConditions',
                            [runner.gpu_field(field_pair.buffer), np.uint32(i)], 'Pi'))

        for i in range(0, 3):
            for field_pair in self._scalar_fields:
                if not field_pair.abstract.need_nn:
                    continue
                macro_kernels[1][i].append(
                        runner.get_kernel('ApplyMacroPeriodicBoundaryConditions',
                            [runner.gpu_field(field_pair.buffer), np.uint32(i)], 'Pi'))

        ret = MacroKernels(macro=macro_kernels, distributions=dist_kernels)
        return ret

    def get_compute_kernels(self, runner, full_output, bulk):
        gpu_rho = runner.gpu_field(self.rho)
        gpu_map = runner.gpu_geo_map()

        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)

        options = 0
        if full_output:
            options |= 1
        if bulk:
            options |= 2

        options = np.uint32(options)
        macro_args1 = [gpu_map, gpu_dist1a, gpu_rho, options]
        macro_args2 = [gpu_map, gpu_dist1b, gpu_rho, options]

        signature = 'P' * (len(macro_args1) - 1) + 'i'

        if runner.gpu_scratch_space is not None:
            macro_args1.append(runner.gpu_scratch_space)
            macro_args2.append(runner.gpu_scratch_space)
            signature += 'P'

        macro_kernels = [
            runner.get_kernel('PrepareMacroFields', macro_args1,
                signature,
                needs_iteration=self.config.needs_iteration_num),
            runner.get_kernel('PrepareMacroFields', macro_args2,
                signature,
                needs_iteration=self.config.needs_iteration_num)]

        sim_kernels = super(LBSingleFluidShanChen, self).get_compute_kernels(
                runner, full_output, bulk)

        return zip(macro_kernels, sim_kernels)

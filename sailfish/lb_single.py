"""Classes for single fluid lattice Boltzmann simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPLv3'

import numpy as np

from sailfish import sym, util
from sailfish.lb_base import LBSim

class GridError(Exception):
    pass

class LBForcedSim(LBSim):
    """Adds support for body forces."""

    def __init__(self, config):
        super(LBForcedSim, self).__init__(config)
        self._forces = {}

    # TODO(michalj): Add support for dynamical forces via sympy expressions
    # and for global force fields via numpy arrays.
    def add_body_force(self, force, grid=0, accel=True):
        """Add a constant global force field acting on the fluid.

        Multiple calls to this function will add the value of `force` to any
        previously set value.  Forces and accelerations are processed separately
        and are never mixed in this addition process.

        :param force: n-vector representing the force value
        :param grid: grid number on which this force is acting
        :param accel: if ``True``, the added field is an acceleration field, otherwise
            it is an actual force field
        """
        dim = self.grids[0].dim
        assert len(force) == dim

        # Create an empty force vector.  Use numpy so that we can easily compute
        # sums.
        self._forces.setdefault(grid, {}).setdefault(accel, np.zeros(dim, np.float64))
        a = self._forces[grid][accel] + np.float64(force)
        self._forces[grid][accel] = a

    def update_context(self, ctx):
        super(LBForcedSim, self).update_context(ctx)
        ctx['forces'] = self._forces


class LBFluidSim(LBSim):
    """Simulates a single phase fluid."""

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
        super(LBFluidSim, self).__init__(config)

        grid = util.get_grid_from_config(config)

        if grid is None:
            raise GridError('Invalid grid selected: {0}'.format(config.grid))

        self.grids = [grid]
        self.equilibrium, self.equilibrium_vars = sym.bgk_equilibrium(grid)

    @property
    def grid(self):
        """Grid with the highest connectivity (Q)."""
        return self.grids[0]

    def update_context(self, ctx):
        super(LBFluidSim, self).update_context(ctx)
        ctx['tau'] = (6.0 * self.config.visc + 1.0)/2.0
        ctx['visc'] = self.config.visc
        ctx['model'] = self.config.model
        ctx['loc_names'] = ['gx', 'gy', 'gz']
        ctx['simtype'] = 'lbm'
        ctx['grid'] = self.grids[0]
        ctx['grids'] = self.grids
        ctx['bgk_equilibrium'] = self.equilibrium
        ctx['bgk_equilibrium_vars'] = self.equilibrium_vars

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
        args2 = [gpu_map, gpu_dist1b, gpu_dist1a, gpu_rho] + gpu_v

        if full_output:
            args1.append(np.uint32(1))
            args2.append(np.uint32(1))
        else:
            args1.append(np.uint32(0))
            args2.append(np.uint32(0))

        kernels = []
        kernels.append(runner.get_kernel(
                'CollideAndPropagate', args1, 'P'*(len(args1)-1)+'i'))
        kernels.append(runner.get_kernel(
                'CollideAndPropagate', args2, 'P'*(len(args2)-1)+'i'))
        return kernels

    def get_pbc_kernels(self, runner):
        gpu_dist1a = runner.gpu_dist(0, 0)
        gpu_dist1b = runner.gpu_dist(0, 1)

        kernels = []
        for i in range(0, 3):
            kernels.append(runner.get_kernel(
                'ApplyPeriodicBoundaryConditions', [gpu_dist1a, np.uint32(i)], 'Pi'))
        for i in range(0, 3):
            kernels.append(runner.get_kernel(
                'ApplyPeriodicBoundaryConditions', [gpu_dist1b, np.uint32(i)], 'Pi'))

        return kernels

    def init_fields(self, runner):
        self.rho = runner.make_scalar_field(name='rho')
        self.v = runner.make_vector_field(name='v')
        self.vx, self.vy = self.v
        runner.add_visualization_field(
                lambda: np.square(self.vx) + np.square(self.vy),
                name='v^2')

# TODO(michalj): Port the single-phase Shan-Chen class.

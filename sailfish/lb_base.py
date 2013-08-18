"""Base class for all lattice Boltzman simulations in Sailfish."""

from collections import namedtuple
from sailfish import sym, util
from sailfish import node_type as nt

import numpy as np

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL'


FieldPair = namedtuple('FieldPair', 'abstract buffer')
ForcePair = namedtuple('ForcePair', 'numeric symbolic')
KernelPair = namedtuple('KernelPair', 'primary secondary')

class LBMixIn(object):
    """Provides additional functionliaty to a simulation class.

    Some attributes of objects descentant from this class are processed
    in addition to the corresponding attribution from a simulation class
    (LBSim descentant). These are:
        - aux_code
        - before_main_loop
        - fields
    """


class LBSim(object):
    """Describes a specific type of a lattice Boltzmann simulation."""

    kernel_file = "__TEMPLATE_NOT_SET__"

    #: List of additional template files or inline code fragments to include
    #  in the generated simulation code. The templates here are appended in
    #  order, after the contents of kernel_file above. List entries are
    #  considered to be code fragments if they have at least two lines.
    aux_code = []

    #: Set this to a class implementing the SubdomainRunner interface in order
    #  to use subdomain runner other than default.
    subdomain_runner = None

    #: How many layers of nearest neighbors nodes are required by the model.
    nonlocality = 0

    #: Tuple of functions to call to get a list of expressions for the equilibrium
    #  distribution function. The i-th element represents the EDF for the i-th
    #  grid.
    equilibria = None

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--dt_per_lattice_time_unit',
                help='physical time delta corresponding to one iteration '
                'of the simulation', type=float, default=0.0)
        grids = [x.__name__ for x in sym.KNOWN_GRIDS if x.dim == dim]
        group.add_argument('--grid', help='LB grid', type=str,
                choices=grids, default=grids[0])
        group.add_argument('--access_pattern', type=str, default='AB',
                choices=['AB', 'AA'], help='Lattice access pattern. Valid '
                'values are: AB (two copies of the whole domain in memory,'
                ' faster), AA (single copy of the domain in memory)')
        group.add_argument('--minimize_roundoff', action='store_true',
                           default=False, help='Tries to minimize round-off '
                           'errors by using a model that avoids adding O(1) '
                           'and O(Ma) quantities. This currently only works '
                           'for BGK-like models.')
        group.add_argument('--propagate_on_read', action='store_true',
                           default=False, help='Uses the propagate-on-read '
                           'scheme in which distributions are saved to the '
                           'local node only, and streaming happens implicitly '
                           'by reading data from neighboring nodes.')
        group.add_argument('--nopropagate_with_shuffle', action='store_false',
                           dest='propagate_with_shuffle',
                           default=True, help='Uses the shuffle operation to '
                           'move data within warps if the device supports it.')

    @classmethod
    def modify_config(cls, config):
        pass

    @classmethod
    def update_defaults(cls, defaults):
        pass

    def fields(self):
        return []

    def constants(self):
        """Returns a dict mapping names to values and defining global constants
        for the simulation."""
        return {}

    @property
    def grid(self):
        """Returns a grid object representing the connectivity of the lattice
        used in the simulation.  If the simulation uses more than 1 grid,
        returns the grid with the highest connectivity."""
        return max(self.grids, key=lambda grid: grid.Q)

    @property
    def dim(self):
        return self.grid.dim

    def update_context(self, ctx):
        """Updates the context dicitionary containing variables used for
        code generation."""
        ctx['grid'] = self.grid
        ctx['grids'] = self.grids
        ctx['loc_names'] = ['gx', 'gy', 'gz']
        ctx['constants'] = self.constants()
        ctx['relaxation_enabled'] = self.config.relaxation_enabled
        ctx['propagation_enabled'] = self.config.propagation_enabled
        ctx['dt_per_lattice_time_unit'] = self.config.dt_per_lattice_time_unit
        ctx['access_pattern'] = self.config.access_pattern
        ctx['propagate_on_read'] = self.config.propagate_on_read
        ctx['propagate_with_shuffle'] = self.config.propagate_with_shuffle
        ctx['needs_iteration_num'] = self.config.needs_iteration_num
        ctx['equilibria'] = self.equilibria
        ctx['config'] = self.config

    def init_fields(self, runner):
        suffixes = ['x', 'y', 'z']
        self._scalar_fields = []
        self._vector_fields = []
        self._fields = {}

        sources = [self]
        # Scan for mixin classes adding their own fields.
        for c in self.__class__.mro()[1:]:
            if issubclass(c, LBMixIn) and hasattr(c, 'fields'):
                sources.append(c)

        for src in sources:
            for field in src.fields():
                if type(field) is ScalarField:
                    f = runner.make_scalar_field(name=field.name, async=True,
                                                 gpu_array=field.gpu_array)
                    f[:] = field.init
                    self._scalar_fields.append(FieldPair(field, f))
                elif type(field) is VectorField:
                    f = runner.make_vector_field(name=field.name, async=True,
                                                 gpu_array=field.gpu_array)
                    self._vector_fields.append(FieldPair(field, f))
                    for i in range(0, self.grid.dim):
                        setattr(self, field.name + suffixes[i], f[i])
                setattr(self, field.name, f)
                self._fields[field.name] = FieldPair(field, f)

    def verify_fields(self):
        """Verifies that fields have not accidentally been overridden."""
        for name, field_pair in self._fields.iteritems():
            assert getattr(self, name) is field_pair.buffer,\
                    'Field {0} redefined (probably in initial_conditions())'.format(
                            name)

    def __init__(self, config):
        self.config = config
        self.S = sym.S
        self.iteration = 0
        self.need_sync_flag = False
        self.need_fields_flag = False

        # For use in unit tests only.
        if config is not None:
            grid = util.get_grid_from_config(config)
            if grid is None:
                raise util.GridError('Invalid grid selected: {0}'.format(config.grid))
            self.grids = [grid]

    def get_state(self):
        return {'iteration': self.iteration}

    def set_state(self, state):
        self.iteration = state['iteration']

    def need_output(self):
        """Returns True when data will be required for output,
        based on command line parameters --from --every.

        Called from SubdomainRunner.main().
        """
        if self.config.output_required:
            return ((self.iteration + 1) % self.config.every) == 0 and self.config.from_ <= (self.iteration)
        else:
            return False

    def need_sync_fields(self):
        """Indicates whether computation/transfer of macroscopic fields is requested.

        This is true if either data is to be output after the current step, or
        data was requested e.g. by after_step(). To get fields on the host in
        after_step(), set need_sync_flag to True. To get fields on the GPU only,
        set need_fields_flags to True.

        Called from SubdomainRunner.main().

        Returns:
          tuple of two boolean values; the first is True if synchronization of
          fields to the host is requested; the second is True if computation of
          macroscopic fields is requested.
        """
        need_sync = self.need_sync_flag or self.need_output()
        need_fields = self.need_fields_flag or need_sync
        self.need_fields_flag = False
        self.need_sync_flag = False
        return need_sync, need_fields

    def need_checkpoint(self):
        """Returns True when a checkpoint is requested after the current
        iteration."""

        return ((self.iteration % self.config.checkpoint_every) == 0 and
            self.iteration >= self.config.checkpoint_from)

    def before_main_loop(self, runner):
        """Called from the subdomain runner before entering the main loop
        of the simulation.

        This function can be used to initialize additional buffers and kernels
        that are then going to be used in :func:`after_step`."""
        pass

    def after_step(self, runner):
        """Called from the main loop after the completion of every step."""
        pass

    def get_compute_kernels(self, runner, full_output, bulk):
        """
        :param runner: SubdomainRunner object
        :param full_output: if True, returns kernels that prepare fields for
                visualization or saving into a file
        :param bulk: if True, returns kernels that process the bulk domain,
                otherwise returns kernels that process the subdomain boundary
        """
        return KernelPair(None, None)

    def get_pbc_kernels(self, runner):
        return []

    def get_aux_kernels(self, runner):
        return KernelPair([], [])

    def initial_conditions(self, runner):
        pass

    # TODO(michalj): Restore support for defining visualization fields.
    # TODO(michalj): Restore support for tracer particles.


class LBForcedSim(LBSim):
    """Adds support for body forces.

    This is a mix-in class. When defining a new simulation, inherit
    from another LBSim-based class first.
    """

    def __init__(self, config):
        super(LBForcedSim, self).__init__(config)
        self._force_couplings = {}
        self._force_term_for_eq = {}

        # grid_id -> accel (bool) -> numpy force vector
        self._forces = {}

        # grid_id -> accel (bool) -> list of nt.DynamicValue objects
        self._symbolic_forces = {}

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--force_implementation',
                           type=str, choices=['guo', 'edm', 'velocity_shift'], default='guo',
                           help='Selects how body forces are introduced into '
                           'the simulation. Available choices are: guo and '
                           'EDM (Exact Difference Method')

    def add_body_force(self, force, grid=0, accel=True):
        """Adds a constant global force field acting on the fluid.

        Multiple calls to this function will add the value of `force` to any
        previously set value.  Forces and accelerations are processed separately
        and are never mixed in this addition process.

        :param force: n-vector representing the force value; this can be a
            vector of basic Python types, a numpy vector, or a DynamicValue
            object
        :param grid: grid number on which this force is acting
        :param accel: if ``True``, the added field is an acceleration field, otherwise
            it is an actual force field
        """
        dim = self.grids[0].dim
        assert len(force) == dim

        if isinstance(force, nt.DynamicValue):
            self._symbolic_forces.setdefault(grid, {}).setdefault(accel, []).append(force)
            if force.has_symbols(sym.S.time):
                self.config.time_dependence = True
            if force.has_symbols(sym.S.gx, sym.S.gy, sym.S.gz):
                self.config.space_dependence = True
        else:
            # Create an empty force vector.  Use numpy so that we can easily compute
            # sums.
            self._forces.setdefault(grid, {}).setdefault(accel, np.zeros(dim, np.float64))
            a = self._forces[grid][accel] + np.float64(force)
            self._forces[grid][accel] = a

    def update_context(self, ctx):
        super(LBForcedSim, self).update_context(ctx)
        ctx['forces'] = ForcePair(self._forces, self._symbolic_forces)
        ctx['force_couplings'] = self._force_couplings
        ctx['force_for_eq'] = self._force_term_for_eq
        ctx['force_implementation'] = self.config.force_implementation

    def use_force_for_equilibrium(self, force_grid, target_grid):
        """Makes it possible to use acceleration from force_grid when calculating
        velocity for the equlibrium of target_grid.

        For instance, to use the acceleration from grid 0 in relaxation of
        grid 1, use the parameters (0, 1).

        To disable acceleration on a grid, pass an invalid grid ID in force_grid
        (e.g. None or -1).

        Note: this is currently only supported in the free-energy MRT model.
        The force reassignment will be silently ignored in other models.

        :param force_grid: grid ID from which the acceleration will be used
        :param target_grid: grid ID on which the acceleration will act
        """
        self._force_term_for_eq[target_grid] = force_grid

    def add_force_coupling(self, grid_a, grid_b, const_name):
        """Adds a Shan-Chen type coupling between two lattices.

        :param grid_a: numerical ID of the first lattice
        :param grid_b: numerical ID of the second lattice
        :param const_name: name of the global variable containing the value of the
                coupling constant
        """
        self._force_couplings[(grid_a, grid_b)] = const_name


class Field(object):
    def __init__(self, name, expr=None, need_nn=False, init=0.0, gpu_array=False):
        """
        :param need_nn: if True, the model needs access to this field
            on the neighboring nodes.
        :param init: Initial value. Only used for scalar fields.
        :param gpu_array: if True, a GPUArray wrapper will be automatically
            created
        """
        self.name = name
        self.expr = expr
        self.init = init
        self.need_nn = need_nn
        self.gpu_array = gpu_array

class ScalarField(Field):
    pass

class VectorField(Field):
    pass

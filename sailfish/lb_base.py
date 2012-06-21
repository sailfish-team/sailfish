"""Base class for all lattice Boltzman simulations in Sailfish."""

from collections import namedtuple
from sailfish import sym

import numpy as np

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL'
__version__ = '0.3-alpha1'


FieldPair = namedtuple('FieldPair', 'abstract buffer')


class LBSim(object):
    """Describes a specific type of a lattice Boltzmann simulation."""

    kernel_file = "__TEMPLATE_NOT_SET__"

    #: Set this to a class implementing the BlockRunner interface in order
    #  to use subdomain runner other than default.
    subdomain_runner = None

    #: How many layers of nearest neighbors nodes are required by the model.
    nonlocality = 0

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--dt_per_lattice_time_unit',
                help='physical time delta corresponding to one iteration '
                'of the simulation', type=float, default=0.0)

    @classmethod
    def modify_config(cls, config):
        pass

    @classmethod
    def update_defaults(cls, defaults):
        pass

    def constants(self):
        """Returns a dict mapping names to values and defining global constants
        for the simulation."""
        return {}

    @property
    def grid(self):
        """Returns a grid object representing the connectivity of the lattice
        used in the simulation.  If the simulation uses more than 1 grid,
        returns the grid with the highest connectivity."""
        raise NotImplementedError("grid() should be defined in a subclass")

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
        ctx['dt_per_lattice_time_unit'] = self.config.dt_per_lattice_time_unit

    def init_fields(self, runner):
        suffixes = ['x', 'y', 'z']
        self._scalar_fields = []
        self._vector_fields = []
        self._fields = {}
        for field in self.fields():
            if type(field) is ScalarField:
                f = runner.make_scalar_field(name=field.name, async=True)
                self._scalar_fields.append(FieldPair(field, f))
            elif type(field) is VectorField:
                f = runner.make_vector_field(name=field.name, async=True)
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
        self.S = sym.S()
        self.iteration = 0

    def need_output(self):
        """Returns True when data for macroscopic fields is necessary
        for the current iteration, based on command line parameters
        --from --every.

        Called from SubdomainRunner.main().
        """
        return ((self.iteration + 1) % self.config.every) == 0 and self.config.from_ <= (self.iteration)

    def after_step(self):
        """Called from the main loop after the completion of every step."""
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
        self._forces = {}
        self._force_couplings = {}
        self._force_term_for_eq = {}

    @classmethod
    def add_options(cls, group, dim):
        pass

    # TODO(michalj): Add support for dynamical forces via sympy expressions
    # and for global force fields via numpy arrays.
    def add_body_force(self, force, grid=0, accel=True):
        """Adds a constant global force field acting on the fluid.

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
        ctx['force_couplings'] = self._force_couplings
        ctx['force_for_eq'] = self._force_term_for_eq

    def use_force_for_equilibrium(self, force_grid, target_grid):
        """Makes it possible to use acceleration from force_grid when calculating
        velocity for the equlibrium of target_grid.

        For instance, to use the acceleration from grid 0 in relaxation of
        grid 1, use the parameters (0, 1).

        To disable acceleration on a grid, pass an invalid grid ID in force_grid
        (e.g. None or -1).

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
    def __init__(self, name, expr=None, need_nn=False):
        """
        :param need_nn: if True, the model needs access to this field
            on the neighboring nodes.
        """
        self.name = name
        self.expr = expr
        self.need_nn = need_nn

class ScalarField(Field):
    pass

class VectorField(Field):
    pass

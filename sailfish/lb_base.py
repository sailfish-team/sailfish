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
        pass

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

    # TODO(michalj): Restore support for iter hooks.
    # TODO(michalj): Restore support for defining visualization fields.
    # TODO(michalj): Restore support for tracer particles.

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

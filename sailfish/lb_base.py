"""Base class for all lattice Boltzman simulations in Sailfish."""

from sailfish import sym

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL'
__version__ = '0.3-alpha1'


class LBSim(object):
    """Describes a specific type of a lattice Boltzmann simulation."""

    kernel_file = "__TEMPLATE_NOT_SET__"

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

    @property
    def grid(self):
        """Returns a grid object representing the connectivity of the lattice
        used in the simulation.  If the simulation uses more than 1 grid,
        returns the grid with the highest connectivity."""
        raise NotImplementedError("grid() should be defined in a subclass")

    def update_context(self, ctx):
        """Updates the context dicitionary containing variables used for
        code generation."""
        pass

    def __init__(self, config):
        self.config = config
        self.S = sym.S()
        self.iteration = 0

    # TODO(michalj): Restore support for force couplings.
    # TODO(michalj): Restore support for fields accessed via textures.
    # TODO(michalj): Restore support for iter hooks.
    # TODO(michalj): Restore support for defining visualization fields.
    # TODO(michalj): Restore support for tracer particles.
    # TODO(michalj): Restore support for free-surface LB.

import numpy as np

from sailfish import sym

class LBSim(object):
    """Describes a specific type of a lattice Boltzmann simulation."""

    kernel_file = "__TEMPLATE_NOT_SET__"

    @classmethod
    def add_options(cls, group, dim):
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


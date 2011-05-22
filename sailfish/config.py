"""Classes and function for specifying and processing simulation
configuration."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import argparse

class LBConfig(argparse.Namespace):
    """Specifies the configuration of a LB simulation.

    This class carries all settings, specified programmatically from a script
    or manually via command line parameters.
    """

    def __init__(self, description=None):
        desc = "Sailfish LB simulator."
        if description is not None:
            desc += " " + description
        self._parser = argparse.ArgumentParser(description=description)

        self._parser.add_argument('-q', '--quiet',
                help='reduce verbosity', action='store_true', default=False)
        self._parser.add_argument('-v', '--verbose',
                help='print additional info about the simulation',
                action='store_true', default=False)

    def add_group(self, name):
        return self._parser.add_argument_group(name)

    def set_defaults(self, defaults):
        return self._parser.set_defaults(**defaults)

    def parse(self):
        self._parser.parse_args(namespace=self)

    @property
    def output_required(self):
        return self.output or self.mode == 'visualization'

"""Classes for specifying and processing simulation configuration."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import ConfigParser
import argparse
import os

class LBConfig(argparse.Namespace):
    """Specifies the configuration of a LB simulation.

    This class carries all settings, specified programmatically from a script
    or manually via command line parameters.
    """
    @property
    def output_required(self):
        return self.output or self.mode == 'visualization'


class LBConfigParser(object):
    def __init__(self, description=None):
        desc = "Sailfish LB simulator."
        if description is not None:
            desc += " " + description

        self._parser = argparse.ArgumentParser(description=desc)
        self._parser.add_argument('-q', '--quiet',
                help='reduce verbosity', action='store_true', default=False)
        self._parser.add_argument('-v', '--verbose',
                help='print additional info about the simulation',
                action='store_true', default=False)

        self.config = LBConfig()

    def add_group(self, name):
        return self._parser.add_argument_group(name)

    def set_defaults(self, defaults):
        return self._parser.set_defaults(**defaults)

    def parse(self):
        config = ConfigParser.ConfigParser()
        config.read(['/etc/sailfishrc', os.path.expanduser('~/.sailfishrc'),
                '.sailfishrc'])
        try:
            self._parser.set_defaults(**dict(config.items('main')))
        except ConfigParser.NoSectionError:
            pass
        self._parser.parse_args(namespace=self.config)

        # Additional internal config options, not settable via
        # command line parameters.
        self.config.relaxation_enabled = True
        return self.config


class MachineSpec(object):
    """Declares information about a machine."""

    def __init__(self, host, addr, gpus=[0], iface='eth0', **kwargs):
        """
        :param host: host name (can be an execnet gateway spec)
        :param gpus: list of GPU IDs on which to run
        :parma addr: host address (IP or domain name)
        """
        self.host = host
        self.addr = addr
        self.gpus = gpus
        self.iface = iface
        self.settings = kwargs

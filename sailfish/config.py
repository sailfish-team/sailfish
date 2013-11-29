"""Classes for specifying and processing simulation configuration."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import ConfigParser
import argparse
import os
import re


class LBConfig(argparse.Namespace):
    """Specifies the configuration of a LB simulation.

    This class carries all settings, specified programmatically from a script
    or manually via command line parameters.
    """
    @property
    def output_required(self):
        return self.output or self.mode == 'visualization'

    @property
    def needs_iteration_num(self):
        return self.time_dependence or self.access_pattern == 'AA'


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
        self._parser.add_argument('--silent',
                help='reduce verbosity further', action='store_true',
                                  default=False)

        self.config = LBConfig()

    def add_group(self, name):
        return self._parser.add_argument_group(name)

    def set_defaults(self, defaults):
        for option in defaults.iterkeys():
            assert self._parser.get_default(option) is not None,\
                    'Unknown option "{0}" specified in update_defaults()'.format(option)
        return self._parser.set_defaults(**defaults)

    def parse(self, args, internal_defaults=None):
        config = ConfigParser.ConfigParser()
        config.read(['/etc/sailfishrc', os.path.expanduser('~/.sailfishrc'),
                '.sailfishrc'])

        # Located here for convenience, so that this attribute can be referenced in
        # the symbolic expressions module even for LB models where this option is not
        # supported.
        self.config.incompressible = False
        try:
            self._parser.set_defaults(**dict(config.items('main')))
        except ConfigParser.NoSectionError:
            pass

        if internal_defaults is not None:
            self._parser.set_defaults(**internal_defaults)

        self._parser.parse_args(args=args, namespace=self.config)

        # Additional internal config options, not settable via
        # command line parameters.
        self.config.relaxation_enabled = True
        self.config.propagation_enabled = True

        # Indicates whether the simulation has any DynamicValues which are
        # time-dependent.
        self.config.time_dependence = False

        # Indicates whether the simulation has any DynamicValues which are
        # location-dependent.
        self.config.space_dependence = False
        self.config.unit_test = False
        return self.config


class MachineSpec(object):
    """Declares information about a machine."""

    def __init__(self, host, addr, gpus=[0], iface='eth0', **kwargs):
        """
        :param host: host name (can be a full execnet gateway spec)
        :param addr: host address (IP or domain name); this will be used to
                establish block-block connections
        :param gpus: list of GPU IDs on which to run
        :param iface: network interface on which to listen for remote
                connections

        Additional keyword parameters will be stored in the machine's
        LBConfig instance when a simulation is run.
        """
        self.host = host
        self.addr = addr
        self.gpus = gpus
        self.iface = iface
        self.settings = kwargs

    # TODO(michalj): Optimize this.
    def get_port(self):
        matches = re.search(':(\d+)', self.host)
        if matches is None:
            return -1
        return int(matches.group(1))

    def set_port(self, port):
        curr_port = self.get_port()
        self.host = self.host.replace(str(curr_port), str(port))

    def __repr__(self):
        return 'MachineSpec({0}, {1}, {2}, {3})'.format(
                self.host, self.addr, self.gpus, self.iface)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not self.__eq__(other)

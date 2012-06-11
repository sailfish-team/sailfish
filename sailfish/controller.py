"""Simulation controller."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import cPickle as pickle
import copy
import math
import imp
import logging
import os
import platform
import re
import shutil
import socket
import subprocess
import stat
import sys
import tempfile
import time
from collections import namedtuple, defaultdict
from multiprocessing import Process

import execnet
import zmq
from sailfish import codegen, config, io, util
from sailfish.geo import LBGeometry2D, LBGeometry3D
from sailfish.subdomain import SubdomainSpec3D, SubdomainPair

def _start_machine_master(config, subdomains, lb_class):
    """Starts a machine master process locally."""
    from sailfish.master import LBMachineMaster
    master = LBMachineMaster(config, subdomains, lb_class)
    master.run()


def _start_cluster_machine_master(channel, args, main_script, lb_class_name,
        subdomain_addr_map, iface):
    """Starts a machine master process on a remote host.

    This function is executed by the execnet module.  In order for it to work,
    it cannot depend on any global symbols."""
    import cPickle as pickle
    import os
    import platform
    import sys
    import traceback

    try:
        sys.path.append(os.path.dirname(main_script))
        import imp
        main = imp.load_source('main', main_script)
        for name in dir(main):
            globals()[name] = getattr(main, name)

        lb_class = globals()[lb_class_name]

        from sailfish.master import LBMachineMaster
        import multiprocessing as mp
        pname = 'Master/{0}'.format(platform.node())
        mp.current_process().name = pname
        master = LBMachineMaster(*pickle.loads(args), lb_class=lb_class,
                subdomain_addr_map=subdomain_addr_map, channel=channel,
                iface=iface)
        master.run()
    except Exception:
        # Send any exceptions by to the controller to aid
        # debugging.
        channel.send(traceback.format_exc())

    channel.send('FIN')


# TODO: This is currently a very dumb procedure.  Ideally, we would
# obtain a speed estimate from each node, calculate the amount of work
# per subdomain, and distribute the work taking all this into account.
def split_subdomains_between_nodes(nodes, subdomains):
    """Assigns subdomains to cluster nodes.

    Returns a list of 'nodes' lists of subdomains."""

    total_gpus = sum([len(node.gpus) for node in nodes])
    n = len(subdomains)
    idx = 0

    assignments = []
    for i, node in enumerate(nodes):
        units = int(math.ceil(float(n) * len(node.gpus) / total_gpus))
        assignments.append(subdomains[idx:idx+units])
        idx += units

        if idx >= n:
            break

    # Add any remaining subdomains to the last node.
    assignments[-1].extend(subdomains[idx:])

    return assignments


class GeometryError(Exception):
    pass


class LBGeometryProcessor(object):
    """Transforms a set of SubdomainSpecs into a another set covering the same
    physical domain, but optimized for execution on the available hardware.
    Initializes logical connections between the subdomains based on their
    location."""

    def __init__(self, subdomains, dim, geo):
        """
        :param subdomains: list of SubdomainSpec objects
        """
        self.subdomains = subdomains
        self.dim = dim
        self.geo = geo

    def _annotate(self):
        # Assign IDs to subdomains.  The subdomain ID corresponds to its position
        # in the internal subdomains list.
        for i, subdomain in enumerate(self.subdomains):
            subdomain.id = i

    def _add_pair(self, pair):
        for i, coord in enumerate(pair.virtual.location):
            self._coord_map_list[i][coord].append(pair)

    def _init_lower_coord_map(self, config):
        # List position corresponds to the principal axis (X, Y, Z).  List
        # items are maps from lower coordinate along the specific axis to
        # a list of SubdomainPairs
        self._coord_map_list = [defaultdict(list), defaultdict(list),
                defaultdict(list)]
        for subdomain in self.subdomains:
            self._add_pair(SubdomainPair(subdomain, subdomain))

        periodicity = [config.periodic_x, config.periodic_y]
        if self.dim == 3:
            periodicity.append(config.periodic_z)

        def _pbc_helper(loc, subdomain, axes):
            if not axes:
                return [tuple(loc)]

            ret = []
            axis_done = False
            while axes and not axis_done:
                ax = axes.pop()
                if not periodicity[ax]:
                    continue

                if subdomain.location[ax] == 0:
                    ret.extend(_pbc_helper(list(loc), subdomain, list(axes)))
                    loc[ax] = self.geo.gsize[ax]
                    ret.extend(_pbc_helper(list(loc), subdomain, list(axes)))
                    loc[ax] = 0
                    axis_done = True
                if subdomain.end_location[ax] == self.geo.gsize[ax]:
                    ret.extend(_pbc_helper(list(loc), subdomain, list(axes)))
                    loc[ax] = -subdomain.size[ax]
                    ret.extend(_pbc_helper(list(loc), subdomain, list(axes)))
                    axis_done = True

            return ret

        # Handle PBCs by creating virtual copies of subdomains touching the
        # lower boundary of the global domain.
        done = set()
        for axis in range(self.dim):
            if not periodicity[axis]:
                continue

            for subdomain, _ in self._coord_map_list[axis][0]:
                if subdomain.id in done:
                    continue
                loc = list(subdomain.location)
                loc[axis] = self.geo.gsize[axis]

                locs = set(_pbc_helper(list(subdomain.location), subdomain,
                    range(axis, self.dim)))
                b = subdomain
                locs.remove(b.location)

                for loc in locs:
                    virtual = b.__class__(loc, b.size, b.envelope_size, b._id)
                    pair = SubdomainPair(b, virtual)
                    self._add_pair(pair)

                done.add(b.id)

    def _connect_subdomains(self, config):
        self._init_lower_coord_map(config)
        connected = set()

        # TOOD(michalj): Fix this for multi-grid models.
        grid = util.get_grid_from_config(config)

        def try_connect(subdomain, pair):
            if subdomain.id == pair.real.id:
                return False

            if subdomain.connect(pair, grid):
                connected.add(subdomain.id)
                connected.add(pair.real.id)
                return True

            return False

        periodicity = [config.periodic_x, config.periodic_y]
        if self.dim == 3:
            periodicity.append(config.periodic_z)

        for axis in range(self.dim):
            if not periodicity[axis]:
                continue

            for real, virtual in self._coord_map_list[axis][0]:
                # If the subdomain spans a whole axis of the domain, mark it
                # as locally periodic.
                if real.end_location[axis] == self.geo.gsize[axis]:
                    real.enable_local_periodicity(axis)

        for axis in range(self.dim):
            for subdomain in sorted(self.subdomains, key=lambda x: x.location[axis]):
                higher_coord = subdomain.end_location[axis]
                if higher_coord not in self._coord_map_list[axis]:
                    continue
                for neighbor_candidate in \
                        self._coord_map_list[axis][higher_coord]:
                    try_connect(subdomain, neighbor_candidate)

        # Ensure every subdomain is connected to at least one other subdomain.
        if len(self.subdomains) > 1 and len(connected) != len(self.subdomains):
            raise GeometryError()

    def transform(self, config):
        self._annotate()
        self._connect_subdomains(config)
        return self.subdomains


class LBSimulationController(object):
    """Controls the execution of a LB simulation."""

    def __init__(self, lb_class, lb_geo=None, default_config=None):
        """
        :param lb_class: class describing the simulation, derived from LBSim
        :param lb_geo: class describing the global geometry in terms of
                SubdomainSpec, derived from LBGeometry
        :param default_config: dictionary mapping command line option names
                to their new default values
        """
        self._config_parser = config.LBConfigParser()
        self._lb_class = lb_class

        # Use a default global geometry is one has not been
        # specified explicitly.
        if lb_geo is None:
            if self.dim == 2:
                lb_geo = LBGeometry2D
            else:
                lb_geo = LBGeometry3D

        self._lb_geo = lb_geo
        self._tmpdir = tempfile.mkdtemp()

        group = self._config_parser.add_group('Runtime mode settings')
        group.add_argument('--mode', help='runtime mode', type=str,
            choices=['batch', 'visualization', 'benchmark'], default='batch'),
        group.add_argument('--every',
            help='save/visualize simulation results every N iterations ',
            metavar='N', type=int, default=100)
        group.add_argument('--from', dest='from_',
            help='save/visualize simulation results from N iterations ', metavar='N',
            type=int, default=0)
        group.add_argument('--max_iters',
            help='number of iterations to run; use 0 to run indefinitely',
            type=int, default=0)
        group.add_argument('--output',
            help='save simulation results to FILE', metavar='FILE',
            type=str, default='')
        group.add_argument('--output_format',
            help='output format', type=str,
            choices=io.format_name_to_cls.keys(), default='npy')
        group.add_argument('--backends',
            type=str, default='cuda,opencl',
            help='computational backends to use; multiple backends '
                 'can be separated by a comma')
        group.add_argument('--visualize',
            type=str, default='2d',
            help='visualization engine to use')
        group.add_argument('--gpus', nargs='+', default=0, type=int,
            help='which GPUs to use')
        group.add_argument('--debug_dump_dists', action='store_true',
                default=False, help='dump the contents of the distribution '
                'arrays to files'),
        group.add_argument('--log', type=str, default='',
                help='name of the file to which data is to be logged')
        group.add_argument('--loglevel', type=int, default=logging.INFO,
                help='minimum log level for the file logger')
        group.add_argument('--bulk_boundary_split', type=bool, default=True,
                help='if True, bulk and boundary nodes will be handled '
                'separately (increases parallelism)')
        group.add_argument('--cluster_spec', type=str, default='',
                help='path of a Python module with the cluster specification')
        group.add_argument('--cluster_sync', type=str, default='',
                help='local_path:dest_path; if specified, will send the '
                'contents of "local_path" to "dest_path" on all cluster '
                'machines before starting the simulation.')
        group.add_argument('--cluster_pbs', type=bool, default=True,
                help='If True, standard PBS variables will be used to run '
                'the job in a cluster.')
        group.add_argument('--cluster_pbs_initscript', type=str,
                default='sailfish-init.sh', help='Script to execute on remote '
                'nodes in order to set the environment prior to starting '
                'a machine master.')
        group.add_argument('--cluster_pbs_interface', type=str,
                default='', help='Network interface to use on PBS nodes for '
                'internode communication.')

        group = self._config_parser.add_group('Simulation-specific settings')
        lb_class.add_options(group, self.dim)

        group = self._config_parser.add_group('Geometry settings')
        lb_geo.add_options(group)

        group = self._config_parser.add_group('Code generator options')
        codegen.BlockCodeGenerator.add_options(group)

        # Backend options
        for backend in util.get_backends():
            group = self._config_parser.add_group(
                    "'{0}' backend options".format(backend.name))
            backend.add_options(group)

        # Do not try to import visualization engine modules if we already
        # know that the simulation will be running in batch mode.
        if (default_config is None or 'mode' not in default_config or
            default_config['mode'] == 'visualization'):
            for engine in util.get_visualization_engines():
                group = self._config_parser.add_group(
                        "'{0}' visualization engine".format(engine.name))
                engine.add_options(group)

        # Set default values defined by the simulation-specific class.
        defaults = {}
        lb_class.update_defaults(defaults)
        self._config_parser.set_defaults(defaults)

        if default_config is not None:
            self._config_parser.set_defaults(default_config)

    def __del__(self):
        shutil.rmtree(self._tmpdir)

    @property
    def dim(self):
        """Dimensionality of the simulation: 2 or 3."""
        return self._lb_class.subdomain.dim

    def _init_subdomain_envelope(self, sim, subdomains):
        """Sets the size of the ghost node envelope for all subdomains."""
        envelope_size = sim.nonlocality
        for vec in sim.grid.basis:
            for comp in vec:
                envelope_size = max(sim.nonlocality, abs(comp))

        # Get rid of any Sympy wrapper objects.
        envelope_size = int(envelope_size)

        for subdomain in subdomains:
            subdomain.set_actual_size(envelope_size)

    def _start_cluster_simulation(self, subdomains, cluster=None):
        """Starts a simulation on a cluster of nodes."""

        if cluster is None:
            try:
                cluster = imp.load_source('cluster', self.config.cluster_spec)
            except IOError, e:
                cluster = imp.load_source('cluster',
                        os.path.expanduser('~/.sailfish/{0}'.format(self.config.cluster_spec)))

        self._cluster_gateways = []
        self._node_subdomains = split_subdomains_between_nodes(cluster.nodes, subdomains)

        for _, node in zip(self._node_subdomains, cluster.nodes):
            self._cluster_gateways.append(execnet.makegateway(node.host))

        # Copy files to remote nodes if necessary.
        if self.config.cluster_sync:
            local, dest = self.config.cluster_sync.split(':')
            assert dest[0] != '/', 'Only relative paths are supported on remote nodes.'
            rsync = execnet.RSync(local)
            for gw in self._cluster_gateways:
                rsync.add_target(gw, dest)
            rsync.send()

        subdomain_id_to_addr = {}
        for node_id, subdomains in enumerate(self._node_subdomains):
            for subdomain in subdomains:
                subdomain_id_to_addr[subdomain.id] = cluster.nodes[node_id].addr

        self._cluster_channels = []
        import sys
        for i, (node, gw) in enumerate(zip(cluster.nodes, self._cluster_gateways)):
            # Assign specific GPUs from this node, as defined by the cluster
            # config file.
            node_config = copy.copy(self.config)
            node_config.gpus = cluster.nodes[i].gpus
            for k, v in node.settings.iteritems():
                setattr(node_config, k, v)

            self._cluster_channels.append(
                    gw.remote_exec(_start_cluster_machine_master,
                    args=pickle.dumps((node_config, self._node_subdomains[i])),
                    main_script=sys.argv[0],
                    lb_class_name=self._lb_class.__name__,
                    subdomain_addr_map=subdomain_id_to_addr,
                    iface=node.iface))

        ports = {}
        for channel in self._cluster_channels:
            data = channel.receive()
            # If a string is received, print it to help with debugging.
            if type(data) is str:
                print data
            else:
                ports.update(data)

        for channel in self._cluster_channels:
            channel.send(ports)

    def _start_local_simulation(self, subdomains):
        """Starts a simulation on the local machine."""
        self._simulation_process = Process(target=_start_machine_master,
                    name='Master/{0}'.format(platform.node()),
                    args=(self.config, subdomains, self._lb_class))
        self._simulation_process.start()

    def _start_pbs_handlers(self):
        cluster = util.gpufile_to_clusterspec(os.environ['PBS_GPUFILE'],
                self.config.cluster_pbs_interface)
        self._pbs_handlers = []
        id_string = 'sailfish-%s' % os.getpid()

        def _start_socketserver(addr, port):
            return subprocess.Popen(['pbsdsh', '-h',
                addr, 'sh', '-c',
                ". %s ; python %s/socketserver.py :%s %s" %
                (self.config.cluster_pbs_initscript,
                    os.path.realpath(os.path.dirname(util.__file__)),
                    port, id_string)])

        for node in cluster.nodes:
            port = node.get_port()
            self._pbs_handlers.append(_start_socketserver(node.addr, port))

        def _try_next_port(i, node, still_starting):
            port = node.get_port() + 1
            node.set_port(port)
            print 'retrying node %s:%s...' % (node.host, node.addr)
            self._pbs_handlers[i] = _start_socketserver(node.addr, port)
            still_starting.append((i, node))

        starting_nodes = list(enumerate(cluster.nodes))
        while starting_nodes:
            still_starting = []

            for i, node in starting_nodes:
                if self._pbs_handlers[i].returncode is not None:
                    # Remote process terminated -- try to start again with a
                    # different port.
                    _try_next_port(i, node, still_starting)
                else:
                    try:
                        s = socket.create_connection((node.addr, node.get_port()), timeout=5.0)
                        if s.recv(256) != id_string:
                            _try_next_port(i, node, still_starting)
                        s.close()
                    except (socket.timeout, socket.error):
                        still_starting.append((i, node))
                        continue

            starting_nodes = still_starting
            sys.stdout.flush()
            time.sleep(0.5)

        return cluster

    def _is_pbs_cluster(self):
        return self.config.cluster_pbs and 'PBS_GPUFILE' in os.environ

    def _start_simulation(self, subdomains):
        """Starts a simulation.

        :param subdomains: list of SubdomainSpec objects
        """

        # A PBS implementation with GPU-aware scheduling is required.
        if self._is_pbs_cluster():
            assert self.config.cluster_spec == '', ('Cluster specifications '
                    'are not supported when running under PBS.')

            cluster = self._start_pbs_handlers()
            self._start_cluster_simulation(subdomains, cluster)
        elif self.config.cluster_spec:
            self._start_cluster_simulation(subdomains)
        else:
            self._start_local_simulation(subdomains)

    def _finish_simulation(self, subdomains, summary_receiver):
        timing_infos = []
        min_timings = []
        max_timings = []

        if self.config.cluster_spec or self._is_pbs_cluster():
            if self.config.mode == 'benchmark':
                for ch, node_subdomains in zip(self._cluster_channels, self._node_subdomains):
                    for sub in node_subdomains:
                        ti, min_ti, max_ti = ch.receive()
                        timing_infos.append(util.TimingInfo(*ti))
                        min_timings.append(util.TimingInfo(*min_ti))
                        max_timings.append(util.TimingInfo(*max_ti))

            for ch in self._cluster_channels:
                data = ch.receive()
                assert data == 'FIN'
            for gw in self._cluster_gateways:
                gw.exit()

            if self._is_pbs_cluster():
                for handler in self._pbs_handlers:
                    handler.terminate()
        else:
            if self.config.mode == 'benchmark':
                # Collect timing information from all subdomains.
                for i in range(len(subdomains)):
                    ti, min_ti, max_ti = summary_receiver.recv_pyobj()
                    summary_receiver.send('ack')
                    timing_infos.append(ti)
                    min_timings.append(min_ti)
                    max_timings.append(max_ti)

            self._simulation_process.join()

        if self.config.mode == 'benchmark':
            mlups_total = 0.0
            mlups_comp = 0.0

            for ti in timing_infos:
                subdomain = subdomains[ti.subdomain_id]
                mlups_total += subdomain.num_nodes / ti.total * 1e-6
                mlups_comp += subdomain.num_nodes / ti.comp * 1e-6

            if not self.config.quiet:
                print ('Total MLUPS: eff:{0:.2f}  comp:{1:.2f}'.format(
                        mlups_total, mlups_comp))
            return timing_infos, min_timings, max_timings, subdomains

        return None, None

    def save_subdomain_config(self, subdomains):
        if self.config.output:
            pickle.dump(subdomains,
                    open(io.subdomains_filename(self.config.output), 'w'))

    def run(self, ignore_cmdline=False):
        """Runs a simulation."""

        if ignore_cmdline:
            args = []
        else:
            args = sys.argv[1:]

        self.config = self._config_parser.parse(args)
        self._lb_class.modify_config(self.config)
        self.geo = self._lb_geo(self.config)

        ctx = zmq.Context()
        summary_receiver = ctx.socket(zmq.REP)
        port = summary_receiver.bind_to_random_port('tcp://127.0.0.1')
        self.config._zmq_port = port

        subdomains = self.geo.subdomains()
        assert subdomains is not None, \
                "Make sure the subdomain list is returned in geo_class.subdomains()"
        assert len(subdomains) > 0, \
                "Make sure at least one subdomain is returned in geo_class.subdomains()"

        sim = self._lb_class(self.config)
        self._init_subdomain_envelope(sim, subdomains)

        proc = LBGeometryProcessor(subdomains, self.dim, self.geo)
        subdomains = proc.transform(self.config)
        self.save_subdomain_config(subdomains)

        self._start_simulation(subdomains)
        return self._finish_simulation(subdomains, summary_receiver)

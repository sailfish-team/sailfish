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
from multiprocessing import Process

import execnet
import zmq
from sailfish import codegen, config, io, util
from sailfish.geo import LBGeometry2D, LBGeometry3D


def _start_machine_master(config, blocks, lb_class):
    """Starts a machine master process locally."""
    from sailfish.master import LBMachineMaster
    master = LBMachineMaster(config, blocks, lb_class)
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
    Initializes logical connections between the blocks based on their
    location."""

    def __init__(self, blocks, dim, geo):
        self.blocks = blocks
        self.dim = dim
        self.geo = geo

    def _annotate(self):
        # Assign IDs to blocks.  The block ID corresponds to its position
        # in the internal blocks list.
        for i, block in enumerate(self.blocks):
            block.id = i

    def _init_lower_coord_map(self):
        # List position corresponds to the principal axis (X, Y, Z).  List
        # items are maps from lower coordinate along the specific axis to
        # a list of block IDs.
        self._coord_map_list = [{}, {}, {}]
        for block in self.blocks:
            for i, coord in enumerate(block.location):
                self._coord_map_list[i].setdefault(coord, []).append(block)

    def _connect_blocks(self, config):
        connected = [False] * len(self.blocks)

        # TOOD(michalj): Fix this for multi-grid models.
        grid = util.get_grid_from_config(config)

        def try_connect(block1, block2, geo=None, axis=None):
            if block1.connect(block2, geo, axis, grid):
                connected[block1.id] = True
                connected[block2.id] = True

        for axis in range(self.dim):
            for block in sorted(self.blocks, key=lambda x: x.location[axis]):
                higher_coord = block.location[axis] + block.size[axis]
                if higher_coord not in self._coord_map_list[axis]:
                    continue
                for neighbor_candidate in \
                        self._coord_map_list[axis][higher_coord]:
                    try_connect(block, neighbor_candidate)

        # In case the simulation domain is globally periodic, try to connect
        # the blocks at the lower boundary of the domain along the periodic
        # axis (i.e. coordinate = 0) with blocks which have a boundary at the
        # highest global coordinate (gx, gy, gz).
        if config.periodic_x:
            for block in self._coord_map_list[0][0]:
                # If the block spans the whole X axis of the domain, mark it
                # as locally periodic and do not try to find any neigbor
                # candidates.
                if block.location[0] + block.size[0] == self.geo.gx:
                    block.enable_local_periodicity(0)
                    continue

                # Iterate over all blocks, for each one calculate the location
                # of its top boundary and compare it to the size of the whole
                # simulation domain.
                for x0, candidates in self._coord_map_list[0].iteritems():
                    for candidate in candidates:
                        if (candidate.location[0] + candidate.size[0]
                               == self.geo.gx):
                            try_connect(block, candidate, self.geo, 0)

        if config.periodic_y:
            for block in self._coord_map_list[1][0]:
                if block.location[1] + block.size[1] == self.geo.gy:
                    block.enable_local_periodicity(1)
                    continue

                for y0, candidates in self._coord_map_list[1].iteritems():
                    for candidate in candidates:
                        if (candidate.location[1] + candidate.size[1]
                               == self.geo.gy):
                            try_connect(block, candidate, self.geo, 1)

        if self.dim > 2 and config.periodic_z:
            for block in self._coord_map_list[2][0]:
                if block.location[2] + block.size[2] == self.geo.gz:
                    block.enable_local_periodicity(2)
                    continue

                for z0, candidates in self._coord_map_list[2].iteritems():
                    for candidate in candidates:
                        if (candidate.location[2] + candidate.size[2]
                               == self.geo.gz):
                            try_connect(block, candidate, self.geo, 2)

        # Ensure every block is connected to at least one other block.
        if not all(connected) and len(connected) > 1:
            raise GeometryError()

    def transform(self, config):
        self._annotate()
        self._init_lower_coord_map()
        self._connect_blocks(config)
        return self.blocks


class LBSimulationController(object):
    """Controls the execution of a LB simulation."""

    def __init__(self, lb_class, lb_geo=None, default_config=None):
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

        group = self._config_parser.add_group('Runtime mode settings')
        group.add_argument('--mode', help='runtime mode', type=str,
            choices=['batch', 'visualization', 'benchmark']),
        group.add_argument('--every',
            help='save/visualize simulation results every N iterations ',
            metavar='N', type=int, default=100)
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
                'machines before starting the simulation')

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

    @property
    def dim(self):
        """Dimensionality of the simulation: 2 or 3."""
        return self._lb_class.subdomain.dim

    def _init_block_envelope(self, sim, blocks):
        """Sets the size of the ghost node envelope for all blocks."""
        envelope_size = sim.nonlocality
        for vec in sim.grid.basis:
            for comp in vec:
                envelope_size = max(sim.nonlocality, abs(comp))

        # Get rid of any Sympy wrapper objects.
        envelope_size = int(envelope_size)

        for block in blocks:
            block.set_actual_size(envelope_size)

    def _start_cluster_simulation(self, subdomains):
        """Starts a simulation on a cluster of nodes."""
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

    def _start_simulation(self, subdomains):
        """Starts a simulation.

        :param subdomains: list of SubdomainSpec objects
        """

        if self.config.cluster_spec:
            self._start_cluster_simulation(subdomains)
        else:
            self._start_local_simulation(subdomains)

    def _finish_simulation(self, blocks, summary_receiver):
        timing_infos = []

        if self.config.cluster_spec:
            if self.config.mode == 'benchmark':
                for ch, subdomains in zip(self._cluster_channels, self._node_subdomains):
                    for sub in subdomains:
                        timing_infos.append(util.TimingInfo(*ch.receive()))

            for ch in self._cluster_channels:
                data = ch.receive()
                assert data == 'FIN'
            for gw in self._cluster_gateways:
                gw.exit()
        else:
            if self.config.mode == 'benchmark':
                # Collect timing information from all blocks.
                for i in range(len(blocks)):
                    ti = summary_receiver.recv_pyobj()
                    summary_receiver.send_pyobj('ack')
                    timing_infos.append(ti)

            self._simulation_process.join()

        if self.config.mode == 'benchmark':
            mlups_total = 0.0
            mlups_comp = 0.0

            for ti in timing_infos:
                block = blocks[ti.block_id]
                mlups_total += block.num_nodes / ti.total * 1e-6
                mlups_comp += block.num_nodes / ti.comp * 1e-6

            if not self.config.quiet:
                print ('Total MLUPS: eff:{0:.2f}  comp:{1:.2f}'.format(
                        mlups_total, mlups_comp))
            return timing_infos, blocks

        return None, None


    def run(self):
        """Runs a simulation."""

        self.config = self._config_parser.parse()
        self._lb_class.modify_config(self.config)
        self.geo = self._lb_geo(self.config)

        ctx = zmq.Context()
        summary_receiver = ctx.socket(zmq.REP)
        port = summary_receiver.bind_to_random_port('tcp://127.0.0.1')
        self.config._zmq_port = port

        blocks = self.geo.blocks()
        assert blocks is not None, \
                "Make sure the block list is returned in geo_class.blocks()"
        assert len(blocks) > 0, \
                "Make sure at least one block is returned in geo_class.blocks()"

        sim = self._lb_class(self.config)
        self._init_block_envelope(sim, blocks)

        proc = LBGeometryProcessor(blocks, self.dim, self.geo)
        blocks = proc.transform(self.config)

        self._start_simulation(blocks)
        return self._finish_simulation(blocks, summary_receiver)

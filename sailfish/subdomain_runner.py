"""Code for controlling a single block of a LB simulation."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict, namedtuple
import math
import operator
import os
import numpy as np
import time
import zmq
from sailfish import codegen, util

# Used to hold a reference to a CUDA kernel and a grid on which it is
# to be executed.
KernelGrid = namedtuple('KernelGrid', 'kernel grid')

class MacroConnectionBuffer(object):
    """Contains buffers needed to transfer a single macroscopic field
    between two subdomains."""

    def __init__(self, face, cpair, coll_buf, coll_idx, recv_buf, dist_idx,
            field):
        """
        :param face: face ID
        :param cpair: ConnectionPair, a tuple of two LBConnection objects
        :param coll_buf: GPUBuffer for information collected on the source
            subdomain to be sent to the destination subdomain
        :param coll_idx: GPUBuffer with an array of indices indicating sparse
            nodes from which data is to be collected; this is only used for
            connections via the X_LOW and X_HIGH faces
        :param recv_buf: GPUBuffer for information received from the remote
            subdomain
        :param dist_idx: GPUBuffer with an array of indices indicating
            the position of nodes to which data is to be distributed; this is
            only used for connections via the X_LOW and X_HIGH faces
        :param field: ScalarField that is being transferred using this buffer
        """
        self.face = face
        self.cpair = cpair
        self.coll_buf = coll_buf
        self.coll_idx = coll_idx
        self.recv_buf = recv_buf
        self.dist_idx = dist_idx
        self.field = field

class ConnectionBuffer(object):
    """Contains buffers needed to transfer distributions between two
    subdomains."""

    def __init__(self, face, cpair, coll_buf, coll_idx, recv_buf,
            dist_partial_buf, dist_partial_idx, dist_partial_sel,
            dist_full_buf, dist_full_idx, grid_id=0):
        """
        :param face: face ID
        :param cpair: ConnectionPair, a tuple of two LBConnection objects
        :param coll_buf: GPUBuffer for information collected on the source
            subdomain to be sent to the destination subdomain
        :param coll_idx: GPUBuffer with an array of indices indicating sparse
            nodes from which data is to be collected; this is only used for
            connections via the X_LOW and X_HIGH faces
        :param recv_buf: page-locked numpy buffer with information
            received from the remote subdomain
        :param dist_partial_buf: GPUBuffer, for partial distributions
        :param dist_partial_idx: GPUBuffer, for indices of the partial distributions
        :param dist_partial_sel: selector in recv_buf to get the partial
            distributions
        :param dist_full_buf: GPUBuffer for information to be distributed; this
            is used for nodes where a complete set of distributions is available
            (e.g. not corner or edge nodes)
        :param dist_full_idx: GPUBuffer with an array of indices indicating
            the position of nodes to which data is to be distributed; this is
            only used for connections via the X_LOW and X_HIGH faces
        :param grid_id: grid ID
        """
        self.face = face
        self.cpair = cpair
        self.coll_buf = coll_buf
        self.coll_idx = coll_idx
        self.recv_buf = recv_buf
        self.dist_partial_buf = dist_partial_buf
        self.dist_partial_idx = dist_partial_idx
        self.dist_partial_sel = dist_partial_sel
        self.dist_full_buf = dist_full_buf
        self.dist_full_idx = dist_full_idx
        self.grid_id = grid_id

    def distribute(self, backend, stream):
        # Serialize partial distributions into a contiguous buffer.
        if self.dist_partial_sel is not None:
            self.dist_partial_buf.host[:] = self.recv_buf[self.dist_partial_sel]
            backend.to_buf_async(self.dist_partial_buf.gpu, stream)

        if self.cpair.dst.dst_slice:
            slc = [slice(0, self.recv_buf.shape[0])] + list(
                    reversed(self.cpair.dst.dst_full_buf_slice))
            self.dist_full_buf.host[:] = self.recv_buf[slc]
            backend.to_buf_async(self.dist_full_buf.gpu, stream)


class GPUBuffer(object):
    """Numpy array and a corresponding GPU buffer."""
    def __init__(self, host_buffer, backend):
        self.host = host_buffer
        if host_buffer is not None:
            self.gpu = backend.alloc_buf(like=host_buffer)
        else:
            self.gpu = None


class TimeProfile(object):
    # GPU events.
    BULK = 0
    BOUNDARY = 1
    COLLECTION = 2
    DISTRIB = 3
    MACRO_BULK = 4
    MACRO_BOUNDARY = 5
    MACRO_COLLECTION = 6
    MACRO_DISTRIB = 7

    # CPU events.
    SEND_DISTS = 8
    RECV_DISTS = 9
    SEND_MACRO = 10
    RECV_MACRO = 11
    NET_RECV = 12

    # This event needs to have the highest ID>
    STEP = 13

    def __init__(self, runner):
        self._runner = runner
        self._make_event = runner.backend.make_event
        self._events_start = {}
        self._events_end = {}
        self._times_start = [0.0] * (self.STEP + 1)
        self._timings = [0.0] * (self.STEP + 1)
        self._min_timings = [1000.0] * (self.STEP + 1)
        self._max_timings = [0.0] * (self.STEP + 1)

    def record_start(self):
        self.t_start = time.time()

    def record_end(self):
        self.t_end = time.time()
        if self._runner.config.mode != 'benchmark':
            return
        mi = self._runner.config.max_iters

        ti = util.TimingInfo(
                comp=(self._timings[self.BULK] + self._timings[self.BOUNDARY]) / mi,
                bulk=self._timings[self.BULK] / mi,
                bnd =self._timings[self.BOUNDARY] / mi,
                coll=self._timings[self.COLLECTION] / mi,
                net_wait=self._timings[self.NET_RECV] / mi,
                recv=self._timings[self.RECV_DISTS] / mi,
                send=self._timings[self.SEND_DISTS] / mi,
                total=self._timings[self.STEP] / mi,
                subdomain_id=self._runner._block.id)


        min_ti = util.TimingInfo(
                comp=(self._min_timings[self.BULK] + self._min_timings[self.BOUNDARY]),
                bulk=self._min_timings[self.BULK],
                bnd =self._min_timings[self.BOUNDARY],
                coll=self._min_timings[self.COLLECTION],
                net_wait=self._min_timings[self.NET_RECV],
                recv=self._min_timings[self.RECV_DISTS],
                send=self._min_timings[self.SEND_DISTS],
                total=self._min_timings[self.STEP],
                subdomain_id=self._runner._block.id)

        max_ti = util.TimingInfo(
                comp=(self._max_timings[self.BULK] + self._max_timings[self.BOUNDARY]),
                bulk=self._max_timings[self.BULK],
                bnd =self._max_timings[self.BOUNDARY],
                coll=self._max_timings[self.COLLECTION],
                net_wait=self._max_timings[self.NET_RECV],
                recv=self._max_timings[self.RECV_DISTS],
                send=self._max_timings[self.SEND_DISTS],
                total=self._max_timings[self.STEP],
                subdomain_id=self._runner._block.id)

        self._runner.send_summary_info(ti, min_ti, max_ti)

    def start_step(self):
        self.record_cpu_start(self.STEP)

    def end_step(self):
        self.record_cpu_end(self.STEP)

        for i, ev_start in self._events_start.iteritems():
            duration = self._events_end[i].time_since(ev_start) / 1e3
            self._timings[i] += duration
            self._min_timings[i] = min(self._min_timings[i], duration)
            self._max_timings[i] = max(self._max_timings[i], duration)

    def record_gpu_start(self, event, stream):
        ev = self._make_event(stream, timing=True)
        self._events_start[event] = ev
        return ev

    def record_gpu_end(self, event, stream):
        ev = self._make_event(stream, timing=True)
        self._events_end[event] = ev
        return ev

    def record_cpu_start(self, event):
        self._times_start[event] = time.time()

    def record_cpu_end(self, event):
        t_end = time.time()
        duration = t_end - self._times_start[event]
        self._min_timings[event] = min(self._min_timings[event], duration)
        self._max_timings[event] = max(self._max_timings[event], duration)
        self._timings[event] += duration


def profile(profile_event):
    def _profile(f):
        def decorate(self, *args, **kwargs):
            self._profile.record_cpu_start(profile_event)
            ret = f(self, *args, **kwargs)
            self._profile.record_cpu_end(profile_event)
            return ret
        return decorate
    return _profile

class SubdomainRunner(object):
    """Runs the simulation for a single Subdomain.

    The simulation proceeds into two streams, which is used for overlapping
    calculations with data transfers.  The subdomain is divided into two
    regions -- boundary and bulk.  The nodes in the bulk region are those
    that do not send information to any nodes in other subdomains.  The
    boundary region includes all the remaining nodes, including the ghost
    node envelope used for data storage only.::

        calc stream                    data stream
        -----------------------------------------------
        boundary sim. step    --->     collect data
        bulk sim. step                 ...
                                       distribute data
                           <- sync ->

    An arrow above symbolizes a dependency between the two streams.
    """
    def __init__(self, simulation, block, output, backend, quit_event,
            summary_addr=None, master_addr=None, summary_channel=None):
        """
        :param simulation: instance of a simulation class, descendant from LBSim
        :param block: SubdomainSpec that this runner is to handle
        :param backend: instance of a Sailfish backend class to handle
                GPU interaction
        :param quit_event: multiprocessing Event object; if set, the master is
                requesting early termination of the simulation
        :param master_addr: if not None, zmq address of the machine master
        :param summary_addr: if not None, zmq address string to which summary
                information will be sent.
        """

        self._summary_sender = None
        self._ppid = os.getppid()

        self._ctx = zmq.Context()

        self._block = block
        block.runner = self

        self._output = output
        self.backend = backend

        self._bcg = codegen.BlockCodeGenerator(simulation)
        self._sim = simulation

        if self._bcg.is_double_precision():
            self.float = np.float64
        else:
            self.float = np.float32

        self._scalar_fields = []
        self._vector_fields = []
        self._gpu_field_map = {}
        self._gpu_grids_primary = []
        self._gpu_grids_secondary = []
        self._vis_map_cache = None
        self._quit_event = quit_event

        self._profile = TimeProfile(self)
        # This only happens in unit tests.
        if master_addr is not None:
            self._init_network(master_addr, summary_addr)

    def _init_network(self, master_addr, summary_addr):
        self._master_sock = self._ctx.socket(zmq.PAIR)
        self._master_sock.connect(master_addr)
        if summary_addr is not None:
            if summary_addr != master_addr:
                self._summary_sender = self._ctx.socket(zmq.REQ)
                self._summary_sender.connect(summary_addr)
            else:
                self._summary_sender = self._master_sock

        # For remote connections, we start the listening side of the
        # connection first so that random ports can be selected and
        # communicated to the master.
        unready = []
        ports = {}
        for b_id, connector in self._block._connectors.iteritems():
            if connector.is_ready():
                connector.init_runner(self._ctx)
                if connector.port is not None:
                    ports[(self._block.id, b_id)] = (connector.get_addr(),
                            connector.port)
            else:
                unready.append(b_id)

        self._master_sock.send_pyobj(ports)
        remote_ports = self._master_sock.recv_pyobj()

        for b_id in unready:
            connector = self._block._connectors[b_id]
            addr, port = remote_ports[(b_id, self._block.id)]
            connector.port = port
            connector.set_addr(addr)
            connector.init_runner(self._ctx)

    @property
    def config(self):
        return self._sim.config

    @property
    def dim(self):
        return self._block.dim

    def update_context(self, ctx):
        """Called by the codegen module."""
        self._block.update_context(ctx)
        self._subdomain.update_context(ctx)
        ctx.update(self.backend.get_defines())

        # Size of the lattice.
        ctx['lat_ny'] = self._lat_size[-2]
        ctx['lat_nx'] = self._lat_size[-1]

        # Actual size of the array, including any padding.
        ctx['arr_nx'] = self._physical_size[-1]
        ctx['arr_ny'] = self._physical_size[-2]

        bnd_limits = list(self._block.actual_size[:])

        if self.dim == 3:
            ctx['lat_nz'] = self._lat_size[-3]
            ctx['arr_nz'] = self._physical_size[-3]
            periodic_z = int(self._block.periodic_z)
            ctx['block_periodicity'] = [self._block.periodic_x,
                    self._block.periodic_y, self._block.periodic_z]
        else:
            ctx['lat_nz'] = 1
            ctx['arr_nz'] = 1
            periodic_z = 0
            bnd_limits.append(1)
            ctx['block_periodicity'] = [self._block.periodic_x,
                    self._block.periodic_y, False]

        ctx['boundary_size'] = self._boundary_size
        ctx['lat_linear'] = self.lat_linear
        ctx['lat_linear_dist'] = self.lat_linear_dist
        ctx['lat_linear_macro'] = self.lat_linear_macro

        ctx['bnd_limits'] = bnd_limits
        ctx['dist_size'] = self._get_nodes()
        ctx['sim'] = self._sim
        ctx['block'] = self._block
        ctx['time_dependence'] = self.config.time_dependence
        ctx['check_invalid_values'] = self.config.check_invalid_results_gpu

        arr_nx = self._physical_size[-1]
        arr_ny = self._physical_size[-2]

        ctx['pbc_offsets'] = [{-1:  self.config.lat_nx,
                                1: -self.config.lat_nx},
                              {-1:  self.config.lat_ny * arr_nx,
                                1: -self.config.lat_ny * arr_nx}]

        if self._block.dim == 3:
            ctx['pbc_offsets'].append(
                              {-1:  self.config.lat_nz * arr_ny * arr_nx,
                                1: -self.config.lat_nz * arr_ny * arr_nx})


    def add_visualization_field(self, field_cb, name):
        self._output.register_field(field_cb, name, visualization=True)

    def make_scalar_field(self, dtype=None, name=None, register=True, async=False):
        """Allocates a scalar NumPy array.

        The array includes padding adjusted for the compute device (hidden from
        the end user), as well as space for any ghost nodes.  The returned
        object is a view into the underlying array that hides all ghost nodes.
        Ghost nodes can still be accessed via the 'base' attribute of the
        returned ndarray.

        :param register: if True, the field will be registered for output and
            for automated creation of equivalent field on the compute device.
        """
        if dtype is None:
            dtype = self.float

        size = self._get_nodes()
        strides = self._get_strides(dtype)

        if async:
            buf = self.backend.alloc_async_host_buf(size, dtype=dtype)
        else:
            buf = np.zeros(size, dtype=dtype)

        field = np.ndarray(self._physical_size, buffer=buf,
                           dtype=dtype, strides=strides)

        # Initialize floating point fields to inf to help surfacing problems
        # with uninitialized nodes.
        if dtype == self.float:
            field[:] = np.inf

        assert field.base is buf
        fview = field[self._block._nonghost_slice]
        assert fview.base is field

        # Zero the non-ghost part of the field.
        fview[:] = 0

        if name is not None and register:
            self._output.register_field(fview, name)

        if register:
            self._scalar_fields.append(fview)
        return fview

    def make_vector_field(self, name=None, output=False, async=False):
        """Allocates several scalar arrays representing a vector field."""
        components = []
        view_components = []

        for x in range(0, self._block.dim):
            field = self.make_scalar_field(self.float, register=False, async=async)
            components.append(field)

        if name is not None:
            self._output.register_field(components, name)

        self._vector_fields.append(components)
        return components

    def visualization_map(self):
        if self._vis_map_cache is None:
            self._vis_map_cache = self._subdomain.visualization_map()
        return self._vis_map_cache

    def _init_geometry(self):
        self.config.logger.debug("Initializing geometry.")
        self._init_shape()
        self._subdomain = self._sim.subdomain(self._global_size, self._block,
                self._sim.grid)
        self._subdomain.reset()

    def _init_shape(self):
        # Logical size of the lattice (including ghost nodes).
        # X dimension is the last one on the list (nz, ny, nx)
        self._lat_size = list(reversed(self._block.actual_size))

        # Physical in-memory size of the lattice, adjusted for optimal memory
        # access from the compute unit.  Size of the X dimension is rounded up
        # to a multiple of block_size.  Order is [nz], ny, nx
        self._physical_size = list(reversed(self._block.actual_size))
        bs = self.config.block_size
        self._physical_size[-1] = int(math.ceil(float(self._physical_size[-1]) / bs)) * bs

        self.config.logger.debug('Effective lattice size is: {0}'.format(
            list(reversed(self._physical_size))))

        # CUDA block/grid size for standard kernel call.
        self._kernel_block_size = (bs, 1)
        self._boundary_size = self._block.envelope_size * 2
        bns = self._boundary_size
        assert bns < bs

        if self._block.dim == 2:
            arr_ny, arr_nx = self._physical_size
            lat_ny, lat_nx = self._lat_size
        else:
            arr_nz, arr_ny, arr_nx = self._physical_size
            lat_nz, lat_ny, lat_nx = self._lat_size

        padding = arr_nx - lat_nx
        block = self._block

        x_conns = 0
        # Sometimes, due to misalignment, two blocks might be necessary to
        # cover the right boundary.
        if block.has_face_conn(block.X_HIGH) or block.periodic_x:
            if bs - padding < bns:
                x_conns = 1    # 1 block on the left, 1 block on the right
            else:
                x_conns = 2    # 1 block on the left, 2 blocks on the right

        if block.has_face_conn(block.X_LOW) or block.periodic_x:
            x_conns += 1

        y_conns = 0        # north-south
        if block.has_face_conn(block.Y_LOW) or block.periodic_y:
            y_conns += 1
        if block.has_face_conn(block.Y_HIGH) or block.periodic_y:
            y_conns += 1

        if self._block.dim == 3:
            z_conns = 0        # top-bottom
            if block.has_face_conn(block.Z_LOW) or block.periodic_z:
                z_conns += 1
            if block.has_face_conn(block.Z_HIGH) or block.periodic_z:
                z_conns += 1

        # Number of blocks to be handled by the boundary kernel.  This is also
        # the grid size for boundary kernels.  Note that the number of X-connection
        # blocks does not have the 'bns' factor to account for the thickness of
        # the boundary layer, as the X connection is handled by whole compute
        # device blocks which are assumed to be larger than the boundary layer
        # (i.e. bs > bns).
        if block.dim == 2:
            self._boundary_blocks = (
                    (bns * arr_nx / bs) * y_conns +      # top & bottom
                    (arr_ny - y_conns * bns) * x_conns)  # left & right (w/o top & bottom rows)
            self._kernel_grid_bulk = [arr_nx - x_conns * bs, arr_ny - y_conns * bns]
            self._kernel_grid_full = [arr_nx / bs, arr_ny]
        else:
            self._boundary_blocks = (
                    arr_nx * arr_ny * bns / bs * z_conns +                    # T/B faces
                    arr_nx * (arr_nz - z_conns * bns) / bs * bns * y_conns +  # N/S faces
                    (arr_ny - y_conns * bns) * (arr_nz - z_conns * bns) * x_conns)
            self._kernel_grid_bulk = [
                    (arr_nx - x_conns * bs) * (arr_ny - y_conns * bns),
                    arr_nz - z_conns * bns]
            self._kernel_grid_full = [arr_nx * arr_ny / bs, arr_nz]

        if self._boundary_blocks >= 65536:
            # Use an artificial 2D grid to work around device limits.
            self._boundary_blocks = (4096, int(math.ceil(self._boundary_blocks / 4096.0)))
        else:
            self._boundary_blocks = (self._boundary_blocks, 1)

        self._kernel_grid_bulk[0] /= bs

        # Special cases: boundary kernels can cover the whole domain or this is
        # the only block participating in the simulation.
        if (0 in self._kernel_grid_bulk or self._kernel_grid_bulk[0] < 0 or
                self._kernel_grid_bulk[1] < 0 or len(self._block._connectors) == 0 or
                not self.config.bulk_boundary_split):
            self.config.logger.debug("Disabling bulk/boundary split.")
            # Disable the boundary kernels and ensure that the bulk kernel will
            # cover the whole domain.
            self._boundary_blocks = None
            self._boundary_size = 0
            self._kernel_grid_bulk = self._kernel_grid_full

        self.config.logger.debug('Bulk grid: %s' % repr(self._kernel_grid_bulk))
        self.config.logger.debug('Boundary grid: %s' %
                repr(self._boundary_blocks))

        # Global grid size as seen by the simulation class.
        if self._block.dim == 2:
            self._global_size = (self.config.lat_ny, self.config.lat_nx)
        else:
            self._global_size = (self.config.lat_nz, self.config.lat_ny,
                    self.config.lat_nx)

        # Used so that face values map to the limiting coordinate
        # along a specific axis, e.g. lat_linear[X_LOW] = 0
        evs = self._block.envelope_size
        self.lat_linear = [0, self._lat_size[-1] - 1, 0, self._lat_size[-2] - 1]
        self.lat_linear_dist = [self._lat_size[-1] - 1 - evs, evs,
                self._lat_size[-2] - 1 - evs, evs]
        self.lat_linear_macro = [evs, self._lat_size[-1] - 1 - evs, evs,
                self._lat_size[-2] - 1 - evs]
        # No need to define this, as it's the same as lat_linear above:
        # self.lat_linear_macro_dist = [self._lat_size[-1] - 1, 0,
        #         self._lat_size[-2] - 1, 0]

        if self._block.dim == 3:
            self.lat_linear.extend([0, self._lat_size[-3] - 1])
            self.lat_linear_dist.extend([self._lat_size[-3] - 1 - evs, evs])
            self.lat_linear_macro.extend([evs, self._lat_size[-3] - 1 - evs])

    def _get_strides(self, type_):
        """Returns a list of strides for the NumPy array storing the lattice."""
        t = type_().nbytes
        return list(reversed(reduce(lambda x, y: x + [x[-1] * y],
                self._physical_size[-1:0:-1], [t])))

    def _get_nodes(self):
        """Returns the total amount of actual nodes in the lattice."""
        return reduce(operator.mul, self._physical_size)

    @property
    def num_nodes(self):
        return reduce(operator.mul, self._lat_size)

    def _get_dist_bytes(self, grid):
        """Returns the number of bytes required to store a single set of
           distributions for the whole simulation domain."""
        return self._get_nodes() * grid.Q * self.float().nbytes

    def _get_compute_code(self):
        return self._bcg.get_code(self, self.backend.name)

    def _get_global_idx(self, location, dist_num=0):
        """Returns a global index (in the distributions array).

        :param location: position of the node in the natural order
        :param dist_num: distribution number
        """
        if self.dim == 2:
            gx, gy = location
            arr_nx = self._physical_size[1]
            return gx + arr_nx * gy + (self._get_nodes() * dist_num)
        else:
            gx, gy, gz = location
            arr_nx = self._physical_size[2]
            arr_ny = self._physical_size[1]
            return ((gx + arr_nx * gy + arr_nx * arr_ny * gz) +
                    (self._get_nodes() * dist_num))

    def _idx_helper(self, gx, buf_slice, dists):
        sel = [slice(0, len(dists))]
        idx = np.mgrid[sel + list(reversed(buf_slice))].astype(np.uint32)
        for i, dist_num in enumerate(dists):
            idx[0][i,:] = dist_num
        if self.dim == 2:
            return self._get_global_idx((gx, idx[1]), idx[0]).astype(np.uint32)
        else:
            return self._get_global_idx((gx, idx[2], idx[1]),
                    idx[0]).astype(np.uint32)

    def _get_src_slice_indices(self, face, cpair):
        """Returns a numpy array of indices of sparse nodes from which
        data is to be collected.

        :param face: face ID for the connection
        :param cpair: ConnectionPair describing the connection
        """
        if face in (self._block.X_LOW, self._block.X_HIGH):
            gx = self.lat_linear[face]
        else:
            return None
        return self._idx_helper(gx, cpair.src.src_slice, cpair.src.dists)

    def _get_dst_slice_indices(self, face, cpair):
        """Returns a numpy array of indices of sparse nodes to which
        data is to be distributed.

        :param face: face ID for the connection
        :param cpair: ConnectionPair describing the connection
        """
        if not cpair.dst.dst_slice:
            return None
        if face in (self._block.X_LOW, self._block.X_HIGH):
            gx = self.lat_linear_dist[self._block.opposite_face(face)]
        else:
            return None
        es = self._block.envelope_size
        dst_slice = [
                slice(x.start + es, x.stop + es) for x in
                cpair.dst.dst_slice]
        return self._idx_helper(gx, dst_slice, cpair.dst.dists)

    def _dst_face_loc_to_full_loc(self, face, face_loc):
        axis = self._block.face_to_axis(face)
        missing_loc = self.lat_linear_dist[self._block.opposite_face(face)]
        if axis == 0:
            return [missing_loc] + face_loc
        elif axis == 1:
            if self.dim == 2:
                return (face_loc[0], missing_loc)
            else:
                return (face_loc[0], missing_loc, face_loc[1])
        elif axis == 2:
            return face_loc + [missing_loc]

    def _get_partial_dst_indices(self, face, cpair):
        """Returns objects used to store data for nodes for which a partial
        set of distributions is available.

        This has the form of a triple:
        - a page-locked numpy buffer for the distributions
        - a numpy array of indices of nodes to which partial data is to be
          distributed
        - a selector to be applied to the receive buffer to get the partial
          distributions.

        :param face: face ID for the connection
        :param cpair: ConnectionPair describing the connection
        """
        if cpair.dst.partial_nodes == 0:
            return None, None, None
        buf = self.backend.alloc_async_host_buf(cpair.dst.partial_nodes,
                dtype=self.float)
        idx = np.zeros(cpair.dst.partial_nodes, dtype=np.uint32)
        dst_low = [x + self._block.envelope_size for x in cpair.dst.dst_low]
        sel = []
        i = 0
        for dist_num, locations in sorted(cpair.dst.dst_partial_map.items()):
            for loc in locations:
                dst_loc = [x + y for x, y in zip(dst_low, loc)]
                dst_loc = self._dst_face_loc_to_full_loc(face, dst_loc)

                # Reverse 'loc' here to go from natural order (x, y, z) to the
                # in-face buffer order z, y, x
                sel.append([cpair.dst.dists.index(dist_num)] + list(reversed(loc)))
                idx[i] = self._get_global_idx(dst_loc, dist_num)
                i += 1
        sel2 = []
        for i in range(0, len(sel[0])):
            sel2.append([])

        for loc in sel:
            for i, coord in enumerate(loc):
                sel2[i].append(coord)

        return buf, idx, sel2

    def _init_buffers(self):
        """Creates buffers for inter-block communication."""
        alloc = self.backend.alloc_async_host_buf

        # Maps block ID to a list of ConnectionBuffer objects.  The list will
        # typically contain just 1 element, unless periodic boundary conditions
        # are used.
        self._block_to_connbuf = defaultdict(list)
        for face, block_id in self._block.connecting_subdomains():
            cpairs = self._block.get_connections(face, block_id)
            for cpair in cpairs:
                coll_idx = self._get_src_slice_indices(face, cpair)
                coll_idx = GPUBuffer(coll_idx, self.backend)

                dist_full_idx = self._get_dst_slice_indices(face, cpair)
                dist_full_idx = GPUBuffer(dist_full_idx, self.backend)

                for i, grid in enumerate(self._sim.grids):
                    # TODO(michalj): Optimize this by providing proper padding.
                    coll_buf = alloc(cpair.src.transfer_shape, dtype=self.float)
                    recv_buf = alloc(cpair.dst.transfer_shape, dtype=self.float)
                    dist_full_buf = alloc(cpair.dst.full_shape, dtype=self.float)

                    # Any partial dists are serialized into a single continuous buffer.
                    dist_partial_buf, dist_partial_idx, dist_partial_sel = \
                            self._get_partial_dst_indices(face, cpair)

                    cbuf = ConnectionBuffer(face, cpair,
                            GPUBuffer(coll_buf, self.backend),
                            coll_idx,
                            recv_buf,
                            GPUBuffer(dist_partial_buf, self.backend),
                            GPUBuffer(dist_partial_idx, self.backend),
                            dist_partial_sel,
                            GPUBuffer(dist_full_buf, self.backend),
                            dist_full_idx, i)

                    self.config.logger.debug('adding buffer for conn: {0} -> {1} '
                            '(face {2})'.format(self._block.id, block_id, face))
                    self._block_to_connbuf[block_id].append(cbuf)

        # Explicitly sort connection buffers by their face ID.  Create a
        # separate dictionary where the order of the connection buffers
        # corresponds to that used by the _other_ subdomain.
        self._recv_block_to_connbuf = defaultdict(list)
        for subdomain_id, cbufs in self._block_to_connbuf.iteritems():
            cbufs.sort(key=lambda x: (x.face, x.grid_id))
            recv_bufs = list(cbufs)
            recv_bufs.sort(key=lambda x: (self._block.opposite_face(x.face),
                x.grid_id))
            self._recv_block_to_connbuf[subdomain_id] = recv_bufs


    def _init_compute(self):
        self.config.logger.debug("Initializing compute unit...")
        code = self._get_compute_code()
        self.config.logger.debug("... compute code prepared.")
        self.module = self.backend.build(code)
        self.config.logger.debug("... compute code compiled.")
        self._init_streams()
        self.config.logger.debug("... done.")

    def _init_streams(self):
        self._data_stream = self.backend.make_stream()
        self._calc_stream = self.backend.make_stream()

    def _init_gpu_data(self):
        self.config.logger.debug("Initializing compute unit data.")

        for field in self._scalar_fields:
            self._gpu_field_map[id(field)] = self.backend.alloc_buf(like=field.base)

        for field in self._vector_fields:
            gpu_vector = []
            for component in field:
                gpu_vector.append(self.backend.alloc_buf(like=component.base))
            self._gpu_field_map[id(field)] = gpu_vector

        for grid in self._sim.grids:
            size = self._get_dist_bytes(grid)
            self.config.logger.debug("Using {0} bytes for buffer".format(size))
            self._gpu_grids_primary.append(self.backend.alloc_buf(size=size))
            self._gpu_grids_secondary.append(self.backend.alloc_buf(size=size))

        self._gpu_geo_map = self.backend.alloc_buf(
                like=self._subdomain.encoded_map())

        if self._subdomain.scratch_space_size > 0:
            self.gpu_scratch_space = self.backend.alloc_buf(
                    size=self._subdomain.scratch_space_size * self.float().nbytes)
        else:
            self.gpu_scratch_space = None

    def gpu_field(self, field):
        """Returns the GPU copy of a field."""
        return self._gpu_field_map[id(field)]

    def gpu_dist(self, num, copy):
        """Returns a GPU dist array."""
        if copy == 0:
            return self._gpu_grids_primary[num]
        else:
            return self._gpu_grids_secondary[num]

    def gpu_geo_map(self):
        return self._gpu_geo_map

    def get_kernel(self, name, args, args_format, block_size=None,
            needs_iteration=False):
        if block_size is None:
            block = self._kernel_block_size
        else:
            block = block_size
        return self.backend.get_kernel(self.module, name, args=args,
                args_format=args_format, block=block,
                needs_iteration=needs_iteration)

    def exec_kernel(self, name, args, args_format, needs_iteration=False):
        kernel = self.get_kernel(name, args, args_format,
                needs_iteration=needs_iteration)
        self.backend.run_kernel(kernel, self._kernel_grid_full)

    def step(self, output_req):
        self._step_boundary(output_req)
        self._step_bulk(output_req)
        self._sim.iteration += 1
        self._send_dists()
        # Run this at a point after the compute step is fully scheduled for execution
        # on the GPU and where it doesn't unnecessarily delay othe operations.
        self.backend.set_iteration(self._sim.iteration)
        self._recv_dists()
        self._profile.record_gpu_start(TimeProfile.DISTRIB, self._data_stream)
        for kernel, grid in self._distrib_kernels[self._sim.iteration & 1]:
            self.backend.run_kernel(kernel, grid, self._data_stream)
        self._profile.record_gpu_end(TimeProfile.DISTRIB, self._data_stream)

    def _get_bulk_kernel(self, output_req):
        if output_req:
            kernel = self._kernels_bulk_full[self._sim.iteration & 1]
        else:
            kernel = self._kernels_bulk_none[self._sim.iteration & 1]

        return kernel, self._kernel_grid_bulk

    def _apply_pbc(self, kernels):
        base = 1 - (self._sim.iteration & 1)
        ceil = math.ceil
        ls = self._lat_size
        bs = self.config.block_size

        if self._block.periodic_x:
            if self._block.dim == 2:
                grid_size = (int(ceil(ls[0] / float(bs))), 1)
            else:
                grid_size = (int(ceil(ls[1] / float(bs))), ls[0])
            for kernel in kernels[base][0]:
                self.backend.run_kernel(kernel, grid_size, self._calc_stream)

        if self._block.periodic_y:
            if self._block.dim == 2:
                grid_size = (int(ceil(ls[1] / float(bs))), 1)
            else:
                grid_size = (int(ceil(ls[2] / float(bs))), ls[0])
            for kernel in kernels[base][1]:
                self.backend.run_kernel(kernel, grid_size, self._calc_stream)

        if self._block.dim == 3 and self._block.periodic_z:
            grid_size = (int(ceil(ls[2] / float(bs))), ls[1])
            for kernel in kernels[base][2]:
                self.backend.run_kernel(kernel, grid_size, self._calc_stream)

    def _step_bulk(self, output_req):
        """Runs one simulation step in the bulk domain.

        Bulk domain is defined to be all nodes that belong to CUDA
        blocks that do not depend on input from any ghost nodes.
        """
        # The bulk kernel only needs to be run if the simulation has a bulk/boundary split.
        # If this split is not present, the whole domain is simulated in _step_boundary and
        # _step_bulk only needs to handle PBC (below).
        self._profile.record_gpu_start(TimeProfile.BULK, self._calc_stream)
        if self._boundary_blocks is not None:
            kernel, grid = self._get_bulk_kernel(output_req)
            self.backend.run_kernel(kernel, grid, self._calc_stream)

        self._apply_pbc(self._pbc_kernels)
        self._profile.record_gpu_end(TimeProfile.BULK, self._calc_stream)

    def _step_boundary(self, output_req):
        """Runs one simulation step for the boundary blocks.

        Boundary blocks are CUDA blocks that depend on input from
        ghost nodes."""

        if self._boundary_blocks is not None:
            if output_req:
                kernel = self._kernels_bnd_full[self._sim.iteration & 1]
            else:
                kernel = self._kernels_bnd_none[self._sim.iteration & 1]
            grid = self._boundary_blocks
        else:
            kernel, grid = self._get_bulk_kernel(output_req)

        blk_str = self._calc_stream

        self._profile.record_gpu_start(TimeProfile.BOUNDARY, blk_str)
        self.backend.run_kernel(kernel, grid, blk_str)
        ev = self._profile.record_gpu_end(TimeProfile.BOUNDARY, blk_str)

        # Enqueue a wait so that the data collection will not start until the
        # kernel handling boundary calculations is completed (that kernel runs
        # in the calc stream so that it is automatically synchronized with
        # bulk calculations).
        self._data_stream.wait_for_event(ev)
        self._profile.record_gpu_start(TimeProfile.COLLECTION, self._data_stream)
        for kernel, grid in self._collect_kernels[self._sim.iteration & 1]:
            self.backend.run_kernel(kernel, grid, self._data_stream)
        self._profile.record_gpu_end(TimeProfile.COLLECTION, self._data_stream)

    @profile(TimeProfile.SEND_DISTS)
    def _send_dists(self):
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._block_to_connbuf[b_id]
            for x in conn_bufs:
                self.backend.from_buf_async(x.coll_buf.gpu, self._data_stream)

        self._data_stream.synchronize()
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._block_to_connbuf[b_id]
            if len(conn_bufs) > 1:
                connector.send(np.hstack(
                    [np.ravel(x.coll_buf.host) for x in conn_bufs]))
            else:
                # TODO(michalj): Use non-blocking sends here?
                connector.send(np.ravel(conn_bufs[0].coll_buf.host).copy())

    @profile(TimeProfile.RECV_DISTS)
    def _recv_dists(self):
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._recv_block_to_connbuf[b_id]
            if len(conn_bufs) > 1:
                dest = np.hstack([np.ravel(x.recv_buf) for x in conn_bufs])
                self._profile.record_cpu_start(TimeProfile.NET_RECV)
                # Returns false only if quit event is active.
                if not connector.recv(dest, self._quit_event):
                    return
                self._profile.record_cpu_end(TimeProfile.NET_RECV)
                i = 0
                for cbuf in conn_bufs:
                    l = cbuf.recv_buf.size
                    cbuf.recv_buf[:] = dest[i:i+l].reshape(cbuf.recv_buf.shape)
                    i += l
                    cbuf.distribute(self.backend, self._data_stream)
            else:
                cbuf = conn_bufs[0]
                dest = np.ravel(cbuf.recv_buf)
                # Returns false only if quit event is active.
                self._profile.record_cpu_start(TimeProfile.NET_RECV)
                if not connector.recv(dest, self._quit_event):
                    return
                self._profile.record_cpu_end(TimeProfile.NET_RECV)
                # If ravel returned a copy, we need to write the data
                # back to the proper buffer.
                # TODO(michalj): Check if there is any way of avoiding this
                # copy.
                if dest.flags.owndata:
                    cbuf.recv_buf[:] = dest.reshape(cbuf.recv_buf.shape)
                cbuf.distribute(self.backend, self._data_stream)

    def _fields_to_host(self):
        """Copies data for all fields from the GPU to the host."""
        for field in self._scalar_fields:
            self.backend.from_buf_async(self.gpu_field(field), self._calc_stream)

        for field in self._vector_fields:
            for component in self.gpu_field(field):
                self.backend.from_buf_async(component, self._calc_stream)

    def _init_collect_kernels(self, cbuf, grid_dim1, block_size):
        """Returns collection kernels for a connection.

        :param cbuf: ConnectionBuffer
        :param grid_dim1: callable returning the size of the CUDA grid
            given the total size
        :param block_size: CUDA block size for the kernels

        Returns: secondary, primary.
        """
        # Sparse data collection.
        if cbuf.coll_idx.host is not None:
            grid_size = (grid_dim1(cbuf.coll_buf.host.size),)

            def _get_sparse_coll_kernel(i):
                return KernelGrid(
                    self.get_kernel('CollectSparseData',
                    [cbuf.coll_idx.gpu, self.gpu_dist(cbuf.grid_id, i),
                     cbuf.coll_buf.gpu, cbuf.coll_buf.host.size],
                    'PPPi', (block_size,)),
                    grid_size)

            return _get_sparse_coll_kernel(1), _get_sparse_coll_kernel(0)
        # Continuous data collection.
        else:
            # [X, Z * dists] or [X, Y * dists]
            min_max = ([x.start for x in cbuf.cpair.src.src_slice] +
                    list(reversed(cbuf.coll_buf.host.shape[1:])))
            min_max[-1] = min_max[-1] * len(cbuf.cpair.src.dists)
            if self.dim == 2:
                signature = 'PiiiP'
                grid_size = (grid_dim1(cbuf.coll_buf.host.size),)
            else:
                signature = 'PiiiiiP'
                grid_size = (grid_dim1(cbuf.coll_buf.host.shape[-1]),
                    cbuf.coll_buf.host.shape[-2] * len(cbuf.cpair.src.dists))

            def _get_cont_coll_kernel(i):
                return KernelGrid(
                    self.get_kernel('CollectContinuousData',
                    [self.gpu_dist(cbuf.grid_id, i),
                     cbuf.face] + min_max + [cbuf.coll_buf.gpu],
                     signature, (block_size,)),
                     grid_size)

            return _get_cont_coll_kernel(1), _get_cont_coll_kernel(0)

    def _init_distrib_kernels(self, cbuf, grid_dim1, block_size):
        """Returns distribution kernels for a connection.

        :param cbuf: ConnectionBuffer
        :param grid_dim1: callable returning the size of the CUDA grid
            given the total size
        :param block_size: CUDA block size for the kernels

        Returns: lists of: primary, secondary.
        """
        primary = []
        secondary = []

        # Data distribution
        # Partial nodes.
        if cbuf.dist_partial_idx.host is not None:
            grid_size = (grid_dim1(cbuf.dist_partial_buf.host.size),)

            def _get_sparse_dist_kernel(i):
                return KernelGrid(
                        self.get_kernel('DistributeSparseData',
                            [cbuf.dist_partial_idx.gpu,
                             self.gpu_dist(cbuf.grid_id, i),
                             cbuf.dist_partial_buf.gpu,
                             cbuf.dist_partial_buf.host.size],
                            'PPPi', (block_size,)),
                        grid_size)

            primary.append(_get_sparse_dist_kernel(0))
            secondary.append(_get_sparse_dist_kernel(1))

        # Full nodes (all transfer distributions).
        if cbuf.dist_full_buf.host is not None:
            # Sparse indexing (connection along X-axis).
            if cbuf.dist_full_idx.host is not None:
                grid_size = (grid_dim1(cbuf.dist_full_buf.host.size),)

                def _get_sparse_fdist_kernel(i):
                    return KernelGrid(
                            self.get_kernel('DistributeSparseData',
                                [cbuf.dist_full_idx.gpu,
                                 self.gpu_dist(cbuf.grid_id, i),
                                 cbuf.dist_full_buf.gpu,
                                 cbuf.dist_full_buf.host.size],
                            'PPPi', (block_size,)),
                            grid_size)

                primary.append(_get_sparse_fdist_kernel(0))
                secondary.append(_get_sparse_fdist_kernel(1))
            # Continuous indexing.
            elif cbuf.cpair.dst.dst_slice:
                # [X, Z * dists] or [X, Y * dists]
                low = [x + self._block.envelope_size for x in cbuf.cpair.dst.dst_low]
                min_max = ([(x + y.start) for x, y in zip(low, cbuf.cpair.dst.dst_slice)] +
                        list(reversed(cbuf.dist_full_buf.host.shape[1:])))
                min_max[-1] = min_max[-1] * len(cbuf.cpair.dst.dists)

                if self.dim == 2:
                    signature = 'PiiiP'
                    grid_size = (grid_dim1(cbuf.dist_full_buf.host.size),)
                else:
                    signature = 'PiiiiiP'
                    grid_size = (grid_dim1(cbuf.dist_full_buf.host.shape[-1]),
                        cbuf.dist_full_buf.host.shape[-2] * len(cbuf.cpair.dst.dists))

                def _get_cont_dist_kernel(i):
                    return KernelGrid(
                            self.get_kernel('DistributeContinuousData',
                            [self.gpu_dist(cbuf.grid_id, i),
                             self._block.opposite_face(cbuf.face)] +
                            min_max + [cbuf.dist_full_buf.gpu],
                            signature, (block_size,)),
                            grid_size)

                primary.append(_get_cont_dist_kernel(0))
                secondary.append(_get_cont_dist_kernel(1))

        return primary, secondary

    def _init_interblock_kernels(self):
        """Creates kernels for collection and distribution of distribution
        data."""

        collect_primary = []
        collect_secondary = []

        distrib_primary = []
        distrib_secondary = []
        self._distrib_grid = []

        collect_block = 32
        def _grid_dim1(x):
            return int(math.ceil(x / float(collect_block)))

        for b_id, conn_bufs in self._block_to_connbuf.iteritems():
            for cbuf in conn_bufs:
                primary, secondary = self._init_collect_kernels(cbuf,
                        _grid_dim1, collect_block)
                collect_primary.append(primary)
                collect_secondary.append(secondary)

                primary, secondary = self._init_distrib_kernels(cbuf,
                        _grid_dim1, collect_block)
                distrib_primary.extend(primary)
                distrib_secondary.extend(secondary)

        self._collect_kernels = (collect_primary, collect_secondary)
        self._distrib_kernels = (distrib_primary, distrib_secondary)

    def _debug_get_dist(self, output=True):
        """Copies the distributions from the GPU to a properly structured host array.
        :param output: if True, returns the contents of the distributions set *after*
                the current simulation step
        """
        iter_idx = self._sim.iteration & 1
        if not output:
            iter_idx = 1 - iter_idx

        self.config.logger.debug('getting dist {0} ({1})'.format(iter_idx,
            self.gpu_dist(0, iter_idx)))
        dbuf = np.zeros(self._get_dist_bytes(self._sim.grid) / self.float().nbytes,
            dtype=self.float)
        self.backend.from_buf(self.gpu_dist(0, iter_idx), dbuf)
        dbuf = dbuf.reshape([self._sim.grid.Q] + self._physical_size)
        return dbuf

    def _debug_set_dist(self, dbuf, output=True):
        iter_idx = self._sim.iteration & 1
        if not output:
            iter_idx = 1 - iter_idx

        self.backend.to_buf(self.gpu_dist(0, iter_idx), dbuf)

    def _debug_global_idx_to_tuple(self, gi):
        dist_num = gi / self._get_nodes()
        rest = gi % self._get_nodes()
        arr_nx = self._physical_size[-1]
        gx = rest % arr_nx
        gy = rest / arr_nx
        return dist_num, gy, gx

    def send_summary_info(self, timing_info, min_timings, max_timings):
        if self._summary_sender is not None:
            self._summary_sender.send_pyobj((timing_info, min_timings,
                    max_timings))
            self.config.logger.debug('Sending timing information to controller.')
            assert self._summary_sender.recv() == 'ack'

    def _initial_conditions(self):
        self._sim.initial_conditions(self)
        self._sim.verify_fields()

    def run(self):
        self.config.logger.info("Initializing block.")
        self.config.logger.debug(self.backend.info)

        self._init_geometry()
        self._sim.init_fields(self)
        self._init_buffers()
        self._init_compute()
        self.config.logger.debug("Initializing macroscopic fields.")
        self._subdomain.init_fields(self._sim)
        self._init_gpu_data()
        self.config.logger.debug("Applying initial conditions.")

        self._init_interblock_kernels()
        self._kernels_bulk_full = self._sim.get_compute_kernels(self, True, True)
        self._kernels_bulk_none = self._sim.get_compute_kernels(self, False, True)
        self._kernels_bnd_full = self._sim.get_compute_kernels(self, True, False)
        self._kernels_bnd_none = self._sim.get_compute_kernels(self, False, False)
        self._pbc_kernels = self._sim.get_pbc_kernels(self)

        self._initial_conditions()

        if self.config.output:
            self._output.save(self._sim.iteration)

        self.config.logger.info("Starting simulation.")

        if not self.config.max_iters:
            self.config.logger.warning("Running infinite simulation.")

        self.main()

        self.config.logger.info(
            "Simulation completed after {0} iterations.".format(
                self._sim.iteration))

    def need_quit(self):
        # The quit event is used by the visualization interface.
        if self._quit_event.is_set():
            self.config.logger.info("Simulation termination requested.")
            return True

        if self._ppid != os.getppid():
            self.config.logger.info("Master process is dead -- terminating simulation.")
            return True

        return False

    def main(self):
        self._profile.record_start()
        while True:
            self._profile.start_step()
            output_req = self._sim.need_output()

            if output_req and self.config.debug_dump_dists:
                dbuf = self._debug_get_dist(self)
                self._output.dump_dists(dbuf, self._sim.iteration)

            self.step(output_req)

            if output_req and self.config.output_required:
                self._fields_to_host()

            if (self.config.max_iters > 0 and self._sim.iteration >=
                    self.config.max_iters) or self.need_quit():
                break

            self._data_stream.synchronize()
            self._calc_stream.synchronize()
            if output_req and self.config.output_required:
                if self.config.check_invalid_results_host:
                    if not self._output.verify():
                        self.config.logger.error("Invalid value detected in "
                                "output for iteration {0}".format(
                                self._sim.iteration))
                        self._quit_event.set()
                        break
                self._output.save(self._sim.iteration)
            self._profile.end_step()

            self._sim.after_step()

        # Receive any data from remote nodes prior to termination.  This ensures
        # we don't run into problems with zmq.
        self._data_stream.synchronize()
        self._calc_stream.synchronize()
        if output_req and self.config.output_required:
            self._output.save(self._sim.iteration)

        self._profile.record_end()


class NNSubdomainRunner(SubdomainRunner):
    """Runs a fluid simulation for a single subdomain.

    This is a specialization of the SubdomainRunner class for models which
    require access to macroscopic fields from the nearest neighbor nodes.
    It changes the steps executed on the GPU as follows::

        calc stream                    data stream
        -------------------------------------------------------
        boundary macro fields    --->  collect macro data
        bulk macro fields              ...
        boundary sim. step       <---  distribute macro data
                                 --->  collect distrib. data
        bulk sim. step                 ...
                                       distribute distrib. data
                           <- sync ->

        TODO(michalj): Try the alternative scheme:

        calc stream                    data stream
        -------------------------------------------------------
        boundary macro fields    --->  collect macro data
        bulk macro fields              ...
        bulk sim. step
        boundary sim. step       <---  distribute macro data
                                 --->  collect distrib. data
                                       ...
                                       distribute distrib. data
                           <- sync ->


    An arrow above symbolizes a dependency between the two streams.
    """

    @profile(TimeProfile.RECV_MACRO)
    def _recv_macro(self):
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._recv_block_to_macrobuf[b_id]
            if len(conn_bufs) > 1:
                dest = np.hstack([np.ravel(x.recv_buf.host) for x in conn_bufs])
                # Returns false only if quit event is active.
                if not connector.recv(dest, self._quit_event):
                    return
                i = 0
                for cbuf in conn_bufs:
                    l = cbuf.recv_buf.host.size
                    cbuf.recv_buf.host[:] = dest[i:i+l].reshape(cbuf.recv_buf.host.shape)
                    i += l
                    self.backend.to_buf_async(cbuf.recv_buf.gpu, self._data_stream)
            else:
                cbuf = conn_bufs[0]
                dest = np.ravel(cbuf.recv_buf.host)
                # Returns false only if quit event is active.
                if not connector.recv(dest, self._quit_event):
                    return
                if dest.flags.owndata:
                    cbuf.recv_buf.host[:] = dest.reshape(cbuf.recv_buf.host.shape)
                self.backend.to_buf_async(cbuf.recv_buf.gpu, self._data_stream)

    @profile(TimeProfile.SEND_MACRO)
    def _send_macro(self):
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._block_to_macrobuf[b_id]
            for x in conn_bufs:
                self.backend.from_buf_async(x.coll_buf.gpu, self._data_stream)

        self._data_stream.synchronize()
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._block_to_macrobuf[b_id]
            if len(conn_bufs) > 1:
                connector.send(np.hstack(
                    [np.ravel(x.coll_buf.host) for x in conn_bufs]))
            else:
                # TODO(michalj): Use non-blocking sends here?
                connector.send(np.ravel(conn_bufs[0].coll_buf.host).copy())

    def _macro_idx_helper(self, gx, buf_slice):
        idx = np.mgrid[list(reversed(buf_slice))].astype(np.uint32)
        if self.dim == 2:
            return self._get_global_idx((gx, idx[0])).astype(np.uint32)
        else:
            return self._get_global_idx((gx, idx[1], idx[0])).astype(np.uint32)

    def _get_src_macro_indices(self, face, cpair):
        if face in (self._block.X_LOW, self._block.X_HIGH):
            gx = self.lat_linear_macro[face]
        else:
            return None
        return self._macro_idx_helper(gx, cpair.src.src_macro_slice)

    def _get_dst_macro_indices(self, face, cpair):
        if face in (self._block.X_LOW, self._block.X_HIGH):
            gx = self.lat_linear[face]
        else:
            return None
        return self._macro_idx_helper(gx, cpair.src.dst_macro_slice)

    def _init_buffers(self):
        super(NNSubdomainRunner, self)._init_buffers()

        alloc = self.backend.alloc_async_host_buf
        self._block_to_macrobuf = defaultdict(list)
        self._num_nn_fields = sum((1 for fpair in self._sim._scalar_fields if
            fpair.abstract.need_nn))
        for face, block_id in self._block.connecting_subdomains():
            cpairs = self._block.get_connections(face, block_id)
            for cpair in cpairs:
                coll_idx = GPUBuffer(self._get_src_macro_indices(face, cpair), self.backend)
                dist_idx = GPUBuffer(self._get_dst_macro_indices(face, cpair), self.backend)

                for field_pair in self._sim._scalar_fields:
                    if not field_pair.abstract.need_nn:
                        continue

                    recv_buf = alloc(cpair.dst.macro_transfer_shape, dtype=self.float)
                    coll_buf = alloc(cpair.src.macro_transfer_shape, dtype=self.float)

                    cbuf = MacroConnectionBuffer(face, cpair,
                            GPUBuffer(coll_buf, self.backend),
                            coll_idx,
                            GPUBuffer(recv_buf, self.backend),
                            dist_idx, field_pair.buffer)

                    self._block_to_macrobuf[block_id].append(cbuf)

        # Explicitly sort connection buffers by their face ID.  Create a
        # separate dictionary where the order of the connection buffers
        # corresponds to that used by the _other_ subdomain.
        self._recv_block_to_macrobuf = defaultdict(list)
        for subdomain_id, cbufs in self._block_to_macrobuf.iteritems():
            cbufs.sort(key=lambda x:(x.face))
            recv_bufs = list(cbufs)
            recv_bufs.sort(key=lambda x: self._block.opposite_face(x.face))
            self._recv_block_to_macrobuf[subdomain_id] = recv_bufs

    def _init_macro_collect_kernels(self, cbuf, grid_dim1, block_size):
        # Sparse data collection.
        if cbuf.coll_idx.host is not None:
            grid_size = (grid_dim1(cbuf.coll_buf.host.size),)
            return KernelGrid(self.get_kernel('CollectSparseData',
                    [cbuf.coll_idx.gpu, self.gpu_field(cbuf.field),
                     cbuf.coll_buf.gpu, cbuf.coll_buf.host.size], 'PPPi',
                    (block_size,)), grid_size)
        # Continuous data collection.
        else:
            if self.dim == 2:
                grid_size = (grid_dim1(cbuf.coll_buf.host.size),)
            else:
                grid_size = (grid_dim1(cbuf.coll_buf.host.shape[-1]),
                    grid_dim1(cbuf.coll_buf.host.shape[-2]))

            # [X, Z] or [X, Y] in 3D
            min_max = ([x.start for x in cbuf.cpair.src.src_macro_slice] +
                    list(reversed(cbuf.coll_buf.host.shape)))

            if self.dim == 2:
                gy = self.lat_linear_macro[cbuf.face]
                return KernelGrid(self.get_kernel('CollectContinuousMacroData',
                        [self.gpu_field(cbuf.field)] + min_max + [gy,
                         cbuf.coll_buf.gpu], 'PiiiP', (block_size,)),
                         grid_size)
            else:
                return KernelGrid(self.get_kernel('CollectContinuousMacroData',
                        [self.gpu_field(cbuf.field), cbuf.face] +
                        min_max + [cbuf.coll_buf.gpu], 'PiiiiiP', (block_size,)),
                        grid_size)


    def _init_macro_distrib_kernels(self, cbuf, grid_dim1, block_size):
        kernels = []

        # Sparse data distribution.
        if cbuf.dist_idx.host is not None:
            grid_size = (grid_dim1(cbuf.recv_buf.host.size),)
            return KernelGrid(self.get_kernel('DistributeSparseData',
                        [cbuf.dist_idx.gpu, self.gpu_field(cbuf.field),
                         cbuf.recv_buf.gpu, cbuf.recv_buf.host.size], 'PPPi',
                        (block_size,)), grid_size)
        # Continuous data distribution.
        else:
            if self.dim == 2:
                grid_size = (grid_dim1(cbuf.recv_buf.host.size),)
            else:
                grid_size = (grid_dim1(cbuf.recv_buf.host.shape[-1]),
                    grid_dim1(cbuf.recv_buf.host.shape[-2]))

            # [X, Z] or [X, Y] in 3D
            min_max = ([x.start for x in cbuf.cpair.src.dst_macro_slice] +
                    list(reversed(cbuf.recv_buf.host.shape)))

            if self.dim == 2:
                gy = self.lat_linear[cbuf.face]
                return KernelGrid(self.get_kernel('DistributeContinuousMacroData',
                        [self.gpu_field(cbuf.field)] + min_max + [gy,
                         cbuf.recv_buf.gpu], 'PiiiP', (block_size,)),
                         grid_size)
            else:
                return KernelGrid(self.get_kernel('DistributeContinuousMacroData',
                        [self.gpu_field(cbuf.field), cbuf.face] +
                        min_max + [cbuf.recv_buf.gpu], 'PiiiiiP', (block_size,)),
                        grid_size)


    def _init_interblock_kernels(self):
        super(NNSubdomainRunner, self)._init_interblock_kernels()

        collect_block = 32
        def _grid_dim1(x):
            return int(math.ceil(x / float(collect_block)))

        self._macro_collect_kernels = []
        self._macro_distrib_kernels = []

        for b_id, conn_bufs in self._block_to_macrobuf.iteritems():
            for cbuf in conn_bufs:
                self._macro_collect_kernels.append(
                        self._init_macro_collect_kernels(cbuf, _grid_dim1,
                            collect_block))
                self._macro_distrib_kernels.append(
                        self._init_macro_distrib_kernels(cbuf, _grid_dim1,
                            collect_block))

    def _initial_conditions(self):
        """Prepares non-local fields prior to simulation start-up.

        This is necessary for models that use non-local field values to set
        the initial values of the distributions."""

        for kernel, grid in self._macro_collect_kernels:
            self.backend.run_kernel(kernel, grid, self._data_stream)

        self._data_stream.synchronize()
        self._apply_pbc(self._pbc_kernels.macro)
        self._send_macro()
        if self.need_quit():
            return False

        self._recv_macro()

        for kernel, grid in self._macro_distrib_kernels:
            self.backend.run_kernel(kernel, grid, self._data_stream)

        self._data_stream.synchronize()
        super(NNSubdomainRunner, self)._initial_conditions()

    def step(self, output_req):
        """Runs one simulation step."""

        if output_req:
            bnd = self._kernels_bnd_full
            bulk = self._kernels_bulk_full
        else:
            bnd = self._kernels_bnd_none
            bulk = self._kernels_bulk_none

        it = self._sim.iteration
        bnd_kernel_macro, bnd_kernel_sim = bnd[it & 1]
        bulk_kernel_macro, bulk_kernel_sim = bulk[it & 1]

        # Local aliases.
        has_boundary_split = self._boundary_blocks is not None
        str_calc = self._calc_stream
        str_data = self._data_stream
        run = self.backend.run_kernel
        grid_bulk = self._kernel_grid_bulk
        grid_bnd = self._boundary_blocks
        record_gpu_start = self._profile.record_gpu_start
        record_gpu_end   = self._profile.record_gpu_end

        # Macroscopic variables.
        record_gpu_start(TimeProfile.MACRO_BOUNDARY, str_calc)
        if has_boundary_split:
            run(bnd_kernel_macro, grid_bnd, str_calc)
        else:
            run(bulk_kernel_macro, grid_bulk, str_calc)
        ev = record_gpu_end(TimeProfile.MACRO_BOUNDARY, str_calc)
        str_data.wait_for_event(ev)

        record_gpu_start(TimeProfile.MACRO_COLLECTION, str_data)
        for kernel, grid in self._macro_collect_kernels:
            run(kernel, grid, str_data)
        record_gpu_end(TimeProfile.MACRO_COLLECTION, str_data)

        record_gpu_start(TimeProfile.MACRO_BULK, str_calc)
        if has_boundary_split:
            run(bulk_kernel_macro, grid_bulk, str_calc)
        self._apply_pbc(self._pbc_kernels.macro)
        record_gpu_end(TimeProfile.MACRO_BULK, str_calc)

        self._send_macro()
        if self.need_quit():
            return False

        self._recv_macro()

        record_gpu_start(TimeProfile.MACRO_DISTRIB, str_data)
        for kernel, grid in self._macro_distrib_kernels:
            run(kernel, grid, str_data)

        ev = record_gpu_end(TimeProfile.MACRO_DISTRIB, str_data)
        str_calc.wait_for_event(ev)

        # Actual simulation step.
        record_gpu_start(TimeProfile.BOUNDARY, str_calc)
        if has_boundary_split:
            run(bnd_kernel_sim, grid_bnd, str_calc)
        else:
            run(bulk_kernel_sim, grid_bulk, str_calc)
        ev = record_gpu_end(TimeProfile.BOUNDARY, str_calc)
        str_data.wait_for_event(ev)

        record_gpu_start(TimeProfile.COLLECTION, str_data)
        for kernel, grid in self._collect_kernels[it & 1]:
            run(kernel, grid, str_data)
        record_gpu_end(TimeProfile.COLLECTION, str_data)

        record_gpu_start(TimeProfile.BULK, str_calc)
        if has_boundary_split:
            run(bulk_kernel_sim, grid_bulk, str_calc)
        self._apply_pbc(self._pbc_kernels.distributions)
        record_gpu_end(TimeProfile.BULK, str_calc)

        self._sim.iteration += 1

        self._send_dists()
        if self.need_quit():
            return False

        self.backend.set_iteration(self._sim.iteration)

        self._recv_dists()
        record_gpu_start(TimeProfile.DISTRIB, str_data)
        for kernel, grid in self._distrib_kernels[(it + 1) & 1]:
            run(kernel, grid, str_data)
        record_gpu_end(TimeProfile.DISTRIB, str_data)

        return True


"""Code for controlling a single subdomain of a LB simulation."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict, namedtuple
import cPickle as pickle
import math
import operator
import os
import signal
import time
import numpy as np
import zmq
from sailfish import codegen, io
from sailfish.lb_base import LBMixIn
from sailfish.profile import profile, TimeProfile
from sailfish.subdomain_connection import ConnectionBuffer, MacroConnectionBuffer
import sailfish.node_type as nt

# Used to hold a reference to a CUDA kernel and a grid on which it is
# to be executed.
KernelGrid = namedtuple('KernelGrid', 'kernel grid')


class GPUBuffer(object):
    """Numpy array and a corresponding GPU buffer."""
    def __init__(self, host_buffer, backend):
        self.host = host_buffer
        if host_buffer is not None:
            self.gpu = backend.alloc_buf(like=host_buffer)
        else:
            self.gpu = None


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
    def __init__(self, simulation, spec, output, backend, quit_event,
            summary_addr=None, master_addr=None, summary_channel=None):
        """
        :param simulation: instance of a simulation class, descendant from LBSim
        :param spec: SubdomainSpec that this runner is to handle
        :param backend: instance of a Sailfish backend class to handle
                GPU interaction
        :param quit_event: multiprocessing Event object; if set, the master is
                requesting early termination of the simulation
        :param master_addr: if not None, zmq address of the machine master
        :param summary_addr: if not None, zmq address string to which summary
                information will be sent.
        """

        self._summary_sender = None
        self._ppid = os.getppid() if os.name != 'nt' else 0

        self._ctx = zmq.Context()

        self._spec = spec
        spec.runner = self

        self._output = output
        self.backend = backend

        self._bcg = codegen.BlockCodeGenerator(simulation)
        self._sim = simulation

        # Subdomain object handled by this runner.
        self._subdomain = None

        if self._bcg.is_double_precision():
            self.float = np.float64
        else:
            self.float = np.float32

        # Maps ID of field base buffer to the strided view.
        self._field_base = {}
        self._scalar_fields = []
        self._vector_fields = []

        # Sparse fields used for indirect node addressing.
        self._sparse_scalar_fields = []
        self._sparse_vector_fields = []

        # Set of fields that are also wrapped in a GPUArray.
        self._array_fields = set()
        self._gpu_field_map = {}
        self._gpu_grids_primary = []
        self._gpu_grids_secondary = []  # only used for the AB access pattern
        self._gpu_indirect_address = None  # only used for indirect node addressing
        self._host_indirect_address = None
        self._quit_event = quit_event

        self._pbc_kernels = []

        # Dictionary of variables to be exported to the code templates
        # in update_context().
        self._code_context = {}

        self._profile = TimeProfile(self)
        # This only happens in unit tests.
        if master_addr is not None:
            self._init_network(master_addr, summary_addr)

        np.random.seed(self.config.seed)
        self._initialization = self.config.init_iters > 0

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
        for b_id, connector in self._spec._connectors.iteritems():
            if connector.is_ready():
                connector.init_runner(self._ctx)
                if connector.port is not None:
                    ports[(self._spec.id, b_id)] = (connector.get_addr(),
                            connector.port)
            else:
                unready.append(b_id)

        self._master_sock.send_pyobj(ports)
        remote_ports = self._master_sock.recv_pyobj()

        for b_id in unready:
            connector = self._spec._connectors[b_id]
            addr, port = remote_ports[(b_id, self._spec.id)]
            connector.port = port
            connector.set_addr(addr)
            connector.init_runner(self._ctx)

    @property
    def config(self):
        return self._sim.config

    @property
    def dim(self):
        return self._spec.dim

    def update_context(self, ctx):
        """Called by the codegen module."""
        self._spec.update_context(ctx)
        self._subdomain.update_context(ctx)
        ctx.update(self.backend.get_defines())
        ctx.update(self._code_context)

        # Size of the lattice, including ghost nodes (without padding).
        ctx['lat_ny'] = self._lat_size[-2]
        ctx['lat_nx'] = self._lat_size[-1]

        # Actual size of the array, including any padding.
        arr_nx = self._physical_size[-1]
        arr_ny = self._physical_size[-2]
        ctx['arr_nx'] = arr_nx
        ctx['arr_ny'] = arr_ny

        bnd_limits = list(self._spec.actual_size[:])

        if self.dim == 3:
            ctx['lat_nz'] = self._lat_size[-3]
            ctx['arr_nz'] = self._physical_size[-3]
            ctx['block_periodicity'] = [self._spec.periodic_x,
                    self._spec.periodic_y, self._spec.periodic_z]
        else:
            ctx['lat_nz'] = 1
            ctx['arr_nz'] = 1
            bnd_limits.append(1)
            ctx['block_periodicity'] = [self._spec.periodic_x, self._spec.periodic_y, False]

        ctx['lat_linear'] = self.lat_linear
        ctx['lat_linear_with_swap'] = self.lat_linear_with_swap
        ctx['lat_linear_dist'] = self.lat_linear_dist
        ctx['lat_linear_macro'] = self.lat_linear_macro

        ctx['bnd_limits'] = bnd_limits

        if self.config.node_addressing == 'indirect':
            ctx['dist_size'] = self._subdomain.active_nodes
        else:
            ctx['dist_size'] = self.num_phys_nodes
        ctx['sim'] = self._sim
        ctx['block'] = self._spec
        ctx['time_dependence'] = self.config.time_dependence
        ctx['space_dependence'] = self.config.space_dependence
        ctx['gpu_check_invalid_values'] = (
                self.config.check_invalid_results_gpu and
                self.backend.supports_printf)

        if (self.config.check_invalid_results_gpu and
                not self.backend.supports_printf):
            self.config.logger.info('On-GPU invalid result check disabled'
                    ' as the device does not support all required features.')

        ctx['pbc_offsets'] = [{-1:  self.config.lat_nx,
                                1: -self.config.lat_nx},
                              {-1:  self.config.lat_ny * arr_nx,
                                1: -self.config.lat_ny * arr_nx}]

        if self._spec.dim == 3:
            ctx['pbc_offsets'].append(
                              {-1:  self.config.lat_nz * arr_ny * arr_nx,
                                1: -self.config.lat_nz * arr_ny * arr_nx})

        ctx['initialization'] = self._initialization

    def add_visualization_field(self, field_cb, name):
        self._output.register_field(field_cb, name, visualization=True)

    def make_scalar_field(self, dtype=None, name=None, register=True,
                          async=False, gpu_array=False, need_indirect=True,
                          nonghost_view=True):
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

        if async:
            buf = self.backend.alloc_async_host_buf(self.num_phys_nodes, dtype=dtype)
        else:
            buf = np.zeros(self.num_phys_nodes, dtype=dtype)

        strides = self._get_strides(dtype)
        field = np.ndarray(self._physical_size, buffer=buf,
                           dtype=dtype, strides=strides)

        # Initialize floating point fields to inf to help surfacing problems
        # with uninitialized nodes.
        if dtype == self.float:
            field[:] = np.inf

        assert field.base is buf
        if nonghost_view:
            fview = field[self._spec._nonghost_slice]
        else:
            fview = field
        self._field_base[id(fview.base)] = field

        if gpu_array:
            self._array_fields.add(id(fview.base))

        # Prior to numpy 1.7.0, the base was field (first element up the chain).
        assert fview.base is buf or fview.base is field

        # Zero the non-ghost part of the field.
        fview[:] = 0

        if name is not None and register:
            self._output.register_field(fview, name)

        if register:
            self._scalar_fields.append(fview)

        # Create sparse fields if necessary. With indirect node addressing,
        # both distributions and macroscopic fields are stored in 'sparse'
        # arrays that only store data for active nodes. The dense fields
        # allocated above are only used for compatibility with the host
        # initialization code.
        sparse_field = None
        if self.config.node_addressing == 'indirect' and need_indirect:
            if async:
                sparse_field = self.backend.alloc_async_host_buf(
                    self._subdomain.active_nodes, dtype=dtype)
            else:
                sparse_field = np.zeros(self._subdomain.active_nodes, dtype=dtype)
            if register:
                self._sparse_scalar_fields.append(sparse_field)

        return fview, sparse_field

    def field_base(self, field):
        return self._field_base[id(field.base)]

    def make_vector_field(self, name=None, output=False, async=False,
                          gpu_array=False):
        """Allocates several scalar arrays representing a vector field."""
        components = []
        sparse_components = []

        for x in range(0, self._spec.dim):
            field, sparse_field = self.make_scalar_field(
                self.float, register=False, async=async, gpu_array=gpu_array)
            components.append(field)
            sparse_components.append(sparse_field)

        if name is not None:
            self._output.register_field(components, name)

        self._vector_fields.append(components)
        self._sparse_vector_fields.append(sparse_components)
        return components

    def visualization_map(self):
        """Returns an unencoded geometry map used for visualization."""
        return self._subdomain.visualization_map()

    def _init_geometry(self):
        self.config.logger.debug("Initializing geometry.")
        self._init_shape()
        self._subdomain = self._sim.subdomain(self._global_size, self._spec, self._sim.grid)
        self._subdomain.allocate()
        self._subdomain.reset()
        self._output.set_fluid_map(self._subdomain.fluid_map())
        if self.config.debug_dump_node_type_map:
            self._output.dump_node_type(self._subdomain.visualization_map())

    def _init_shape(self):
        # Logical size of the lattice (including ghost nodes).
        # X dimension is the last one on the list (nz, ny, nx)
        self._lat_size = list(reversed(self._spec.actual_size))

        # Physical in-memory size of the lattice, adjusted for optimal memory
        # access from the compute unit. Size of the X dimension is rounded up
        # to a multiple of block_size. Order is [nz], ny, nx
        self._physical_size = list(reversed(self._spec.actual_size))

        # Optimizing memory alignment only makes sense when using direct node
        # addressing.
        if self.config.node_addressing == 'direct':
            alignment = self.config.mem_alignment
            self._physical_size[-1] = int(math.ceil(float(self._physical_size[-1]) / alignment)) * alignment
        else:
            alignment = 1

        # Size of the lattice as necessary for the kernel grid to fully cover
        # all available space. This has to be larger or equal than the real
        # in-memory lattice.
        bs = self.config.block_size
        assert bs >= alignment, ('The block size (--block_size) has to be at '
                'least as large as --memory_alignment')
        grid_nx = int(math.ceil(float(self._spec.actual_size[0]) / bs)) * bs
        self._code_context['grid_nx'] = grid_nx

        self.config.logger.debug('Effective lattice size is: {0}'.format(
            list(reversed(self._physical_size))))

        # CUDA block/grid size for standard kernel call.
        self._kernel_block_size = (bs, 1)
        bns = self._spec.envelope_size * 2
        self._code_context['boundary_size'] = bns
        assert bns < bs

        if self._spec.dim == 2:
            arr_ny, arr_nx = self._physical_size
            lat_ny, lat_nx = self._lat_size
        else:
            arr_nz, arr_ny, arr_nx = self._physical_size
            lat_nz, lat_ny, lat_nx = self._lat_size

        padding = grid_nx - lat_nx
        spec = self._spec

        x_conns = 0
        # Sometimes, due to misalignment, two blocks might be necessary to
        # cover the right boundary.
        if spec.has_face_conn(spec.X_HIGH) or spec.periodic_x:
            if bs - padding < bns:
                x_conns = 1    # 1 spec on the left, 1 spec on the right
            else:
                x_conns = 2    # 1 spec on the left, 2 specs on the right

        if spec.has_face_conn(spec.X_LOW) or spec.periodic_x:
            x_conns += 1

        y_conns = 0        # north-south
        if spec.has_face_conn(spec.Y_LOW) or spec.periodic_y:
            y_conns += 1
        if spec.has_face_conn(spec.Y_HIGH) or spec.periodic_y:
            y_conns += 1

        if self._spec.dim == 3:
            z_conns = 0        # top-bottom
            if spec.has_face_conn(spec.Z_LOW) or spec.periodic_z:
                z_conns += 1
            if spec.has_face_conn(spec.Z_HIGH) or spec.periodic_z:
                z_conns += 1

        # Number of specs to be handled by the boundary kernel.  This is also
        # the grid size for boundary kernels.  Note that the number of X-connection
        # specs does not have the 'bns' factor to account for the thickness of
        # the boundary layer, as the X connection is handled by whole compute
        # device specs which are assumed to be larger than the boundary layer
        # (i.e. bs > bns).
        if spec.dim == 2:
            self._boundary_blocks = (
                    (bns * grid_nx / bs) * y_conns +      # top & bottom
                    (arr_ny - y_conns * bns) * x_conns)  # left & right (w/o top & bottom rows)
            self._kernel_grid_bulk = [grid_nx - x_conns * bs, arr_ny - y_conns * bns]
            self._kernel_grid_full = [grid_nx / bs, arr_ny]
        else:
            self._boundary_blocks = (
                    grid_nx * arr_ny * bns / bs * z_conns +                    # T/B faces
                    grid_nx * (arr_nz - z_conns * bns) / bs * bns * y_conns +  # N/S faces
                    (arr_ny - y_conns * bns) * (arr_nz - z_conns * bns) * x_conns)
            self._kernel_grid_bulk = [
                    (grid_nx - x_conns * bs) * (arr_ny - y_conns * bns),
                    arr_nz - z_conns * bns]
            self._kernel_grid_full = [grid_nx * arr_ny / bs, arr_nz]

        if self._boundary_blocks >= 65536:
            # Use an artificial 2D grid to work around device limits.
            self._boundary_blocks = (4096, int(math.ceil(self._boundary_blocks / 4096.0)))
        else:
            self._boundary_blocks = (self._boundary_blocks, 1)

        self._kernel_grid_bulk[0] /= bs

        # Special cases: boundary kernels can cover the whole domain or this is
        # the only block participating in the simulation.
        if (0 in self._kernel_grid_bulk or self._kernel_grid_bulk[0] < 0 or
                self._kernel_grid_bulk[1] < 0 or len(self._spec._connectors) == 0 or
                not self.config.bulk_boundary_split):
            self.config.logger.debug("Disabling bulk/boundary split.")
            # Disable the boundary kernels and ensure that the bulk kernel will
            # cover the whole domain.
            self._boundary_blocks = None
            self._code_context['boundary_size'] = 0
            self._kernel_grid_bulk = self._kernel_grid_full

        self.config.logger.debug('Bulk grid: %s' % repr(self._kernel_grid_bulk))
        self.config.logger.debug('Boundary grid: %s' %
                repr(self._boundary_blocks))

        # Global grid size as seen by the simulation class.
        if self._spec.dim == 2:
            self._global_size = (self.config.lat_ny, self.config.lat_nx)
        else:
            self._global_size = (self.config.lat_nz, self.config.lat_ny,
                    self.config.lat_nx)

        evs = self._spec.envelope_size
        # Used so that face values map to the limiting coordinate
        # along a specific axis, e.g. lat_linear[X_LOW] = 0
        self.lat_linear = [0, self._lat_size[-1] - 1, 0, self._lat_size[-2] - 1]
        # As above, but the opposite face is used as an index in the list.
        self.lat_linear_with_swap = [self._lat_size[-1] - 1, 0,
                self._lat_size[-2] - 1, 0]
        # As lat_linear above, but the values in the list specify the location
        # of the first/last active node for the _opposite_ face to the
        # one used as an index in the list.
        self.lat_linear_dist = [self._lat_size[-1] - 1 - evs, evs,
                self._lat_size[-2] - 1 - evs, evs]
        # As lat_linear above, but the values in the list specify the
        # location of the first/last active node for the corresponding
        # axis.
        self.lat_linear_macro = [evs, self._lat_size[-1] - 1 - evs, evs,
                self._lat_size[-2] - 1 - evs]

        if self._spec.dim == 3:
            self.lat_linear.extend([0, self._lat_size[-3] - 1])
            self.lat_linear_with_swap.extend([self._lat_size[-3] - 1, 0])
            self.lat_linear_dist.extend([self._lat_size[-3] - 1 - evs, evs])
            self.lat_linear_macro.extend([evs, self._lat_size[-3] - 1 - evs])

    def _get_strides(self, type_):
        """Returns a list of strides for the NumPy array storing the lattice."""
        t = type_().nbytes
        return list(reversed(reduce(lambda x, y: x + [x[-1] * y],
                self._physical_size[-1:0:-1], [t])))

    @property
    def num_phys_nodes(self):
        """Returns the total number of actual nodes (including padding) in the lattice."""
        return reduce(operator.mul, self._physical_size)

    @property
    def num_nodes(self):
        """Returns the total number of lattice nodes (including ghosts,
        excluding padding."""
        return reduce(operator.mul, self._lat_size)

    def _get_dist_bytes(self, grid):
        """Returns the number of bytes required to store a single set of
           distributions for the whole simulation domain."""
        if self.config.node_addressing == 'indirect':
            return self._subdomain.active_nodes * grid.Q * self.float().nbytes
        else:
            return self.num_phys_nodes * grid.Q * self.float().nbytes

    def _get_global_idx(self, location, dist_num=0):
        """Returns a global index (in the distributions array).

        :param location: position of the node in the natural order
        :param dist_num: distribution number
        """
        if self.dim == 2:
            gx, gy = location
            arr_nx = self._physical_size[1]
            return gx + arr_nx * gy + (self.num_phys_nodes * dist_num)
        else:
            gx, gy, gz = location
            arr_nx = self._physical_size[2]
            arr_ny = self._physical_size[1]
            return ((gx + arr_nx * gy + arr_nx * arr_ny * gz) +
                    (self.num_phys_nodes * dist_num))

    def _idx_helper(self, gx, buf_slice, dists):
        """Returns a numpy array of global indices (in the subdomain coordinate
        system).

        :param gx: location along the x axis
        :param buf_slice: slice in the area perpendicular to the X axis
        :param dists: a list of distribution indices
        """
        sel = [slice(0, len(dists))]
        idx = np.mgrid[sel + list(reversed(buf_slice))].astype(np.uint32)
        for i, dist_num in enumerate(dists):
            idx[0][i,:] = dist_num
        if self.dim == 2:
            return self._get_global_idx((gx, idx[1]), idx[0]).astype(np.uint32)
        else:
            return self._get_global_idx((gx, idx[2], idx[1]),
                    idx[0]).astype(np.uint32)

    def _get_src_slice_indices(self, face, cpair, opposite=False):
        """Returns a numpy array of indices of sparse nodes from which
        data is to be collected.

        :param face: face ID for the connection
        :param cpair: ConnectionPair describing the connection
        :param opposite: if True, use opposite distributions than normally,
                and a location suitable for the fully local step in the AA
                access pattern
        """
        if face not in (self._spec.X_LOW, self._spec.X_HIGH):
            return None

        # For the AA access pattern, the locations of the nodes are the same
        # as for the macroscopic fields.
        if opposite:
            gx = self.lat_linear_macro[face]
            return self._idx_helper(gx, cpair.src.src_macro_slice,
                    [self._sim.grid.idx_opposite[d] for d in cpair.src.dists])
        else:
            gx = self.lat_linear[face]
            return self._idx_helper(gx, cpair.src.src_slice, cpair.src.dists)

    def _get_dst_slice_indices(self, face, cpair, opposite=False):
        """Returns a numpy array of indices of sparse nodes to which
        data is to be distributed.

        :param face: face ID for the connection
        :param cpair: ConnectionPair describing the connection
        :param opposite: if True, use opposite distributions than normally,
                and a location suitable for the fully local step in the AA
                access pattern
        """
        if face not in (self._spec.X_LOW, self._spec.X_HIGH):
            return None
        es = self._spec.envelope_size
        if opposite:
            if not cpair.src.dst_macro_slice:
                return None
            gx = self.lat_linear_with_swap[self._spec.opposite_face(face)]
            return self._idx_helper(gx, cpair.src.dst_macro_slice,
                    [self._sim.grid.idx_opposite[d] for d in cpair.dst.dists])
        else:
            if not cpair.dst.dst_slice:
                return None
            dst_slice = [
                slice(x.start + es, x.stop + es) for x in
                cpair.dst.dst_slice]
            gx = self.lat_linear_dist[self._spec.opposite_face(face)]
            return self._idx_helper(gx, dst_slice, cpair.dst.dists)

    def _dst_face_loc_to_full_loc(self, face, face_loc, opposite=False):
        """Expands a location tuple in the (full) face coordinate system into a
        a complete location tuple in the full coordinate system of the subdomain."""
        axis = self._spec.face_to_axis(face)
        # In the fully local step of the AA access pattern, the location along
        # the connection axis is different.
        if opposite:
            missing_loc = self.lat_linear_with_swap[self._spec.opposite_face(face)]
        else:
            missing_loc = self.lat_linear_dist[self._spec.opposite_face(face)]

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

        This has the form of a tuple:
        - a page-locked numpy buffer for the distributions
        - a numpy array of indices of nodes to which partial data is to be
          distributed
        - like the previous item, but for the fully local step in the AA access
          pattern
        - a selector to be applied to the receive buffer to get the partial
          distributions

        :param face: face ID for the connection
        :param cpair: ConnectionPair describing the connection
        """
        if cpair.dst.partial_nodes == 0:
            return None, None, None
        buf = self.backend.alloc_async_host_buf(cpair.dst.partial_nodes,
                dtype=self.float)
        idx = np.zeros(cpair.dst.partial_nodes, dtype=np.uint32)
        dst_low = [x + self._spec.envelope_size for x in cpair.dst.dst_low]
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
        for face, block_id in self._spec.connecting_subdomains():
            cpairs = self._spec.get_connections(face, block_id)
            for cpair in cpairs:
                coll_idx = self._get_src_slice_indices(face, cpair)
                coll_idx = GPUBuffer(coll_idx, self.backend)

                dist_full_idx = self._get_dst_slice_indices(face, cpair)
                dist_full_idx = GPUBuffer(dist_full_idx, self.backend)

                if self.config.access_pattern == 'AA':
                    coll_idx_opposite = self._get_src_slice_indices(face, cpair,
                            True)
                    coll_idx_opposite = GPUBuffer(coll_idx_opposite,
                            self.backend)

                    dist_full_idx_opposite = self._get_dst_slice_indices(face,
                            cpair, True)
                    dist_full_idx_opposite = GPUBuffer(dist_full_idx_opposite,
                            self.backend)
                else:
                    dist_full_idx_opposite = None
                    coll_idx_opposite = None

                for i, grid in enumerate(self._sim.grids):
                    # TODO(michalj): Optimize this by providing proper padding.
                    coll_buf = alloc(cpair.src.transfer_shape, dtype=self.float)
                    recv_buf = alloc(cpair.dst.transfer_shape, dtype=self.float)
                    dist_full_buf = alloc(cpair.dst.full_shape, dtype=self.float)

                    if self.config.access_pattern == 'AA':
                        local_coll_buf = GPUBuffer(alloc(cpair.src.local_transfer_shape, dtype=self.float),
                                self.backend)
                        local_recv_buf = GPUBuffer(alloc(cpair.dst.local_transfer_shape, dtype=self.float),
                                self.backend)
                    else:
                        local_coll_buf = None
                        local_recv_buf = None

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
                            dist_full_idx, i,
                            coll_idx_opposite,
                            dist_full_idx_opposite,
                            local_coll_buf,
                            local_recv_buf)

                    self.config.logger.debug('adding buffer for conn: {0} -> {1} '
                            '(face {2})'.format(self._spec.id, block_id, face))
                    self._block_to_connbuf[block_id].append(cbuf)

        # Explicitly sort connection buffers by their face ID.  Create a
        # separate dictionary where the order of the connection buffers
        # corresponds to that used by the _other_ subdomain.
        self._recv_block_to_connbuf = defaultdict(list)
        for subdomain_id, cbufs in self._block_to_connbuf.iteritems():
            cbufs.sort(key=lambda x: (x.face, x.grid_id))
            recv_bufs = list(cbufs)
            recv_bufs.sort(key=lambda x: (self._spec.opposite_face(x.face),
                x.grid_id))
            self._recv_block_to_connbuf[subdomain_id] = recv_bufs

    def _update_compute_code(self):
        code = self._bcg.get_code(self, self.backend.name)
        self.config.logger.debug("... compute code prepared.")
        self.module = self.backend.build(code)

    def _init_compute(self):
        self.config.logger.debug("Initializing compute unit...")
        self._update_compute_code()
        self.config.logger.debug("... compute code compiled.")
        self._init_streams()
        self.config.logger.debug("... done.")

    def _init_streams(self):
        self._data_stream = self.backend.make_stream()
        self._calc_stream = self.backend.make_stream()

    def _build_indirect_address_map(self):
        """Builds a node addressing map."""
        addr, _ = self.make_scalar_field(dtype=np.uint32, register=False, need_indirect=False,
                                         nonghost_view=False)
        addr[:] = 0xffffffff
        print addr.shape
        self._host_indirect_address = addr
        addr[self._subdomain.active_node_mask] = np.arange(self._subdomain.active_nodes)
        self._gpu_indirect_address = self.backend.alloc_buf(like=self._field_base[id(addr.base)])

    def _init_gpu_data_indirect(self):
        self._build_indirect_address_map()
        addr = self._host_indirect_address
        mask = self._subdomain.active_node_mask

        for field, sparse_field in zip(self._scalar_fields,
                                       self._sparse_scalar_fields):
            # Copy field to the sparse array.
            sparse_field[addr[mask]] = self._field_base[id(field.base)][mask]
            self._gpu_field_map[id(field)] = self.backend.alloc_buf(
                like=sparse_field,
                wrap_in_array=(id(field.base) in self._array_fields))

        for field, sparse_field in zip(self._vector_fields,
                                       self._sparse_vector_fields):
            gpu_vector = []
            for component, sparse_component in zip(field, sparse_field):
                # Copy field to the sparse array.
                sparse_component[addr[mask]] = self._field_base[id(component.base)][mask]
                gpu_vector.append(self.backend.alloc_buf(
                    like=sparse_component,
                    wrap_in_array=(id(component.base) in self._array_fields)))
            self._gpu_field_map[id(field)] = gpu_vector

        self._gpu_geo_map = self.backend.alloc_buf(
                like=self._subdomain.encoded_map(addr))

    def _unravel_fields(self):
        """Copies data from sparse arrays back into the dense arrays used for
        operations on the host."""
        addr = self._host_indirect_address
        mask = self._subdomain.active_node_mask
        for field, sparse_field in zip(self._scalar_fields,
                                       self._sparse_scalar_fields):
             self._field_base[id(field.base)][mask] = sparse_field[addr[mask]]

        for field, sparse_field in zip(self._vector_fields,
                                       self._sparse_vector_fields):
            for component, sparse_component in zip(field, sparse_field):
                self._field_base[id(component.base)][mask] = sparse_component[addr[mask]]

    def _init_gpu_data_direct(self):
        for field in self._scalar_fields:
            self._gpu_field_map[id(field)] = self.backend.alloc_buf(
                like=self._field_base[id(field.base)],
                wrap_in_array=(id(field.base) in self._array_fields))

        for field in self._vector_fields:
            gpu_vector = []
            for component in field:
                gpu_vector.append(self.backend.alloc_buf(
                    like=self._field_base[id(component.base)],
                    wrap_in_array=(id(component.base) in self._array_fields)))
            self._gpu_field_map[id(field)] = gpu_vector

        self._gpu_geo_map = self.backend.alloc_buf(
                like=self._subdomain.encoded_map())

    def _init_gpu_data(self):
        self.config.logger.debug("Initializing compute unit data.")

        if self.config.node_addressing == 'indirect':
            self._init_gpu_data_indirect()
        else:
            self._init_gpu_data_direct()

        for grid in self._sim.grids:
            size = self._get_dist_bytes(grid)
            self.config.logger.debug("Using {0} bytes for buffer".format(size))
            self._gpu_grids_primary.append(self.backend.alloc_buf(size=size))
            if self.config.access_pattern == 'AB':
                self._gpu_grids_secondary.append(self.backend.alloc_buf(size=size))

        if self._subdomain.scratch_space_size > 0:
            self.config.logger.debug("Using {0} scratch space slots".format(
                self._subdomain.scratch_space_size))
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
            if self.config.access_pattern == 'AB':
                return self._gpu_grids_secondary[num]
            # XXX: we should simply not pass the arguments to the kernels
            # instead
            else:
                return self._gpu_grids_primary[num]

    def gpu_geo_map(self):
        return self._gpu_geo_map

    def gpu_indirect_address(self):
        return self._gpu_indirect_address

    def get_kernel(self, name, args, args_format, block_size=None,
            needs_iteration=False, shared=0, more_shared=False):
        if block_size is None:
            block = self._kernel_block_size
        else:
            block = block_size
        return self.backend.get_kernel(self.module, name, args=args,
                args_format=args_format, block=block,
                needs_iteration=needs_iteration, shared=shared,
                more_shared=more_shared)

    def exec_kernel(self, name, args, args_format, needs_iteration=False,
                    grid=None, block_size=None):
        kernel = self.get_kernel(name, args, args_format,
                needs_iteration=needs_iteration, block_size=block_size)
        self.backend.run_kernel(kernel, self._kernel_grid_full if grid is None
                                else grid)

    def step(self, sync_req):
        self._step_boundary(sync_req)
        self._step_bulk(sync_req)
        self._sim.iteration += 1
        self._send_dists()
        if sync_req:
            self._step_aux()
        # Run this at a point after the compute step is fully scheduled for execution
        # on the GPU and where it doesn't unnecessarily delay other operations.
        self.backend.set_iteration(self._sim.iteration)
        self._recv_dists()
        self._profile.record_gpu_start(TimeProfile.DISTRIB, self._data_stream)
        for kernel, grid in self._distrib_kernels[self._sim.iteration & 1]:
            self.backend.run_kernel(kernel, grid, self._data_stream)
        self._profile.record_gpu_end(TimeProfile.DISTRIB, self._data_stream)

    def _get_bulk_kernel(self, sync_req):
        if sync_req:
            kernel = self._kernels_bulk_full[self._sim.iteration & 1][0]
        else:
            kernel = self._kernels_bulk_none[self._sim.iteration & 1][0]

        return kernel, self._kernel_grid_bulk

    def _apply_pbc(self, kernels):
        base = 1 - (self._sim.iteration & 1)
        ceil = math.ceil
        ls = self._lat_size
        bs = self.config.block_size

        if self._spec.periodic_x:
            if self._spec.dim == 2:
                grid_size = (int(ceil(ls[0] / float(bs))), 1)
            else:
                grid_size = (int(ceil(ls[1] / float(bs))), ls[0])
            for kernel in kernels[base][0]:
                self.backend.run_kernel(kernel, grid_size, self._calc_stream)

        if self._spec.periodic_y:
            if self._spec.dim == 2:
                grid_size = (int(ceil(ls[1] / float(bs))), 1)
            else:
                grid_size = (int(ceil(ls[2] / float(bs))), ls[0])
            for kernel in kernels[base][1]:
                self.backend.run_kernel(kernel, grid_size, self._calc_stream)

        if self._spec.dim == 3 and self._spec.periodic_z:
            grid_size = (int(ceil(ls[2] / float(bs))), ls[1])
            for kernel in kernels[base][2]:
                self.backend.run_kernel(kernel, grid_size, self._calc_stream)

    def _step_bulk(self, sync_req):
        """Runs one simulation step in the bulk domain.

        Bulk domain is defined to be all nodes that belong to CUDA
        blocks that do not depend on input from any ghost nodes.
        """
        # The bulk kernel only needs to be run if the simulation has a bulk/boundary split.
        # If this split is not present, the whole domain is simulated in _step_boundary and
        # _step_bulk only needs to handle PBC (below).
        self._profile.record_gpu_start(TimeProfile.BULK, self._calc_stream)
        if self._boundary_blocks is not None:
            kernel, grid = self._get_bulk_kernel(sync_req)
            self.backend.run_kernel(kernel, grid, self._calc_stream)

        self._apply_pbc(self._pbc_kernels)
        self._profile.record_gpu_end(TimeProfile.BULK, self._calc_stream)

    def _step_boundary(self, sync_req):
        """Runs one simulation step for the boundary blocks.

        Boundary blocks are CUDA blocks that depend on input from
        ghost nodes."""

        if self._boundary_blocks is not None:
            if sync_req:
                kernel = self._kernels_bnd_full[self._sim.iteration & 1][0]
            else:
                kernel = self._kernels_bnd_none[self._sim.iteration & 1][0]
            grid = self._boundary_blocks
        else:
            # Run bulk kernel is there is no bulk/boundary split.
            kernel, grid = self._get_bulk_kernel(sync_req)

        blk_str = self._calc_stream

        self._profile.record_gpu_start(TimeProfile.BOUNDARY, blk_str)
        self.backend.run_kernel(kernel, grid, blk_str)
        ev = self._profile.record_gpu_end(TimeProfile.BOUNDARY, blk_str,
                                          need_event=True)

        # Enqueue a wait so that the data collection will not start until the
        # kernel handling boundary calculations is completed (that kernel runs
        # in the calc stream so that it is automatically synchronized with
        # bulk calculations).
        self._data_stream.wait_for_event(ev)
        self._profile.record_gpu_start(TimeProfile.COLLECTION, self._data_stream)
        for kernel, grid in self._collect_kernels[self._sim.iteration & 1]:
            self.backend.run_kernel(kernel, grid, self._data_stream)
        self._profile.record_gpu_end(TimeProfile.COLLECTION, self._data_stream)

    def _step_aux(self):
        for kernel in self._aux_kernels[(self._sim.iteration - 1) & 1]:
            self.backend.run_kernel(kernel, self._kernel_grid_full, self._data_stream)

    @profile(TimeProfile.SEND_DISTS)
    def _send_dists(self):
        if not self._spec._connectors:
            return

        if self.config.access_pattern == 'AA' and self._sim.iteration & 1:
            buf = 'local_coll_buf'
        else:
            buf = 'coll_buf'

        for b_id, connector in self._spec._connectors.iteritems():
            conn_bufs = self._block_to_connbuf[b_id]
            for x in conn_bufs:
                self.backend.from_buf_async(getattr(x, buf).gpu, self._data_stream)

        self.backend.sync_stream(self._data_stream)

        for b_id, connector in self._spec._connectors.iteritems():
            conn_bufs = self._block_to_connbuf[b_id]

            if len(conn_bufs) > 1:
                connector.send(np.hstack(
                    [np.ravel(getattr(x, buf).host) for x in conn_bufs]))
            else:
                # TODO(michalj): Use non-blocking sends here?
                connector.send(np.ravel(getattr(conn_bufs[0], buf).host).copy())

    @profile(TimeProfile.RECV_DISTS)
    def _recv_dists(self):
        def distribute(cbuf):
            # _recv_dists is called after the iteration counter has been updated.
            if self.config.access_pattern == 'AA' and self._sim.iteration & 1:
                cbuf.distribute_unpropagated(self.backend, self._data_stream)
            else:
                cbuf.distribute(self.backend, self._data_stream)

        if self.config.access_pattern == 'AA' and self._sim.iteration & 1:
            get_buf = operator.attrgetter('local_recv_buf.host')
        else:
            get_buf = operator.attrgetter('recv_buf')

        for b_id, connector in self._spec._connectors.iteritems():
            conn_bufs = self._recv_block_to_connbuf[b_id]
            if len(conn_bufs) > 1:
                dest = np.hstack([np.ravel(get_buf(x)) for x in conn_bufs])
                self._profile.record_cpu_start(TimeProfile.NET_RECV)
                # Returns false only if quit event is active.
                if not connector.recv(dest, self._quit_event):
                    return

                self._profile.record_cpu_end(TimeProfile.NET_RECV)
                i = 0
                for cbuf in conn_bufs:
                    recv_buf = get_buf(cbuf)
                    l = recv_buf.size
                    recv_buf[:] = dest[i:i+l].reshape(recv_buf.shape)
                    i += l
                    distribute(cbuf)
            else:
                cbuf = conn_bufs[0]
                recv_buf = get_buf(cbuf)
                dest = np.ravel(recv_buf)

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
                    recv_buf[:] = dest.reshape(recv_buf.shape)
                distribute(cbuf)

    def _fields_to_host(self):
        """Copies data for all fields from the GPU to the host."""
        for field in self._scalar_fields:
            self.backend.from_buf_async(self.gpu_field(field), self._calc_stream)

        for field in self._vector_fields:
            for component in self.gpu_field(field):
                self.backend.from_buf_async(component, self._calc_stream)

    def _init_collect_kernels(self, cbuf, grid_dim1, block_size):
        """Returns collection kernels for a connection.

        The iteration number used for indexing the kernel lists is the same
        as the one used for the simulation kernels.

        :param cbuf: ConnectionBuffer
        :param grid_dim1: callable returning the size of the CUDA grid
            given the total size
        :param block_size: CUDA block size for the kernels

        Returns: primary, secondary.
        """
        # Sparse data collection.
        if cbuf.coll_idx.host is not None:
            def _get_sparse_coll_kernel(i, idx_buffer=cbuf.coll_idx.gpu,
                    coll_buf=cbuf.coll_buf):
                return KernelGrid(
                    self.get_kernel('CollectSparseData',
                    [idx_buffer, self.gpu_dist(cbuf.grid_id, i),
                     coll_buf.gpu, coll_buf.host.size],
                    'PPPi', (block_size,)),
                    grid=(grid_dim1(coll_buf.host.size),))

            if self.config.access_pattern == 'AA':
                return (_get_sparse_coll_kernel(0, cbuf.coll_idx_opposite.gpu, cbuf.local_coll_buf),
                        _get_sparse_coll_kernel(0))
            else:
                return _get_sparse_coll_kernel(1), _get_sparse_coll_kernel(0)
        # Continuous data collection.
        else:
            def _get_cont_coll_kernel(i, kernel='CollectContinuousData', coll_buf=cbuf.coll_buf,
                    src_slice=cbuf.cpair.src.src_slice):
                # [X, Z * dists] or [X, Y * dists]
                min_max = ([x.start for x in src_slice] +
                        list(reversed(coll_buf.host.shape[1:])))
                min_max[-1] = min_max[-1] * len(cbuf.cpair.src.dists)
                if self.dim == 2:
                    signature = 'PiiiP'
                    grid_size = (grid_dim1(coll_buf.host.size),)
                else:
                    signature = 'PiiiiiP'
                    grid_size = (grid_dim1(coll_buf.host.shape[-1]),
                        coll_buf.host.shape[-2] * len(cbuf.cpair.src.dists))

                return KernelGrid(
                    self.get_kernel(kernel,
                    [self.gpu_dist(cbuf.grid_id, i),
                     cbuf.face] + min_max + [coll_buf.gpu],
                     signature, (block_size,)),
                     grid_size)

            if self.config.access_pattern == 'AA':
                return (_get_cont_coll_kernel(0, 'CollectContinuousDataWithSwap', cbuf.local_coll_buf,
                                              src_slice=cbuf.cpair.src.src_macro_slice),
                        _get_cont_coll_kernel(0))
            else:
                return _get_cont_coll_kernel(1), _get_cont_coll_kernel(0)

    def _init_distrib_kernels(self, cbuf, grid_dim1, block_size):
        """Returns distribution kernels for a connection.

        The iteration number used for indexing the kernel lists is +1
        wrt to the iteration number used for the simulation kernels in the
        same step.

        :param cbuf: ConnectionBuffer
        :param grid_dim1: callable returning the size of the CUDA grid
            given the total size
        :param block_size: CUDA block size for the kernels

        Returns: lists of: primary, secondary.
        """
        primary = []
        secondary = []

        primary, secondary = self._init_distrib_kernels_ab(cbuf, grid_dim1,
                block_size)

        if self.config.access_pattern == 'AB':
            return primary, secondary

        secondary = []

        if cbuf.dist_full_idx_opposite.host is not None:
            grid_size = (grid_dim1(cbuf.local_recv_buf.host.size),)
            secondary.append(KernelGrid(
                    self.get_kernel('DistributeSparseData',
                        [cbuf.dist_full_idx_opposite.gpu,
                         self.gpu_dist(cbuf.grid_id, 0),
                         cbuf.local_recv_buf.gpu,
                         cbuf.local_recv_buf.host.size],
                    'PPPi', (block_size,)),
                    grid_size))
        else:
            min_max = ([y.start for y in cbuf.cpair.src.dst_macro_slice] +
                list(reversed(cbuf.local_recv_buf.host.shape[1:])))
            min_max[-1] = min_max[-1] * len(cbuf.cpair.dst.dists)

            if self.dim == 2:
                signature = 'PiiiP'
                grid_size = (grid_dim1(cbuf.local_recv_buf.host.size),)
            else:
                signature = 'PiiiiiP'
                grid_size = (grid_dim1(cbuf.local_recv_buf.host.shape[-1]),
                    cbuf.local_recv_buf.host.shape[-2] * len(cbuf.cpair.dst.dists))

            secondary.append(KernelGrid(
                    self.get_kernel('DistributeContinuousDataWithSwap',
                            [self.gpu_dist(cbuf.grid_id, 0),
                             self._spec.opposite_face(cbuf.face)] +
                            min_max + [cbuf.local_recv_buf.gpu],
                            signature, (block_size,)),
                            grid_size))

        return primary, secondary

    def _init_distrib_kernels_ab(self, cbuf, grid_dim1, block_size):
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
                min_max = ([y.start + self._spec.envelope_size
                            for y in cbuf.cpair.dst.dst_slice] +
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
                             self._spec.opposite_face(cbuf.face)] +
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

    def _debug_get_dist(self, output=True, grid_num=0):
        """Copies the distributions from the GPU to a properly structured host array.
        :param output: if True, returns the contents of the distributions set *after*
                the current simulation step
        """
        iter_idx = self._sim.iteration & 1
        if not output:
            iter_idx = 1 - iter_idx

        self.config.logger.debug('getting dist for grid {0} iter={1} ({2})'.format(
            grid_num, iter_idx, self.gpu_dist(grid_num, iter_idx)))
        dbuf = np.zeros(self._get_dist_bytes(self._sim.grid) / self.float().nbytes,
            dtype=self.float)
        self.backend.from_buf(self.gpu_dist(grid_num, iter_idx), dbuf)
        dbuf = dbuf.reshape([self._sim.grid.Q] + self._physical_size)
        return dbuf

    def _debug_set_dist(self, dbuf, output=True, grid_num=0):
        iter_idx = self._sim.iteration & 1
        if not output:
            iter_idx = 1 - iter_idx

        self.backend.to_buf(self.gpu_dist(grid_num, iter_idx), dbuf)

    def _debug_global_idx_to_tuple(self, gi):
        dist_num = gi / self.num_phys_nodes
        rest = gi % self.num_phys_nodes
        arr_nx = self._physical_size[-1]
        gx = rest % arr_nx
        gy = rest / arr_nx
        return dist_num, gy, gx

    def send_summary_info(self, timing_info, min_timings, max_timings):
        if self._summary_sender is not None:
            self._summary_sender.send_pyobj((timing_info, min_timings,
                    max_timings, self._subdomain.active_nodes))
            self.config.logger.debug('Sending timing information to controller.')
            assert self._summary_sender.recv() == 'ack'

    def _gpu_initial_conditions(self):
        self._sim.verify_fields()
        # Applies initial conditions on the GPU.
        self._sim.initial_conditions(self)

    def save_checkpoint(self):
        if self.config.single_checkpoint:
            fname = io.checkpoint_filename(self.config.checkpoint_file,
                    1, self._spec.id, 0)
        else:
            fname = io.checkpoint_filename(self.config.checkpoint_file,
                    io.filename_iter_digits(self.config.max_iters),
                    self._spec.id, self._sim.iteration)

        sim_state = pickle.dumps(self._sim.get_state(), -1)
        data = { 'state': sim_state }

        for i in range(len(self._sim.grids)):
            data['dist{0}a'.format(i)] = self._debug_get_dist(True, i)
            data['dist{0}b'.format(i)] = self._debug_get_dist(False, i)

        np.savez(fname, **data)

    def restore_checkpoint(self, fname):
        self.config.logger.info('Restoring checkpoint')

        cpoint = np.load(fname)
        sim_state = pickle.loads(str(cpoint['state']))
        self._sim.set_state(sim_state)

        for k, v in cpoint.iteritems():
            if not k.startswith('dist'):
                continue

            is_primary = k.endswith('a')
            dist_num = int(k[4:-1])

            self._debug_set_dist(v, is_primary, dist_num)

    def _prepare_compute_kernels(self):
        gck = self._sim.get_compute_kernels

        self._kernels_bulk_full = gck(self, True, True)
        self._kernels_bulk_none = gck(self, False, True)
        self._kernels_bnd_full = gck(self, True, False)
        self._kernels_bnd_none = gck(self, False, False)

    def _init_force_objects(self):
        """Prepares GPU data structures for tracking momentum exchange
        between fluid and solid objects."""

        if self._sim.force_objects:
            self.config.logger.info('Processing force objects.')

        for fo in self._sim.force_objects:
            dists = self._subdomain.get_fo_distributions(fo)
            if not dists:
                self.config.logger.warning(
                    'No momentum-transferring distributions found for %s' % fo)
                continue

            idxs = np.array([], dtype=np.int32)
            idxs_opp = np.array([], dtype=np.int32)
            for dist_num, locs in sorted(dists.iteritems()):
                opp_idx = self._subdomain.grid.idx_opposite[dist_num]

                # Momentum transferred to the solid node.
                idxs = np.concatenate((idxs, self._get_global_idx(
                    tuple(reversed(locs)), opp_idx).astype(np.int32)))

                dist = self._subdomain.grid.basis[dist_num]

                shifted_locs = []
                for i, loc in enumerate(reversed(locs)):
                    shifted_locs.append(loc + int(dist[i]))

                # Momentum transferred from the solid node.
                idxs_opp = np.concatenate((idxs_opp, self._get_global_idx(
                    shifted_locs, dist_num).astype(np.int32)))

            components = np.zeros((self.dim, idxs.size), dtype=np.int32)
            h = 0
            for dist_num, locs in sorted(dists.iteritems()):
                opp_idx = self._subdomain.grid.idx_opposite[dist_num]
                ei = self._subdomain.grid.basis[opp_idx]
                for i, ei_comp in enumerate(ei):
                    components[i, h:h + locs[0].size] = int(ei_comp)
                h += locs[0].size

            self.config.logger.debug('%s: total momentum links: %d' % (
                fo, len(idxs)))
            fo._components_map = components
            h = np.zeros_like(idxs, dtype=self.float)
            fo.gpu_force_buf = self.backend.alloc_buf(like=h)
            fo.force_buf = h
            fo.gpu_idx_buf = self.backend.alloc_buf(like=idxs)
            fo.gpu_opp_idx_buf = self.backend.alloc_buf(like=idxs_opp)

    # TODO(michalj): Make sure the kernels are cached.
    def update_force_objects(self):
        """
        The kernels called in this function read data *after* propagation.
        """
        for fo in self._sim.force_objects:
            if not fo.initialized:
                continue

            self.exec_kernel('ComputeForceObjects',
                             [fo.gpu_idx_buf, fo.gpu_opp_idx_buf,
                              self.gpu_dist(0, self._sim.iteration & 1),
                              fo.gpu_force_buf, fo.force_buf.size],
                             'PPPPi',
                             block_size=128,
                             grid=[(fo.force_buf.size + 127) / 128])

    def sighup_handler(self, signum, frame):
        self.config.logger.info('Received HUP signal, will save checkpoint (it=%d).' % self._sim.iteration)
        self._checkpoint_req = 2

    def _install_signal_handlers(self):
        signal.signal(signal.SIGHUP, self.sighup_handler)

    def run(self):
        self.config.logger.info("Initializing subdomain.")
        self.config.logger.debug(self.backend.info)

        self._init_geometry()

        # Creates scalar fields on the host. They are used for gpu-host
        # communication and for specifing initial conditions.
        self._sim.init_fields(self)
        self._init_buffers()
        self._init_compute()
        self.config.logger.debug("Initializing macroscopic fields.")
        self._subdomain.init_fields(self._sim)
        self._init_gpu_data()
        self._init_force_objects()
        self.config.logger.debug("Initializing GPU kernels.")

        self._init_interblock_kernels()
        self._prepare_compute_kernels()
        if self._spec.periodic:
            self._pbc_kernels = self._sim.get_pbc_kernels(self)
        self._aux_kernels = self._sim.get_aux_kernels(self)

        self.config.logger.debug("Applying initial conditions.")
        self._gpu_initial_conditions()

        # Save initial state of the simulation.
        if self.config.output and self.config.from_ == 0:
            self._output.save(self._sim.iteration)

        if not self.config.max_iters:
            self.config.logger.warning("Running infinite simulation.")

        if self.config.restore_from:
            self.restore_checkpoint(io.subdomain_checkpoint(
                self.config.restore_from, self._spec.id))

        if self._initialization:
            self.initialize()

        self._sim.before_main_loop(self)
        # Allow mix-ins to have their own before_main_loop routines.
        for c in self._sim.__class__.mro()[1:]:
            if issubclass(c, LBMixIn) and hasattr(c, 'before_main_loop'):
                c.before_main_loop(self._sim, self)

        self._install_signal_handlers()

        self.config.logger.info("Starting simulation.")
        self.main()

        self.config.logger.info(
            "Simulation completed after {0} iterations.".format(
                self._sim.iteration))

    def need_quit(self):
        if (self.config.max_iters > 0 and
            self._sim.iteration >= self.config.max_iters):
            return True

        # The quit event is used by the visualization interface.
        if self._quit_event.is_set():
            self.config.logger.info("Simulation termination requested.")
            return True

        if os.name != 'nt' and self._ppid != os.getppid():
            self.config.logger.info("Master process is dead -- terminating simulation.")
            return True

        return False

    def initialize(self):
        self.config.logger.info("Consistent intialization started.")
        visc = self.config.visc
        self.config.visc = 1.0/6.0
        try:
            for init_it in xrange(0, self.config.init_iters):
                self.step(False)
                self._data_stream.synchronize()
                self._calc_stream.synchronize()
                self._sim.iteration = 0
                # TODO: Add an option to stop when:
                # ||rho(t) - rho(t-1)|| < eps
        except self.backend.FatalError:
            self.config.logger.exception("Fatal on-device error at iteration "
                    "{0}.".format(self._sim.iteration))
            self.config.logger.error("Requesting quit.")
            self._quit_event.set()

        self.config.logger.info("Initialization phase complete.")
        self._sim.iteration = 0
        self.config.visc = visc

        # TODO: Make it possible to cache all fields and distributions here.
        # Rebuild the compute code for the real run.
        self._initialization = False
        self._update_compute_code()
        self._prepare_compute_kernels()
        self._initial_conditions()

    def _handle_geo_updates(self):
        q = self._spec.geo_queue
        if q.empty():
            return
        hx, hy = self._subdomain._get_mgrid()
        while not q.empty():
            (x, y), node_type = q.get_nowait()
            self._subdomain.update_node((hy == y) & (hx == x),
                                        nt.NTFullBBWall(orientation=np.int32(0))
                                        if node_type else nt._NTFluid)
        self.backend.to_buf(self._gpu_geo_map)

    _checkpoint_req = 0
    t_prev_checkpoint = 0.0
    def main(self):
        is_quit = False

        try:
            self._profile.record_start()
            while True:
                self._profile.start_step()

                output_req = self._sim.need_output()
                sync_req, fields_req = self._sim.need_sync_fields()

                # Distribution dumping.
                if sync_req and self.config.debug_dump_dists:
                    bufs = []
                    for i in range(len(self._sim.grids)):
                        bufs.append(self._debug_get_dist(self, grid_num=i))
                    self._output.dump_dists(bufs, self._sim.iteration)
                    del bufs

                # Updates the iteration number.
                self.step(fields_req)

                if sync_req:
                    self._fields_to_host()

                # Periodically log effective performance.
                pse = self.config.perf_stats_every
                if (pse > 0 and self._sim.iteration % pse == 0):
                    t_now = time.time()
                    if self.t_prev_checkpoint > 0.0:
                        dt = t_now - self.t_prev_checkpoint
                        mlups = self._subdomain.active_nodes * pse / dt * 1e-6
                        self.config.logger.info(
                            "iteration:{0}  speed:{1:.2f} MLUPS".format(
                                self._sim.iteration, mlups))
                    self.t_prev_checkpoint = t_now

                if self.need_quit():
                    break

                # External geometry updates (from the frontend).
                if self._spec.geo_queue is not None:
                    self._handle_geo_updates()

                # Wait for calculations to complete. All code handling misc host
                # tasks should be above this line to minimize performance
                # impact.
                self.backend.sync_stream(self._data_stream, self._calc_stream)

                if sync_req and self._host_indirect_address is not None:
                    self._unravel_fields()

                if output_req:
                    if (self.config.check_invalid_results_host and
                        not self._output.verify()):
                        self.config.logger.error("Invalid value detected in "
                                "output for iteration {0}".format(
                                self._sim.iteration))
                        self._quit_event.set()
                        break
                    self._output.save(self._sim.iteration)
                self._profile.end_step()

                if self.config.checkpoint_file and (
                        self._sim.need_checkpoint() or self._checkpoint_req > 0):
                    self._checkpoint_req -= 1
                    self.save_checkpoint()

                self._sim.after_step(self)

            # Receive any data from remote nodes prior to termination.  This ensures
            # we don't run into problems with zmq.
            self._data_stream.synchronize()
            self._calc_stream.synchronize()
            if output_req:
                if self._host_indirect_address is not None:
                    self._unravel_fields()
                self._output.save(self._sim.iteration)

            self._profile.record_end()

            if (self._sim.iteration >= self.config.max_iters and
                    self.config.checkpoint_file and self.config.final_checkpoint):
                self.save_checkpoint()

            self._sim.after_main_loop(self)

        except self.backend.FatalError:
            is_quit = True
            self.config.logger.exception("Fatal on-device error at iteration "
                    "{0}.".format(self._sim.iteration))

        if is_quit:
            self.config.logger.error("Requesting quit.")
            self._quit_event.set()

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
        for b_id, connector in self._spec._connectors.iteritems():
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
        for b_id, connector in self._spec._connectors.iteritems():
            conn_bufs = self._block_to_macrobuf[b_id]
            for x in conn_bufs:
                self.backend.from_buf_async(x.coll_buf.gpu, self._data_stream)

        self.backend.sync_stream(self._data_stream)

        for b_id, connector in self._spec._connectors.iteritems():
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
        if face in (self._spec.X_LOW, self._spec.X_HIGH):
            gx = self.lat_linear_macro[face]
        else:
            return None
        return self._macro_idx_helper(gx, cpair.src.src_macro_slice)

    def _get_dst_macro_indices(self, face, cpair):
        if face in (self._spec.X_LOW, self._spec.X_HIGH):
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
        for face, block_id in self._spec.connecting_subdomains():
            cpairs = self._spec.get_connections(face, block_id)
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
            recv_bufs.sort(key=lambda x: self._spec.opposite_face(x.face))
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

    def _gpu_initial_conditions(self):
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
        super(NNSubdomainRunner, self)._gpu_initial_conditions()

    def step(self, sync_req):
        """Runs one simulation step."""

        if sync_req:
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
        ev = record_gpu_end(TimeProfile.MACRO_BOUNDARY, str_calc,
                            need_event=True)
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

        ev = record_gpu_end(TimeProfile.MACRO_DISTRIB, str_data, need_event=True)
        str_calc.wait_for_event(ev)

        # Actual simulation step.
        record_gpu_start(TimeProfile.BOUNDARY, str_calc)
        if has_boundary_split:
            for k in bnd_kernel_sim:
                run(k, grid_bnd, str_calc)
        else:
            for k in bulk_kernel_sim:
                run(k, grid_bulk, str_calc)
        ev = record_gpu_end(TimeProfile.BOUNDARY, str_calc, need_event=True)
        str_data.wait_for_event(ev)

        record_gpu_start(TimeProfile.COLLECTION, str_data)
        for kernel, grid in self._collect_kernels[it & 1]:
            run(kernel, grid, str_data)
        record_gpu_end(TimeProfile.COLLECTION, str_data)

        record_gpu_start(TimeProfile.BULK, str_calc)
        if has_boundary_split:
            for k in bulk_kernel_sim:
                run(k, grid_bulk, str_calc)
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


"""Code for controlling a single block of a LB simulation."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict, namedtuple
import math
import operator
import numpy as np
import zmq
from sailfish import codegen, util

# Used to hold a reference to a CUDA kernel and a grid on which it is
# to be executed.
KernelGrid = namedtuple('KernelGrid', 'kernel grid')
TimingInfo = namedtuple('TimingInfo', 'comp data recv send total block_id')


class ConnectionBuffer(object):
    def __init__(self, face, cpair, coll_buf, coll_idx, recv_buf,
            dist_partial_buf, dist_partial_idx, dist_partial_sel,
            dist_full_buf, dist_full_idx):
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

    def distribute(self, backend):
        if self.dist_partial_sel is not None:
            self.dist_partial_buf.host[:] = self.recv_buf[self.dist_partial_sel]
            backend.to_buf(self.dist_partial_buf.gpu)

        if self.cpair.dst.dst_slice:
            slc = [slice(0, self.recv_buf.shape[0])] + list(reversed(self.cpair.dst.dst_full_buf_slice))
            self.dist_full_buf.host[:] = self.recv_buf[slc]
            backend.to_buf(self.dist_full_buf.gpu)

class GPUBuffer(object):
    """Numpy array and a corresponding GPU buffer."""
    def __init__(self, host_buffer, backend):
        self.host = host_buffer
        if host_buffer is not None:
            self.gpu = backend.alloc_buf(like=host_buffer)
        else:
            self.gpu = None


class BlockRunner(object):
    """Runs the simulation for a single LBBlock.
    """
    def __init__(self, simulation, block, output, backend, quit_event, summary_addr):
        # Create a 2-way connection between the LBBlock and this BlockRunner
        self._ctx = zmq.Context()
        self._summary_sender = self._ctx.socket(zmq.REQ)
        self._summary_sender.connect(summary_addr)

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

    @property
    def config(self):
        return self._sim.config

    @property
    def dim(self):
        return self._block.dim

    def update_context(self, ctx):
        """Called by the codegen module."""
        self._block.update_context(ctx)
        self._geo_block.update_context(ctx)
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
        else:
            ctx['lat_nz'] = 1
            ctx['arr_nz'] = 1
            periodic_z = 0
            bnd_limits.append(1)

        ctx['lat_linear'] = self.lat_linear
        ctx['lat_linear_dist'] = self.lat_linear_dist

        ctx['periodic_x'] = 0 #int(self._block.periodic_x)
        ctx['periodic_y'] = 0 #int(self._block.periodic_y)
        ctx['periodic_z'] = 0 #periodic_z
        ctx['periodicity'] = [0, 0, 0]

#[int(self._block.periodic_x), int(self._block.periodic_y), periodic_z]

        ctx['bnd_limits'] = bnd_limits
        ctx['dist_size'] = self._get_nodes()
        ctx['sim'] = self._sim
        ctx['block'] = self._block

        # FIXME Additional constants.
        ctx['constants'] = []

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

    def make_scalar_field(self, dtype=None, name=None):
        """Allocates a scalar NumPy array.

        The array includes padding adjusted for the compute device (hidden from
        the end user), as well as space for any ghost nodes.  The returned
        object is a view into the underlying array that hides all ghost nodes.
        Ghost nodes can still be accessed via the 'base' attribute of the
        returned ndarray."""
        if dtype is None:
            dtype = self.float

        size = self._get_nodes()
        strides = self._get_strides(dtype)

        field = np.ndarray(self._physical_size, buffer=np.zeros(size, dtype=dtype),
                           dtype=dtype, strides=strides)
        fview = field[self._block._nonghost_slice]

        if name is not None:
            self._output.register_field(fview, name)

        self._scalar_fields.append(fview)
        return fview

    def make_vector_field(self, name=None, output=False):
        """Allocates several scalar arrays representing a vector field."""
        components = []
        view_components = []

        for x in range(0, self._block.dim):
            field = self.make_scalar_field(self.float)
            components.append(field)

        if name is not None:
            self._output.register_field(components, name)

        self._vector_fields.append(components)
        return components

    def visualization_map(self):
        if self._vis_map_cache is None:
            self._vis_map_cache = self._geo_block.visualization_map()
        return self._vis_map_cache

    def _init_geometry(self):
        self.config.logger.debug("Initializing geometry.")
        self._init_shape()
        self._geo_block = self._sim.geo(self._global_size, self._block,
                                        self._sim.grid)
        self._geo_block.reset()

    def _init_shape(self):
        # Logical size of the lattice (including ghost nodes).
        # X dimension is the last one on the list (nz, ny, nx)
        self._lat_size = list(reversed(self._block.actual_size))

        # Physical in-memory size of the lattice, adjusted for optimal memory
        # access from the compute unit.  Size of the X dimension is rounded up
        # to a multiple of block_size.  Order is [nz], ny, nx
        self._physical_size = list(reversed(self._block.actual_size))
        self._physical_size[-1] = (int(math.ceil(float(self._physical_size[-1]) /
                                                 self.config.block_size)) *
                                       self.config.block_size)

        # CUDA block/grid size for standard kernel call.
        if self._block.dim == 2:
            self._kernel_grid_size = list(reversed(self._physical_size))
        else:
            self._kernel_grid_size = [self._physical_size[2] *
                    self._physical_size[1],  self._physical_size[0]]
        self._kernel_grid_size[0] /= self.config.block_size

        self._kernel_block_size = (self.config.block_size, 1)

        # Global grid size as seen by the simulation class.
        if self._block.dim == 2:
            self._global_size = (self.config.lat_ny, self.config.lat_nx)
        else:
            self._global_size = (self.config.lat_nz, self.config.lat_ny,
                    self.config.lat_nx)

        # Used so that face values map to the limiting coordinate
        # along a specific axis, e.g. lat_linear[_X_LOW] = 0
        # TODO(michalj): Should this use _block.envelope_size instead of -1?
        self.lat_linear = [0, self._lat_size[-1]-1, 0, self._lat_size[-2]-1]
        self.lat_linear_dist = [self._lat_size[-1]-2, 1,  self._lat_size[-2]-2, 1]

        if self._block.dim == 3:
            self.lat_linear.extend([0, self._lat_size[-3]-1])
            self.lat_linear_dist.extend([self._lat_size[-3]-2, 1])

    def _get_strides(self, type_):
        """Returns a list of strides for the NumPy array storing the lattice."""
        t = type_().nbytes
        return list(reversed(reduce(lambda x, y: x + [x[-1] * y],
                self._physical_size[-1:0:-1], [t])))

    def _get_nodes(self):
        """Returns the total amount of actual nodes in the lattice."""
        return reduce(operator.mul, self._physical_size)

    def _get_dist_bytes(self, grid):
        """Returns the number of bytes required to store a single set of
           distributions for the whole simulation domain."""
        return self._get_nodes() * grid.Q * self.float().nbytes

    def _get_compute_code(self):
        return self._bcg.get_code(self)

    def _get_global_idx(self, location, dist_num):
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
        if face in (self._block._X_LOW, self._block._X_HIGH):
            gx = self.lat_linear[face]
        else:
            return None
        return self._idx_helper(gx, cpair.src.src_slice, cpair.src.dists)

    def _get_dst_slice_indices(self, face, cpair):
        if not cpair.dst.dst_slice:
            return None
        if face in (self._block._X_LOW, self._block._X_HIGH):
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
        if cpair.dst.partial_nodes == 0:
            return None, None, None
        buf = np.zeros(cpair.dst.partial_nodes, dtype=self.float)
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
        # TODO(michalj): Fix this for multi-grid models.
        grid = util.get_grid_from_config(self.config)

        # Maps block ID to a list of 1 or 2 ConnectionBuffer
        # objects.  The list will contain 2 elements  only when
        # global periodic boundary conditions are enabled).
        self._block_to_connbuf = defaultdict(list)
        for face, block_id in self._block.connecting_blocks():
            cpair = self._block.get_connection(face, block_id)

            # Buffers for collecting and sending information.
            # TODO(michalj): Optimize this by providing proper padding.
            coll_buf = np.zeros(cpair.src.transfer_shape, dtype=self.float)
            coll_idx = self._get_src_slice_indices(face, cpair)

            # Buffers for receiving and distributing information.
            recv_buf = np.zeros(cpair.dst.transfer_shape, dtype=self.float)
            # Any partial dists are serialized into a single continuous buffer.
            dist_partial_buf, dist_partial_idx, dist_partial_sel = \
                    self._get_partial_dst_indices(face, cpair)
            dist_full_buf = np.zeros(cpair.dst.full_shape, dtype=self.float)
            dist_full_idx = self._get_dst_slice_indices(face, cpair)

            cbuf = ConnectionBuffer(face, cpair,
                    GPUBuffer(coll_buf, self.backend),
                    GPUBuffer(coll_idx, self.backend),
                    recv_buf,
                    GPUBuffer(dist_partial_buf, self.backend),
                    GPUBuffer(dist_partial_idx, self.backend),
                    dist_partial_sel,
                    GPUBuffer(dist_full_buf, self.backend),
                    GPUBuffer(dist_full_idx, self.backend))

            self.config.logger.debug('adding buffer for conn: {0} -> {1} '
                    '(face {2})'.format(self._block.id, block_id, face))
            self._block_to_connbuf[block_id].append(cbuf)

    def _init_compute(self):
        self.config.logger.debug("Initializing compute unit.")
        code = self._get_compute_code()
        self.module = self.backend.build(code)


        self._boundary_streams = (self.backend.make_stream(),
                                  self.backend.make_stream())
        self._bulk_stream = self.backend.make_stream()

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
            self._gpu_grids_primary.append(self.backend.alloc_buf(size=size))
            self._gpu_grids_secondary.append(self.backend.alloc_buf(size=size))

        self._gpu_geo_map = self.backend.alloc_buf(
                like=self._geo_block.encoded_map())

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

    def get_kernel(self, name, args, args_format, block_size=None):
        if block_size is None:
            block = self._kernel_block_size
        else:
            block = block_size
        return self.backend.get_kernel(self.module, name, args=args,
                args_format=args_format, block=block)

    def exec_kernel(self, name, args, args_format):
        kernel = self.get_kernel(name, args, args_format)
        self.backend.run_kernel(kernel, self._kernel_grid_size)

    def step(self, output_req):
        # call _step_boundary here
        self._step_bulk(output_req)
        self._sim.iteration += 1

    def _step_bulk(self, output_req):
        """Runs one simulation step in the bulk domain.

        Bulk domain is defined to be all nodes that belong to CUDA
        blocks that do not depend on input from any ghost nodes.
        """
        if output_req:
            kernel = self._kernels_full[self._sim.iteration & 1]
        else:
            kernel = self._kernels_none[self._sim.iteration & 1]

        # TODO(michalj): Do we need the sync here?
        self.backend.sync()

        self.backend.run_kernel(kernel, self._kernel_grid_size)

        if self._sim.iteration & 1:
            base = 0
        else:
            base = 3

        # TODO(michalj): Do we need the sync here?
        self.backend.sync()

        if self._block.periodic_x:
            kernel = self._pbc_kernels[base]
            if self._block.dim == 2:
                grid_size = (
                        int(math.ceil(self._lat_size[0] /
                            float(self.config.block_size))), 1)
            else:
                grid_size = (
                        int(math.ceil(self._lat_size[1] /
                            float(self.config.block_size))),
                        self._lat_size[0])

            self.backend.run_kernel(kernel, grid_size)

        if self._block.periodic_y:
            kernel = self._pbc_kernels[base + 1]
            if self._block.dim == 2:
                grid_size = (
                        int(math.ceil(self._lat_size[1] /
                            float(self.config.block_size))), 1)
            else:
                grid_size = (
                        int(math.ceil(self._lat_size[2] /
                            float(self.config.block_size))),
                        self._lat_size[0])
            self.backend.run_kernel(kernel, grid_size)

        if self._block.dim == 3 and self._block.periodic_z:
            kernel = self._pbc_kernels[base + 2]
            grid_size = (
                    int(math.ceil(self._lat_size[2] /
                        float(self.config.block_size))),
                    self._lat_size[1])
            self.backend.run_kernel(kernel, grid_size)

        for kernel, grid in self._collect_kernels[self._sim.iteration & 1]:
            self.backend.run_kernel(kernel, grid)

    def _step_boundary(self):
        """Runs one simulation step for the boundary blocks.

        Boundary blocks are CUDA blocks that depend on input from
        ghost nodes."""

        stream = self._boundary_stream[self._step]

    def send_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._block_to_connbuf[b_id]
            for x in conn_bufs:
                self.backend.from_buf(x.coll_buf.gpu)

            if len(conn_bufs) > 1:
                connector.send(np.hstack(
                    [np.ravel(x.coll_buf.host) for x in conn_bufs]))
            else:
                connector.send(np.ravel(conn_bufs[0].coll_buf.host))

    def recv_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            conn_bufs = self._block_to_connbuf[b_id]
            if len(conn_bufs) > 1:
                dest = np.hstack([np.ravel(x.recv_buf) for x in conn_bufs])
                if not connector.recv(dest, self._quit_event):
                    return
                i = 0
                # In case there are 2 connections between the blocks, reverse the
                # order of subbuffers in the recv buffer.  Note that this implicitly
                # assumes the order of conn_bufs is the same for both blocks.
                # TODO(michalj): Consider explicitly sorting conn_bufs.
                for cbuf in reversed(conn_bufs):
                    l = cbuf.recv_buf.size
                    cbuf.recv_buf[:] = dest[i:i+l].reshape(cbuf.recv_buf.shape)
                    i += l
                    cbuf.distribute(self.backend)
            else:
                cbuf = conn_bufs[0]
                dest = np.ravel(cbuf.recv_buf)
                if not connector.recv(dest, self._quit_event):
                    return
                # If ravel returned a copy, we need to write the data
                # back to the proper buffer.
                # TODO(michalj): Check if there is any way of avoiding this
                # copy.
                if dest.flags.owndata:
                    cbuf.recv_buf[:] = dest.reshape(cbuf.recv_buf.shape)
                cbuf.distribute(self.backend)

    def _fields_to_host(self):
        """Copies data for all fields from the GPU to the host."""
        for field in self._scalar_fields:
            self.backend.from_buf(self._gpu_field_map[id(field)])

        for field in self._vector_fields:
            for component in self._gpu_field_map[id(field)]:
                self.backend.from_buf(component)

    def _init_interblock_kernels(self):
        # TODO(michalj): Extend this for multi-grid models.

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
                # Data collection.
                if cbuf.coll_idx.host is not None:
                    grid_size = (_grid_dim1(cbuf.coll_buf.host.size),)

                    def _get_sparse_coll_kernel(i):
                        return KernelGrid(
                            self.get_kernel('CollectSparseData',
                            [cbuf.coll_idx.gpu, self.gpu_dist(0, i),
                             cbuf.coll_buf.gpu, cbuf.coll_buf.host.size],
                            'PPPi', (collect_block,)),
                            grid_size)

                    collect_primary.append(_get_sparse_coll_kernel(1))
                    collect_secondary.append(_get_sparse_coll_kernel(0))
                else:
                    # [X, Z * dists] or [X, Y * dists]
                    min_max = ([x.start for x in cbuf.cpair.src.src_slice] +
                            list(reversed(cbuf.coll_buf.host.shape[1:])))
                    min_max[-1] = min_max[-1] * len(cbuf.cpair.src.dists)
                    if self.dim == 2:
                        signature = 'PiiiP'
                        grid_size = (_grid_dim1(cbuf.coll_buf.host.size),)
                    else:
                        signature = 'PiiiiiP'
                        grid_size = (_grid_dim1(cbuf.coll_buf.host.shape[-1]),
                            cbuf.coll_buf.host.shape[-2] * len(cbuf.cpair.src.dists))

                    def _get_cont_coll_kernel(i):
                        return KernelGrid(
                            self.get_kernel('CollectContinuousData',
                            [self.gpu_dist(0, i),
                             cbuf.face] + min_max + [cbuf.coll_buf.gpu],
                             signature, (collect_block,)),
                             grid_size)

                    collect_primary.append(_get_cont_coll_kernel(1))
                    collect_secondary.append(_get_cont_coll_kernel(0))

                # Data distribution
                # Partial nodes.
                if cbuf.dist_partial_idx.host is not None:
                    grid_size = (_grid_dim1(cbuf.dist_partial_buf.host.size),)

                    def _get_sparse_dist_kernel(i):
                        return KernelGrid(
                                self.get_kernel('DistributeSparseData',
                                    [cbuf.dist_partial_idx.gpu,
                                     self.gpu_dist(0, i),
                                     cbuf.dist_partial_buf.gpu,
                                     cbuf.dist_partial_buf.host.size],
                                    'PPPi', (collect_block,)),
                                grid_size)

                    distrib_primary.append(_get_sparse_dist_kernel(0))
                    distrib_secondary.append(_get_sparse_dist_kernel(1))

                # Full nodes (all transfer distributions).
                if cbuf.dist_full_buf.host is not None:
                    # Sparse indexing (connection along X-axis).
                    if cbuf.dist_full_idx.host is not None:
                        grid_size = (_grid_dim1(cbuf.dist_full_buf.host.size),)

                        def _get_sparse_fdist_kernel(i):
                            return KernelGrid(
                                    self.get_kernel('DistributeSparseData',
                                        [cbuf.dist_full_idx.gpu,
                                         self.gpu_dist(0, i),
                                         cbuf.dist_full_buf.gpu,
                                         cbuf.dist_full_buf.host.size],
                                    'PPPi', (collect_block,)),
                                    grid_size)

                        distrib_primary.append(_get_sparse_fdist_kernel(0))
                        distrib_secondary.append(_get_sparse_fdist_kernel(1))
                    # Continuous indexing.
                    elif cbuf.cpair.dst.dst_slice:
                        # [X, Z * dists] or [X, Y * dists]
                        low = [x + self._block.envelope_size for x in cbuf.cpair.dst.dst_low]
                        min_max = ([(x + y.start) for x, y in zip(low, cbuf.cpair.dst.dst_slice)] +
                                list(reversed(cbuf.dist_full_buf.host.shape[1:])))
                        min_max[-1] = min_max[-1] * len(cbuf.cpair.dst.dists)

                        if self.dim == 2:
                            signature = 'PiiiP'
                            grid_size = (_grid_dim1(cbuf.dist_full_buf.host.size),)
                        else:
                            signature = 'PiiiiiP'
                            grid_size = (_grid_dim1(cbuf.dist_full_buf.host.shape[-1]),
                                cbuf.dist_full_buf.host.shape[-2] * len(cbuf.cpair.dst.dists))

                        def _get_cont_dist_kernel(i):
                            return KernelGrid(
                                    self.get_kernel('DistributeContinuousData',
                                    [self.gpu_dist(0, i),
                                     self._block.opposite_face(cbuf.face)] +
                                    min_max + [cbuf.dist_full_buf.gpu],
                                    signature, (collect_block,)),
                                    grid_size)

                        distrib_primary.append(_get_cont_dist_kernel(0))
                        distrib_secondary.append(_get_cont_dist_kernel(1))

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
        dbuf = np.zeros(self._get_dist_bytes(self._sim.grids[0]) / self.float().nbytes,
            dtype=self.float)
        self.backend.from_buf(self.gpu_dist(0, iter_idx), dbuf)
        dbuf = dbuf.reshape([self._sim.grids[0].Q] + self._physical_size)
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

    def run(self):
        self.config.logger.info("Initializing block.")

        self._init_geometry()
        self._init_buffers()
        self._init_compute()
        self.config.logger.debug("Initializing macroscopic fields.")
        self._sim.init_fields(self)
        self._geo_block.init_fields(self._sim)
        self._init_gpu_data()
        self.config.logger.debug("Applying initial conditions.")
        self._sim.initial_conditions(self)

        self._init_interblock_kernels()
        self._kernels_full = self._sim.get_compute_kernels(self, True)
        self._kernels_none = self._sim.get_compute_kernels(self, False)
        self._pbc_kernels = self._sim.get_pbc_kernels(self)

        if self.config.output:
            self._output.save(self._sim.iteration)

        self.config.logger.info("Starting simulation.")

        if not self.config.max_iters:
            self.config.logger.warning("Running infinite simulation.")

        if self.config.mode == 'benchmark':
            self.main_benchmark()
        else:
            self.main()

        self.config.logger.info(
            "Simulation completed after {0} iterations.".format(
                self._sim.iteration))

    def main(self):
        while True:
            output_req = ((self._sim.iteration + 1) % self.config.every) == 0

            if output_req and self.config.debug_dump_dists:
                dbuf = self._debug_get_dist(self)
                self._output.dump_dists(dbuf, self._sim.iteration)

            self.step(output_req)
            self.send_data()

            if output_req and self.config.output_required:
                self._fields_to_host()
                self._output.save(self._sim.iteration)

            if (self.config.max_iters > 0 and self._sim.iteration >=
                    self.config.max_iters):
                break

            if self._quit_event.is_set():
                self.config.logger.info("Simulation termination requested.")
                break

            self.recv_data()
            if self._quit_event.is_set():
                self.config.logger.info("Simulation termination requested.")

            for kernel, grid in self._distrib_kernels[self._sim.iteration & 1]:
                self.backend.run_kernel(kernel, grid)

    def main_benchmark(self):
        t_comp = 0.0
        t_total = 0.0
        t_send = 0.0
        t_recv = 0.0
        t_data = 0.0

        import time

        for i in xrange(self.config.max_iters):
            output_req = ((self._sim.iteration + 1) % self.config.every) == 0

            t1 = time.time()
            self.step(output_req)
            t2 = time.time()

            self.send_data()

            t3 = time.time()
            if output_req:
                self._fields_to_host()
            t4 = time.time()

            self.recv_data()

            for kernel, grid in self._distrib_kernels[self._sim.iteration & 1]:
                self.backend.run_kernel(kernel, grid)

            t5 = time.time()

            t_comp += t2 - t1
            t_total += t5 - t1
            t_recv += t5 - t4
            t_send += t3 - t2
            t_data += t4 - t3

            if output_req:
                mlups_base = self._sim.iteration * reduce(operator.mul,
                             self._lat_size)
                mlups_total = mlups_base / t_total * 1e-6
                mlups_comp = mlups_base / t_comp * 1e-6
                self.config.logger.info(
                        'MLUPS eff:{0:.2f}  comp:{1:.2f}  overhead:{2:.3f}'.format(
                            mlups_total, mlups_comp, t_total / t_comp - 1.0))

                j = self._sim.iteration
                self.config.logger.debug(
                        'time comp:{0:e}  data:{1:e}  recv:{2:e}  send:{3:e}'
                        '  total:{4:e}'.format(
                            t_comp / j, t_data / j, t_recv / j, t_send / j,
                            t_total / j))

        mi = self.config.max_iters
        ti = TimingInfo(t_comp / mi, t_data / mi, t_recv / mi, t_send / mi,
                t_total / mi, self._block.id)
        self._summary_sender.send_pyobj(ti)
        self.config.logger.debug('Sending timing information to controller.')
        assert self._summary_sender.recv_pyobj() == 'ack'

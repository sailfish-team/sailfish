import math
import operator
import numpy as np
from sailfish import codegen, sym, util

class BlockRunner(object):
    """Runs the simulation for a single LBBlock
    """
    def __init__(self, simulation, block, output, backend, quit_event):
        # Create a 2-way connection between the LBBlock and this BlockRunner
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

        # Indicates whether this block is connected to any other blocks.
        self._connected = True

    @property
    def config(self):
        return self._sim.config

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

        if self._block.dim == 3:
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

        # FIXME: Geometry parameters.
        ctx['num_params'] = 0
        ctx['geo_params'] = []

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

        if self._x_connected:
            ctx['distrib_collect_x_size'] = len(self._x_ghost_recv_buffer)
        else:
            ctx['distrib_collect_x_size'] = 0

    def make_scalar_field(self, dtype=None, name=None):
        """Allocates a scalar NumPy array.

        The array includes padding adjusted for the compute device (hidden from
        the end user), as well as space for any ghost nodes (not hidden)."""
        if dtype is None:
            dtype = self.float

        size = self._get_nodes()
        strides = self._get_strides(dtype)

        field = np.ndarray(self._physical_size, buffer=np.zeros(size, dtype=dtype),
                           dtype=dtype, strides=strides)

        if name is not None:
            f_view = field.view()[self._block._nonghost_slice]
            self._output.register_field(f_view, name)

        self._scalar_fields.append(field)
        return field

    def make_vector_field(self, name=None, output=False):
        """Allocates several scalar arrays representing a vector field."""
        components = []
        view_components = []

        for x in range(0, self._block.dim):
            field = self.make_scalar_field(self.float)
            components.append(field)
            view_components.append(field.view()[self._block._nonghost_slice])

        if name is not None:
            self._output.register_field(view_components, name)

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
        # X dimension is the last one on the list.
        self._lat_size = list(reversed(self._block.actual_size))

        # Physical in-memory size of the lattice, adjusted for optimal memory
        # access from the compute unit.  Size of the X dimension is rounded up
        # to a multiple of block_size.
        self._physical_size = list(reversed(self._block.actual_size))
        self._physical_size[-1] = (int(math.ceil(float(self._physical_size[-1]) /
                                                 self.config.block_size)) *
                                       self.config.block_size)

        # CUDA block/grid size for standard kernel call.
        self._kernel_grid_size = list(reversed(self._physical_size))
        self._kernel_grid_size[0] /= self.config.block_size

        self._kernel_block_size = [1] * len(self._lat_size)
        self._kernel_block_size[0] = self.config.block_size

        # Global grid size as seen by the simulation class.
        if self._block.dim == 2:
            self._global_size = (self.config.lat_ny, self.config.lat_nx)
        else:
            self._global_size = (self.config.lat_nz, self.config.lat_ny,
                    self.config.lat_nx)

        # Used so that axis_dir values map to the limiting coordinate
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

    def _init_buffers(self):
        # TODO(michalj): Fix this for multi-grid models.
        grid = util.get_grid_from_config(self.config)

        # Maps block_id to a list of (axis, span, size).
        block_axis_span = {}

        # Maps
        axis_span = {}
        def max_span(axis, span_tuple):
            if axis not in axis_span:
                axis_span[axis] = span_tuple
            else:
                curr = axis_span[axis]
                if self._block.dim == 2:
                    axis_span[axis] = (
                            min(curr[0], span_tuple[0]),
                            max(curr[1], span_tuple[1]))
                else:
                    axis_span[axis] = (
                            (min(curr[0][0], span_tuple[0][0]),
                             max(curr[0][1], span_tuple[0][1])),
                            (min(curr[1][0], span_tuple[1][0]),
                             max(curr[1][1], span_tuple[1][1])))

        def span_to_tuple(span):
            ret = []
            for coord in span:
                if type(coord) is slice:
                    ret.append((coord.start, coord.stop))
            return tuple(ret)

        def tuple_to_span(span):
            ret = []
            for coord in span:
                ret.append(slice(coord[0], coord[1]))
            return ret

        def relative_span(span1, span2):
            if self._block.dim == 2:
                return ((span2[0][0] - span1[0][0], span2[0][1] - span1[0][0]),)
            else:
                return ((span2[0][0] - span1[0][0], span2[0][1] - span1[0][0]),
                        (span2[1][0] - span1[1][0], span2[1][1] - span1[1][0]))


        for face, block_id in self._block.connecting_blocks():
            span = self._block.get_connection_span(face, block_id)
            size = self._block.connection_buf_size(grid, face, block_id)
            block_axis_span.setdefault(block_id, []).append((face, span, size))
            max_span(face, span_to_tuple(span))

        total_ortho_size = 0
        total_x_size = 0

        # Maps face to gx_start, num_nodes * num_dists, buf_offset
        #  gx_start: coordinates of the first node in the ghost patch on a face
        #  buf_offset: item offset in the global ghost buffer for all faces
        self._ghost_info = {}
        for face, span in axis_span.iteritems():
            buf_size = len(sym.get_prop_dists(grid,
                    self._block.axis_dir_to_dir(face),
                    self._block.axis_dir_to_axis(face)))
            for low, high in span:
                buf_size *= (high - low)

            if face < 2:
                self._ghost_info[face] = (low, buf_size, total_x_size)
                total_x_size += buf_size
            else:
                self._ghost_info[face] = (low, buf_size, total_ortho_size)
                total_ortho_size += buf_size

        self._x_connected = total_x_size > 0
        self._connected = (total_ortho_size > 0) or self._x_connected

        self._ortho_ghost_recv_buffer = None
        self._ortho_ghost_send_buffer = None

        if total_ortho_size > 0:
            self._ortho_ghost_recv_buffer = np.zeros(total_ortho_size,
                dtype=self.float)
            self._ortho_ghost_send_buffer = np.zeros(total_ortho_size,
                dtype=self.float)

        if total_x_size > 0:
            self._x_ghost_recv_buffer = np.zeros(total_x_size, dtype=self.float)
            self._x_ghost_send_buffer = np.zeros(total_x_size, dtype=self.float)
            # Lookup tables for distribution of the ghost node data.
            self._x_ghost_collect_idx = np.zeros(total_x_size, dtype=np.uint32)
            self._x_ghost_distrib_idx = np.zeros(total_x_size, dtype=np.uint32)

        def get_global_indices_array(face, span, gx_map):
            dists = sym.get_prop_dists(grid,
                    self._block.axis_dir_to_dir(face),
                    self._block.axis_dir_to_axis(face))
            gx = gx_map[face]
            gy = np.uint32(
                    range(self._block.envelope_size,
                          self._lat_size[-2] +
                          self._block.envelope_size)[tuple_to_span(span)[0]])
            gy = np.kron(gy, np.ones(len(dists), dtype=np.uint32))
            num_nodes = span[0][1] - span[0][0]
            dist_num = np.kron(
                    np.ones(num_nodes, dtype=np.uint32),
                    np.uint32(dists))
            return get_global_id(gx, gy, dist_num)

        def get_global_id(gx, gy, dist_num):
            arr_nx = self._physical_size[-1]
            return gx + arr_nx * gy + self._get_nodes() * dist_num

        self._blockface2view = {}

        face2view = {}
        idx = 0
        for face, (_, buf_size, offset) in self._ghost_info.iteritems():
            span = axis_span[face]

            if face < 2:
                recv_view = self._x_ghost_recv_buffer.view()
                send_view = self._x_ghost_send_buffer.view()
            else:
                recv_view = self._ortho_ghost_recv_buffer.view()
                send_view = self._ortho_ghost_send_buffer.view()

            recv_view = recv_view[offset:offset + buf_size]
            send_view = send_view[offset:offset + buf_size]

            nodes = 1
            for low, high in span:
                nodes *= (high - low)
            dists = buf_size / nodes

            recv_view = recv_view.reshape(dists, nodes)
            send_view = send_view.reshape(dists, nodes)

            # For 2D blocks, there is no need to reshape the view, which
            # is already a 1D array.
            # XXX: this is off by prop_dists
            if self._block.dim == 3:
                recv_view = recv_view.reshape(
                        (span[0][1]-span[0][0], span[1][1]-span[1][0]))
                send_view = send_view.reshape(
                        (span[0][1]-span[0][0], span[1][1]-span[1][0]))

            face2view[face] = (recv_view, send_view)

            # Add global indicies for connections along the X axis.
            if face < 2:
                self._x_ghost_collect_idx[idx:idx + size] = \
                        get_global_indices_array(face, span, self.lat_linear)
                self._x_ghost_distrib_idx[idx:idx + size] = \
                        get_global_indices_array(
                                self._block.opposite_axis_dir(face), span,
                                self.lat_linear_dist)
                idx += size

        for block_id, item_list in block_axis_span.iteritems():
            for face, span, size in item_list:
                view = face2view[face]
                global_span = axis_span[face]
                rel_span = relative_span(global_span, span_to_tuple(span))
                rel_span = tuple_to_span(rel_span)
                self._blockface2view.setdefault(block_id, []).append(
                        (face, view[0][:][rel_span], view[1][:][rel_span]))


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
            self._gpu_field_map[id(field)] = self.backend.alloc_buf(like=field)

        for field in self._vector_fields:
            gpu_vector = []
            for component in field:
                gpu_vector.append(self.backend.alloc_buf(like=component))
            self._gpu_field_map[id(field)] = gpu_vector

        if self._x_connected:
            self._gpu_x_ghost_recv_buffer = self.backend.alloc_buf(
                    like=self._x_ghost_recv_buffer)
            self._gpu_x_ghost_send_buffer = self.backend.alloc_buf(
                    like=self._x_ghost_send_buffer)

            self._gpu_x_ghost_collect_idx = self.backend.alloc_buf(
                    like=self._x_ghost_collect_idx)
            self._gpu_x_ghost_distrib_idx = self.backend.alloc_buf(
                    like=self._x_ghost_distrib_idx)

        if self._connected and self._ortho_ghost_recv_buffer is not None:
            self._gpu_ortho_ghost_recv_buffer = self.backend.alloc_buf(
                    like=self._ortho_ghost_recv_buffer)
            self._gpu_ortho_ghost_send_buffer = self.backend.alloc_buf(
                    like=self._ortho_ghost_send_buffer)

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
                        int(math.ceil(self._lat_size[0] /
                            float(self.config.block_size))))

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
                        int(math.ceil(self._lat_size[0] /
                            float(self.config.block_size))))
            self.backend.run_kernel(kernel, grid_size)

        if self._block.dim == 3 and self._block.periodic_z:
            kernel = self._pbc_kernels[base + 2]
            grid_size = (
                    int(math.ceil(self._lat_size[2] /
                        float(self.config.block_size))),
                    int(math.ceil(self._lat_size[1] /
                        float(self.config.block_size))))
            self.backend.run_kernel(kernel, grid_size)

        if self._connected:
            for kernel, grid in zip(
                    self._collect_kernels[self._sim.iteration & 1],
                    self._distrib_grid):
                self.backend.run_kernel(kernel, grid)

    def _step_boundary(self):
        """Runs one simulation step for the boundary blocks.

        Boundary blocks are CUDA blocks that depend on input from
        ghost nodes."""

        stream = self._boundary_stream[self._step]

    def send_data(self):
        if self._x_connected:
            self.backend.from_buf(self._gpu_x_ghost_send_buffer)

        if self._ortho_ghost_recv_buffer is not None:
            self.backend.from_buf(self._gpu_ortho_ghost_send_buffer)

        for b_id, connector in self._block._connectors.iteritems():
            faces = self._blockface2view[b_id]
            if len(faces) > 1:
                connector.send(np.hstack([np.ravel(x[2]) for x in faces]))
            else:
                connector.send(np.ravel(faces[0][2]))

    def recv_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            faces = self._blockface2view[b_id]
            if len(faces) > 1:
                dest = np.hstack([np.ravel(x[1]) for x in faces])
                if not connector.recv(dest, self._quit_event):
                    return
                idx = 0
                # If there are two connections between the blocks,
                # reverse the order of faces in the buffer.
                for views in reversed(faces):
                    dst_view = np.ravel(views[1])
                    dst_view[:] = dest[idx:idx + dst_view.shape[0]]
                    idx += dst_view.shape[0]
            else:
                if not connector.recv(np.ravel(faces[0][1]), self._quit_event):
                    return

        if self._x_connected:
            self.backend.to_buf(self._gpu_x_ghost_recv_buffer)
        if self._ortho_ghost_recv_buffer is not None:
            self.backend.to_buf(self._gpu_ortho_ghost_recv_buffer)

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

        # XXX: add a num_nodes parameter here as well.
        if self._x_connected:
            collect_primary.append(
                    self.get_kernel('CollectXGhostData',
                        [self._gpu_x_ghost_collect_idx,
                         self.gpu_dist(0, 1),
                         self._gpu_x_ghost_send_buffer],
                        'PPP', (collect_block,)))
            collect_secondary.append(
                    self.get_kernel('CollectXGhostData',
                        [self._gpu_x_ghost_collect_idx,
                         self.gpu_dist(0, 0),
                         self._gpu_x_ghost_send_buffer],
                        'PPP', (collect_block,)))
            distrib_primary.append(
                    self.get_kernel('DistributeXGhostData',
                        [self._gpu_x_ghost_distrib_idx,
                         self.gpu_dist(0, 0),
                         self._gpu_x_ghost_recv_buffer],
                        'PPP', (collect_block,)))
            distrib_secondary.append(
                    self.get_kernel('DistributeXGhostData',
                        [self._gpu_x_ghost_distrib_idx,
                         self.gpu_dist(0, 1),
                         self._gpu_x_ghost_recv_buffer],
                        'PPP', (collect_block,)))
            self._distrib_grid.append((int(math.ceil(
                    len(self._x_ghost_recv_buffer) /
                    float(collect_block))),))

        self._collect_kernels = (collect_primary, collect_secondary)
        self._distrib_kernels = (distrib_primary, distrib_secondary)

        # TODO(michalj): Can the buffer and offset be merged into a single
        # field?
        for face, (gx_start, num_nodes, buf_offset) in self._ghost_info.iteritems():
            # X-faces are already processed above.
            if face < 2:
                continue

            for i in range(0, 2):  # primary, secondary
                self._collect_kernels[i].append(
                        self.get_kernel('CollectOrthogonalGhostData',
                            [self.gpu_dist(0, 1-i),
                                np.int32(gx_start + self._block.envelope_size),
                                np.int32(face),
                                np.int32(num_nodes),
                                self._gpu_ortho_ghost_send_buffer, buf_offset],
                            'PiiiPi', (collect_block,)))
                self._distrib_kernels[i].append(
                        self.get_kernel('DistributeOrthogonalGhostData',
                            [self.gpu_dist(0, i),
                                np.int32(gx_start + self._block.envelope_size),
                                np.int32(self._block.opposite_axis_dir(face)),
                                np.int32(num_nodes),
                                self._gpu_ortho_ghost_recv_buffer, buf_offset],
                            'PiiiPi', (collect_block,)))
            self._distrib_grid.append((int(math.ceil(
                    num_nodes /
                    float(collect_block))),))

    def _debug_get_dist(self, output=True):
        """Copies the distributions from the GPU to a properly structured host array.
        :param output: if True, returns the contents of the distributions set *after*
                the current simulation step
        """
        iter_idx = self._sim.iteration & 1
        if not output:
            iter_idx = 1 - iter_idx

        dbuf = np.zeros(self._get_dist_bytes(self._sim.grids[0]) / self.float().nbytes,
             dtype=self.float)
        dbuf = dbuf.reshape([self._sim.grids[0].Q] + self._physical_size)
        self.backend.from_buf(self.gpu_dist(0, iter_idx), dbuf)
        return dbuf


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
            if self._connected:
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

            if self._connected:
                self.recv_data()
                if self._quit_event.is_set():
                    self.config.logger.info("Simulation termination requested.")

                for kernel, grid in zip(
                        self._distrib_kernels[self._sim.iteration & 1],
                        self._distrib_grid):
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

            if self._connected:
                self.send_data()

            t3 = time.time()
            if output_req:
                self._fields_to_host()
            t4 = time.time()

            if self._connected:
                self.recv_data()

                for kernel, grid in zip(
                        self._distrib_kernels[self._sim.iteration & 1],
                        self._distrib_grid):
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

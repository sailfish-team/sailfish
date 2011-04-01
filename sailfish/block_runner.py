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

        ctx['periodic_x'] = 0 #int(self._block.periodic_x)
        ctx['periodic_y'] = 0 #int(self._block.periodic_y)
        ctx['periodic_z'] = 0 #periodic_z
        ctx['periodicity'] = [0, 0, 0]

#[int(self._block.periodic_x), int(self._block.periodic_y), periodic_z]

        ctx['bnd_limits'] = bnd_limits
        ctx['dist_size'] = self._get_nodes()
        ctx['sim'] = self._sim

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

        if self._connected:
            ctx['distrib_collect_size'] = len(self._x_ghost_recv_buffer)
        else:
            ctx['distrib_collect_size'] = 0

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
            self._output.register_field(field, name)

        self._scalar_fields.append(field)
        return field

    def make_vector_field(self, name=None, output=False):
        """Allocates several scalar arrays representing a vector field."""
        components = []

        for x in range(0, self._block.dim):
            components.append(self.make_scalar_field(self.float))

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
        # TOOD(michalj): Fix this for multi-grid models.
        grid = util.get_grid_from_config(self.config)

        block_axis_span = {}

        total_size = 0

        for axis, block_id in self._block.connecting_blocks():
            span = self._block.get_connection_span(axis, block_id)
            size = self._block.connection_buf_size(grid, axis, block_id)
            block_axis_span.setdefault(block_id, []).append((axis, span, size))

            if axis <=1:
                total_size += size
            else:
                raise ValueError('Connections along non-X axis unsupported')


        # Nothing to do if there are no connecting blocks to share data with.
        if total_size == 0:
            self._connected = False
            return

        self._x_ghost_recv_buffer = np.zeros(total_size, dtype=self.float)
        self._x_ghost_send_buffer = np.zeros(total_size, dtype=self.float)

        # Lookup tables for distribution of the ghost node data.
        self._x_ghost_collect_idx = np.zeros(total_size, dtype=np.uint32)
        self._x_ghost_distrib_idx = np.zeros(total_size, dtype=np.uint32)

        def get_global_id(gx, gy, dist_num):
            arr_nx = self._physical_size[-1]
            return gx + arr_nx * gy + self._get_nodes() * dist_num

        idx = 0

        # TODO(michalj): Move this to block class?
        # TODO(michalj): Add a unit test for this.
        # TODO(muchalj): Extend this for 3D.
        for block_id, items in block_axis_span.iteritems():
            axis_idx = idx
            for axis, span, size in items:
                direction = util.span_to_direction(span)
                dists_collect = sym.get_prop_dists(
                        grid, direction,
                        self._block.axis_dir_to_axis(axis))

                gx = np.zeros(size, dtype=np.uint32)
                if direction == -1:
                    gx[:] = span[0]
                else:
                    gx[:] = span[0] + 2 * self._block.envelope_size

                gy = np.uint32(
                        range(self._block.envelope_size, self._lat_size[-2]
                            + self._block.envelope_size)[span[1]])
                gy = np.kron(gy, np.ones(len(dists_collect), dtype=np.uint32))

                num_nodes = span[1].stop - span[1].start
                dist_num = np.kron(
                        np.ones(num_nodes, dtype=np.uint32),
                        np.uint32(dists_collect))

                gi = get_global_id(gx, gy, dist_num)
                self._x_ghost_collect_idx[axis_idx:axis_idx + size] = gi
                axis_idx += size

            # Unless periodic conditions are applied, there is normally only a
            # single connection per block_id so there is no ordering of 'items'.
            # For the case of PC however, the order in the this loop has to be
            # reversed so that the different axes match in the subbuffer.  E.g.
            #  (PBC along the X axis)
            #  block 1 send buffer: 0 (low), 1 (high)
            #  block 2 recv buffer: 1 (high), 0 (low)
            #  (low) --- block 1 --- (high) | (low) --- block 2 --- (low)
            for axis, span, size in reversed(items):
                direction = util.span_to_direction(span)
                dists_distrib = sym.get_prop_dists(
                        grid, direction * -1,
                        self._block.axis_dir_to_axis(axis))

                gx = np.zeros(size, dtype=np.uint32)
                if direction == -1:
                    gx = span[0] + self._block.envelope_size
                else:
                    gx = span[0] + self._block.envelope_size

                gy = np.uint32(
                        range(self._block.envelope_size, self._lat_size[-2]
                            + self._block.envelope_size)[span[1]])
                gy = np.kron(gy, np.ones(len(dists_collect), dtype=np.uint32))

                num_nodes = span[1].stop - span[1].start
                dist_num = np.kron(
                        np.ones(num_nodes, dtype=np.uint32),
                        np.uint32(dists_distrib))

                gi = get_global_id(gx, gy, dist_num)
                self._x_ghost_distrib_idx[idx:idx + size] = gi
                idx += size

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

        if self._connected:
            self._gpu_x_ghost_recv_buffer = self.backend.alloc_buf(
                    like=self._x_ghost_recv_buffer)
            self._gpu_x_ghost_send_buffer = self.backend.alloc_buf(
                    like=self._x_ghost_send_buffer)

            self._gpu_x_ghost_collect_idx = self.backend.alloc_buf(
                    like=self._x_ghost_collect_idx)
            self._gpu_x_ghost_distrib_idx = self.backend.alloc_buf(
                    like=self._x_ghost_distrib_idx)

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

    def get_kernel(self, name, args, args_format):
        return self.backend.get_kernel(self.module, name, args=args,
                args_format=args_format, block=self._kernel_block_size)

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
            self.backend.run_kernel(
                    self._collect_kernels[self._sim.iteration & 1],
                    self._distrib_collect_grid_size)

    def _step_boundary(self):
        """Runs one simulation step for the boundary blocks.

        Boundary blocks are CUDA blocks that depend on input from
        ghost nodes."""

        stream = self._boundary_stream[self._step]

    def send_data(self):
        self.backend.from_buf(self._gpu_x_ghost_send_buffer)
        for b_id, connector in self._block._connectors.iteritems():
            connector.send(self._x_ghost_send_buffer)

    def recv_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            if not connector.recv(self._x_ghost_recv_buffer,
                    self._quit_event):
                return
        self.backend.to_buf(self._gpu_x_ghost_recv_buffer)


    def _fields_to_host(self):
        """Copies data for all fields from the GPU to the host."""
        for field in self._scalar_fields:
            self.backend.from_buf(self._gpu_field_map[id(field)])

        for field in self._vector_fields:
            for component in self._gpu_field_map[id(field)]:
                self.backend.from_buf(component)

    def _init_interblock_kernels(self):
        if not self._connected:
            return
        # TODO(michalj): Extend this for multi-grid models.
        self._collect_kernels = [
            self.get_kernel('CollectXGhostData',
                    [self._gpu_x_ghost_collect_idx,
                     self.gpu_dist(0, 1),
                     self._gpu_x_ghost_send_buffer],
                    'PPP'),
            self.get_kernel('CollectXGhostData',
                    [self._gpu_x_ghost_collect_idx,
                     self.gpu_dist(0, 0),
                     self._gpu_x_ghost_send_buffer],
                    'PPP')
            ]

        self._distrib_kernels = [
            self.get_kernel('DistributeXGhostData',
                    [self._gpu_x_ghost_distrib_idx,
                     self.gpu_dist(0, 0),
                     self._gpu_x_ghost_recv_buffer],
                    'PPP'),
            self.get_kernel('DistributeXGhostData',
                    [self._gpu_x_ghost_distrib_idx,
                     self.gpu_dist(0, 1),
                     self._gpu_x_ghost_recv_buffer],
                    'PPP')
            ]

        self._distrib_collect_grid_size = (
                int(math.ceil(
                    len(self._x_ghost_recv_buffer) /
                    float(self.config.block_size))),)

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

                self.backend.run_kernel(
                    self._distrib_kernels[self._sim.iteration & 1],
                    self._distrib_collect_grid_size)

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

                self.backend.run_kernel(
                    self._distrib_kernels[self._sim.iteration & 1],
                    self._distrib_collect_grid_size)

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

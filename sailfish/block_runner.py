import math
import operator
import numpy as np
from sailfish import codegen

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

        bnd_limits = list(self._block.size[:])

        if self._block.dim == 3:
            ctx['lat_nz'] = self._lat_size[-3]
            ctx['arr_nz'] = self._physical_size[-3]
            periodic_z = int(self._block.periodic_z)
        else:
            ctx['lat_nz'] = 1
            ctx['arr_nz'] = 1
            periodic_z = 0
            bnd_limits.append(1)

        ctx['periodic_x'] = int(self._block.periodic_x)
        ctx['periodic_y'] = int(self._block.periodic_y)
        ctx['periodic_z'] = periodic_z

        ctx['periodicity'] = [int(self._block.periodic_x),
                              int(self._block.periodic_y),
                              periodic_z]

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
        self._kernel_grid_size = list(self._block.actual_size)
        self._kernel_grid_size[0] /= self.config.block_size

        self._kernel_block_size = [1] * len(self._lat_size)
        self._kernel_block_size[0] = self.config.block_size

        # Global grid size.
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

    def _init_compute(self):
        self.config.logger.debug("Initializing compute unit.")
        code = self._get_compute_code()
        self.module = self.backend.build(code)

        self._boundary_streams = (self.backend.make_stream(),
                                  self.backend.make_stream())
        self._bulk_stream = self.backend.make_stream()


        # Allocate a transfer buffer suitable for asynchronous transfers.

    def _init_gpu_data(self):
        self.config.logger.debug("Initializing compute unit data.")

        for field in self._scalar_fields:
            self._gpu_field_map[id(field)] = self.backend.alloc_buf(like=field)

        for field in self._vector_fields:
            gpu_vector = []
            for component in field:
                gpu_vector.append(self.backend.alloc_buf(like=component))
            self._gpu_field_map[id(field)] = gpu_vector

        for grid in self._sim.grids:
            size = self._get_dist_bytes(grid)
            self._gpu_grids_primary.append(self.backend.alloc_buf(size=size))
            self._gpu_grids_secondary.append(self.backend.alloc_buf(size=size))

        self._gpu_geo_map = self.backend.alloc_buf(
                like=self._geo_block.encoded_map())

        print self._geo_block.encoded_map()

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
        self.backend.run_kernel(kernel, self._kernel_grid_size)

    def _step_boundary(self):
        """Runs one simulation step for the boundary blocks.

        Boundary blocks are CUDA blocks that depend on input from
        ghost nodes."""

        stream = self._boundary_stream[self._step]

    # XXX: Make these functions do something useful.
    def send_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            connector.send(None)

        print "block %d: send done" % self._block.id

    def recv_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            connector.recv(None)

        print "block %d: recv done" % self._block.id

    def _fields_to_host(self):
        """Copies data for all fields from the GPU to the host."""
        for field in self._scalar_fields:
            self.backend.from_buf(self._gpu_field_map[id(field)])

        for field in self._vector_fields:
            for component in self._gpu_field_map[id(field)]:
                self.backend.from_buf(component)

    def run(self):
        self.config.logger.info("Initializing block.")

        self._init_geometry()
        self._init_compute()
        self.config.logger.debug("Initializing macroscopic fields.")
        self._sim.init_fields(self)
        self._geo_block.init_fields(self._sim)
        self._init_gpu_data()
        self.config.logger.debug("Applying initial conditions.")
        self._sim.initial_conditions(self)

        self._kernels_full = self._sim.get_compute_kernels(self, True)
        self._kernels_none = self._sim.get_compute_kernels(self, False)

        if self.config.output:
            self._output.save(self._sim.iteration)

        self.config.logger.info("Starting simulation.")

        if not self.config.max_iters:
            self.config.logger.warning("Running infinite simulation.")

        while True:
            output_req = ((self._sim.iteration + 1) % self.config.every) == 0
            self.step(output_req)

            if output_req and self.config.output_required:
                self._fields_to_host()
                self._output.save(self._sim.iteration)

            # TODO: send data to other blocks

            if (self.config.max_iters > 0 and self._sim.iteration >=
                    self.config.max_iters):
                break

            if self._quit_event.is_set():
                self.config.logger.info("Simulation termination requested.")
                break

            # TODO: recv data from other blocks

        self.config.logger.info(
            "Simulation completed after {} iterations.".format(
                self._sim.iteration))


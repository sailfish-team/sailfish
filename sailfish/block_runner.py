import math
import operator
import numpy as np
from sailfish import codegen

class BlockRunner(object):
    """Runs the simulation for a single LBBlock
    """
    def __init__(self, simulation, block, output, backend):
        # Create a 2-way connection between the LBBlock and this BlockRunner
        self._block = block
        block.runner = self

        self._output = output
        self._backend = backend

        self._bcg = codegen.BlockCodeGenerator(simulation)
        self._sim = simulation

        if self._bcg.is_double_precision():
            self.float = np.float64
        else:
            self.float = np.float32

    @property
    def config(self):
        return self._sim.config

    def update_context(self, ctx):
        self._block.update_context(ctx)
        ctx.update(self._backend.get_defines())

        # Size of the lattice.
        ctx['lat_ny'] = self._lat_size[-2]
        ctx['lat_nx'] = self._lat_size[-1]

        # Actual size of the array, including any padding.
        ctx['arr_nx'] = self._physical_size[-1]
        ctx['arr_ny'] = self._physical_size[-2]

        ctx['periodic_x'] = int(self._block.periodic_x)
        ctx['periodic_y'] = int(self._block.periodic_y)

        bnd_limits = list(self._block.size[:])

        if self._block.dim == 3:
            ctx['lat_nz'] = self._lat_size[-3]
            ctx['arr_nz'] = self._physical_size[-3]
            ctx['periodic_z'] = int(self._block.periodic_z)
        else:
            ctx['lat_nz'] = 1
            ctx['arr_nz'] = 1
            ctx['periodic_z'] = 0
            bnd_limits.append(1)

        ctx['bnd_limits'] = bnd_limits
        ctx['dist_size'] = self._get_nodes()
#        ctx['periodicity'] = [int(self.options.periodic_x),
#                              int(self.options.periodic_y),
#                              int(self.options.periodic_z)]

#        ctx['pbc_offsets'] = [{-1: self.options.lat_nx,
#                                1: -self.options.lat_nx},
#                              {-1: self.options.lat_ny*self.arr_nx,
#                                1: -self.options.lat_ny*self.arr_nx},
#                              {-1: self.options.lat_nz*self.arr_ny*self.arr_nx,
#                                1: -self.options.lat_nz*self.arr_ny*self.arr_nx}]

    def make_scalar_field(self, dtype, name=None):
        size = self._get_nodes()
        strides = self._get_strides(dtype)

        field = np.ndarray(self._physical_size, buffer=np.zeros(size, dtype=dtype),
                           dtype=dtype, strides=strides)

        if name is not None:
            self._output.register_field(field, name)

        return field

    def _init_geometry(self):
        self._init_shape()
        self._geo_block = self._sim.geo(self._physical_size, self._block)
        self._geo_block.reset()

        # XXX: think how to deal with node encoding
        # XXX: later, allocate device buffers

    def _init_shape(self):
        # Logical size of the lattice.  X dimension is the last one on the
        # list.
        self._lat_size = list(reversed(self._block.actual_size))

        # Physical in-memory size of the lattice, adjusted for optimal memory
        # access from the compute unit.  Size of the X dimension is rounded up
        # to a multiple of block_size.
        self._physical_size = list(reversed(self._block.actual_size))
        self._physical_size[-1] = (int(math.ceil(float(self._physical_size[-1]) /
                                                 self.config.block_size)) *
                                       self.config.block_size)

    def _get_strides(self, type_):
        """Returns a list of strides for the NumPy array storing the lattice."""
        t = type_().nbytes
        return list(reversed(reduce(lambda x, y: x + [x[-1] * y],
                self._physical_size[-1:0:-1], [t])))

    def _get_nodes(self):
        """Returns the total amount of actual nodes in the lattice."""
        return reduce(operator.mul, self._physical_size)

    def _get_compute_code(self):
        return self._bcg.get_code(self)

    def _init_compute(self):
        code = self._get_compute_code()
        self.module = self.backend.build(code)

        self._boundary_streams = (self.backend.make_stream(),
                                  self.backend.make_stream())
        self._bulk_stream = self.backend.make_stream()

        # Allocate a transfer buffer suitable for asynchronous transfers.


    def _step_bulk(self):
        """Runs one simulation step in the bulk domain.

        Bulk domain is defined to be all nodes that belong to CUDA
        blocks that do not depend on input from any ghost nodes.
        """
        kernel = None
        self.backend.run_kernel(kernel, self._bulk_grid_size)

    def _step_boundary(self):
        """Runs one simulation step for the boundary blocks.

        Boundary blocks are CUDA blocks that depend on input from
        ghost nodes."""

        stream = self._boundary_stream[self._step]

        self.backend.to_buf(buf, stream=stream)
        self.backend.make_sync_event(stream)
        self.backend.run_kernel(populate_data_kernel, None,
                                stream=stream)
        self.backend.make_sync_event(stream)
        self.backend.run_kernel(time_step, None, stream=stream)
        self.backend.make_sync_event(stream)
        self.backend.run_kernel(copy_kernel, None, stream=stream)
        self.backend.make_sync_event(stream)
        self.backend.from_buf(buf, stream=stream)

    """
    Boundary streams: (odd and even)
    - copy data from other blocks to transfer buffer in GPU memory
    - sync
    - write data to correct places in the grid
    - sync
    - time step for boundary blocks
    - sync
    - copy data to transfer buffer
    - sync
    - read data from transfer buffer to the host
    """

    # XXX: Make these functions do something useful.
    def send_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            connector.send(None)

        print "block %d: send done" % self._block.id

    def recv_data(self):
        for b_id, connector in self._block._connectors.iteritems():
            connector.recv(None)

        print "block %d: recv done" % self._block.id

    def run(self):
        self._init_geometry()
        self._init_compute()

        return

if __name__ == "__main__":
    import doctest
    doctest.testmod()

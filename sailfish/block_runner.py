from sailfish import codegen, config


class BlockRunner(object):
    """Runs the simulation for single LBBlock
    """
    def __init__(self, simulation, block, output, backend):
        # Create a 2-way connection between the LBBlock and this BlockRunner
        self._block = block
        block.runner = self

        self._output = output
        self._backend = backend

        self._bcg = codegen.BlockCodeGenerator(simulation)
        self._sim = simulation

    def _init_geometry(self):
        pass

    def _get_compute_code(self):
        return self._bcg.get_code()

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

    def run(self):

        import time
        time.sleep(10)
        return
"""
        self._init_geometry()
        self._init_code()
        self._init_compute()
"""

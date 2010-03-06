import pycuda.autoinit
import pycuda.compiler
import pycuda.driver as cuda

from struct import calcsize, pack

def _expand_block(block):
    if block is int:
        return (block, 1, 1)
    elif len(block) == 1:
        return (block[0], 1, 1)
    elif len(block) == 2:
        return (block[0], block[1], 1)
    else:
        return block

def _expand_grid(grid):
    if len(grid) == 1:
        return (grid[0], 1)
    else:
        return grid

class CUDABackend(object):

    @classmethod
    def add_options(cls, group):
        return 0

    def __init__(self, options):
        self.buffers = {}

    def alloc_buf(self, size=None, like=None):
        if like is not None:
            buf = cuda.mem_alloc(like.size * like.dtype.itemsize)
            self.buffers[buf] = like
            self.to_buf(buf)
        else:
            buf = cuda.mem_alloc(size)

        return buf

    def to_buf(self, cl_buf, source=None):
        if source is None:
            if cl_buf in self.buffers:
                cuda.memcpy_htod(cl_buf, self.buffers[cl_buf])
            else:
                raise ValueError('Unknown compute buffer and source not specified.')
        else:
            cuda.memcpy_htod(cl_buf, source)

    def from_buf(self, cl_buf, target=None):
        if target is None:
            if cl_buf in self.buffers:
                cuda.memcpy_dtoh(self.buffers[cl_buf], cl_buf)
            else:
                raise ValueError('Unknown compute buffer and target not specified.')
        else:
            cuda.memcpy_dtoh(target, cl_buf)

    def build(self, source):
        return pycuda.compiler.SourceModule(source)#, options=['--use_fast_math'])

    def get_kernel(self, prog, name, block, args, args_format, shared=None):
        kern = prog.get_function(name)
        kern.param_set_size(calcsize(args_format))
        setattr(kern, 'args', (args, args_format))
        kern.set_block_shape(*_expand_block(block))
        if shared is not None:
            kern.set_shared_size(shared)
        return kern

    def run_kernel(self, kernel, grid_size):
        kernel.param_setv(0, pack(kernel.args[1], *kernel.args[0]))
        kernel.launch_grid(*_expand_grid(grid_size))

    def sync(self):
        cuda.Context.synchronize()

    def get_defines(self):
        return {
            'shared_var': '__shared__',
            'kernel': '__global__',
            'global_ptr': '',
            'const_ptr': '',
            'device_func': '__device__',
            'const_var': '__constant__',
        }


backend=CUDABackend

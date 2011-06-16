"""Sailfish OpenCL backend."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import os
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.reduction as reduction
import pyopencl.tools

class OpenCLBackend(object):

    @classmethod
    def add_options(cls, group):
        group.add_option('--opencl-interactive-select', dest='opencl_interactive',
                help='select the OpenCL device in an interactive manner', action='store_true', default=False)
        return 1

    def __init__(self, options):
        if options.opencl_interactive:
            self.ctx = cl.create_some_context(True)
        else:
            if 'OPENCL_PLATFORM' in os.environ:
                platform_num = int(os.environ['OPENCL_PLATFORM'])
            else:
                platform_num = 0

            platform = cl.get_platforms()[platform_num]
            devices = platform.get_devices(device_type=cl.device_type.GPU)

            if 'OPENCL_DEVICE' in os.environ:
                device = int(os.environ['OPENCL_DEVICE'])
                devices = [devices[device]]
            self.ctx = cl.Context(devices=devices, properties=[(cl.context_properties.PLATFORM, platform)])

        self.queue = cl.CommandQueue(self.ctx)
        self.buffers = {}
        self.arrays = {}

    def alloc_buf(self, size=None, like=None, wrap_in_array=True):
        mf = cl.mem_flags
        if like is not None:
            if like.base is not None:
                hbuf = like.base
            else:
                hbuf = like

            buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=hbuf)
            self.buffers[buf] = hbuf
            self.to_buf(buf)
            if wrap_in_array:
                self.arrays[buf] = clarray.Array(self.ctx, like.shape, like.dtype, data=buf)
        else:
            buf = cl.Buffer(self.ctx, mf.READ_WRITE, size)

        return buf

    def to_buf(self, cl_buf, source=None):
        if source is None:
            if cl_buf in self.buffers:
                cl.enqueue_write_buffer(self.queue, cl_buf, self.buffers[cl_buf])
            else:
                raise ValueError('Unknown compute buffer and source not specified.')
        else:
            if source.base is not None:
                cl.enqueue_write_buffer(self.queue, cl_buf, source.base)
            else:
                cl.enqueue_write_buffer(self.queue, cl_buf, source)

    def from_buf(self, cl_buf, target=None):
        if target is None:
            if cl_buf in self.buffers:
                cl.enqueue_read_buffer(self.queue, cl_buf, self.buffers[cl_buf])
            else:
                raise ValueError('Unknown compute buffer and target not specified.')
        else:
            if target.base is not None:
                cl.enqueue_read_buffer(self.queue, cl_buf, target.base)
            else:
                cl.enqueue_read_buffer(self.queue, cl_buf, target)

    def build(self, source):
        preamble = '#pragma OPENCL EXTENSION cl_khr_fp64: enable\n'
        return cl.Program(self.ctx, preamble + source).build() #'-cl-single-precision-constant -cl-fast-relaxed-math')

    def get_kernel(self, prog, name, block, args, args_format, shared=0, fields=[]):
        kern = getattr(prog, name)
        for i, arg in enumerate(args):
            kern.set_arg(i, arg)
        setattr(kern, 'block', block)
        return kern

    def run_kernel(self, kernel, grid, *args):
        global_size = []
        for i, dim in enumerate(grid):
            global_size.append(dim * kernel.block[i])

        cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, kernel.block[0:len(global_size)])

    def get_reduction_kernel(self, reduce_expr, map_expr, neutral, *args):
        """Generate and return reduction kernel; see PyOpenCL documentation
        of pyopencl.reduction.ReductionKernel for detailed description.
        Function expects buffers that are in device address space,
        stored in gpu_* variables.

        :param reduce_expr: expression used to reduce two values into one,
            must use a and b as values names, e.g. 'a+b'
        :param map_expr: expression used to map value from input array,
            arrays are named x0, x1, etc., e.g. 'x0[i]*x1[i]
        :param neutral: neutral value in reduce_expr, e.g. '0'
        :param args: buffers on which to calculate reduction, e.g. backend.gpu_rho
        """
        arrays = []
        arguments = []
        for i, arg in enumerate(args):
            array = self.arrays[arg]
            arrays.append(array)
            arguments.append('const {0} *x{1}'.format(pyopencl.tools.dtype_to_ctype(array.dtype), i))
        kernel = reduction.ReductionKernel(arrays[0].dtype, neutral=neutral,
                reduce_expr=reduce_expr, map_expr=map_expr,
                arguments=', '.join(arguments))
        return lambda : kernel(*arrays).get()

    def sync(self):
        self.queue.finish()

    def get_defines(self):
        return {
            'shared_var': '__local',
            'kernel': '__kernel',
            'global_ptr': '__global',
            'const_ptr': '__constant',
            'device_func': '',
            'const_var': '__constant',
        }

backend=OpenCLBackend

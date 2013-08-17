"""Sailfish OpenCL backend."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import operator
import os
import pyopencl as cl
import pyopencl.array as clarray
import pyopencl.reduction as reduction
import pyopencl.tools
import numpy as np

class OpenCLBackend(object):
    name ='opencl'
    array = clarray
    FatalError = pyopencl.RuntimeError

    @classmethod
    def add_options(cls, group):
        group.add_argument('--opencl-interactive-select',
                dest='opencl_interactive',
                help='select the OpenCL device in an interactive manner',
                action='store_true', default=False)
        return 1

    def __init__(self, options, gpu_id):
        """Initializes the OpenCL backend.

        :param gpu_id: number of the GPU to use
        """
        if options.opencl_interactive:
            self.ctx = cl.create_some_context(True)
        else:
            if 'OPENCL_PLATFORM' in os.environ:
                platform_num = int(os.environ['OPENCL_PLATFORM'])
            else:
                platform_num = 0

            platform = cl.get_platforms()[platform_num]
            devices = platform.get_devices(device_type=cl.device_type.GPU)
            devices = [devices[gpu_id]]
            self.ctx = cl.Context(devices=devices, properties=[(cl.context_properties.PLATFORM, platform)])

        self.default_queue = cl.CommandQueue(self.ctx)
        self.buffers = {}
        self.arrays = {}
        self._iteration_kernels = []

    @property
    def info(self):
        return ''

    @property
    def supports_printf(self):
        return False

    def set_iteration(self, it):
        self._iteration = it
        for kernel in self._iteration_kernels:
            kernel.set_arg(kernel.numargs - 1, np.uint32(it))

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

    def alloc_async_host_buf(self, shape, dtype):
        """Allocates a buffer that can be used for asynchronous data
        transfers."""
        mf = cl.mem_flags
        # OpenCL does not offer direct control over how memory is allocated,
        # but ALLOC_HOST_PTR is supposed to prefer pinned memory.
        if type(shape) is tuple or type(shape) is list:
            size = reduce(operator.mul, shape)
        else:
            size = shape
        host_buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.ALLOC_HOST_PTR,
                size=size * dtype().nbytes)
        return host_buf.get_host_array(shape, dtype)

    def to_buf(self, cl_buf, source=None):
        if source is None:
            if cl_buf in self.buffers:
                cl.enqueue_write_buffer(self.default_queue, cl_buf,
                        self.buffers[cl_buf]).wait()
            else:
                raise ValueError('Unknown compute buffer and source not specified.')
        else:
            if source.base is not None:
                cl.enqueue_write_buffer(self.default_queue, cl_buf,
                        source.base).wait()
            else:
                cl.enqueue_write_buffer(self.default_queue, cl_buf,
                        source).wait()

    def from_buf(self, cl_buf, target=None):
        if target is None:
            if cl_buf in self.buffers:
                cl.enqueue_read_buffer(self.default_queue, cl_buf,
                        self.buffers[cl_buf]).wait()
            else:
                raise ValueError('Unknown compute buffer and target not specified.')
        else:
            if target.base is not None:
                cl.enqueue_read_buffer(self.default_queue, cl_buf,
                        target.base).wait()
            else:
                cl.enqueue_read_buffer(self.default_queue, cl_buf,
                        target).wait()

    def to_buf_async(self, cl_buf, stream=None):
        queue = stream.queue if stream is not None else self.default_queue
        cl.enqueue_write_buffer(queue, cl_buf, self.buffers[cl_buf],
                is_blocking=False)

    def from_buf_async(self, cl_buf, stream=None):
        queue = stream.queue if stream is not None else self.default_queue
        cl.enqueue_read_buffer(queue, cl_buf, self.buffers[cl_buf],
                is_blocking=False)

    def build(self, source):
        preamble = '#pragma OPENCL EXTENSION cl_khr_fp64: enable\n'
        return cl.Program(self.ctx, preamble + source).build() #'-cl-single-precision-constant -cl-fast-relaxed-math')

    def get_kernel(self, prog, name, block, args, args_format, shared=0,
            needs_iteration=False, more_shared=False):
        """
        :param needs_iteration: if True, the kernel needs access to the current iteration
            number, which will be provided to it as the last argument
        """
        kern = getattr(prog, name)
        if needs_iteration:
            args.append(np.uint32(0))
            self._iteration_kernels.append(kern)

        for i, arg in enumerate(args):
            kern.set_arg(i, arg)
        setattr(kern, 'block', block)
        setattr(kern, 'numargs', len(args))
        return kern

    def run_kernel(self, kernel, grid_size, stream=None):
        global_size = []
        for i, dim in enumerate(grid_size):
            global_size.append(dim * kernel.block[i])

        cl.enqueue_nd_range_kernel(self.default_queue, kernel, global_size, kernel.block[0:len(global_size)])

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
        self.default_queue.finish()

    def make_stream(self):
        return StreamWrapper(cl.CommandQueue(self.ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE))

    def make_event(self, stream, timing=False):
        return EventWrapper(cl.enqueue_marker(stream.queue))

    def get_defines(self):
        return {
            # FIXME
            'warp_size': 32,
            'supports_shuffle': False,
            'shared_var': '__local',
            'kernel': '__kernel',
            'global_ptr': '__global',
            'const_ptr': '',
            'device_func': '',
            'const_var': '__constant',
        }

    def sync_stream(self, *streams):
        for s in streams:
            s.synchronize()


class EventWrapper(object):
    def __init__(self, event):
        self.event = event

    def time_since(self, other):
        return 0
        #return self.event.profile.end - other.event.profile.start


class StreamWrapper(object):
    def __init__(self, cmd_queue):
        self.queue = cmd_queue

    def wait_for_event(self, event):
        cl.enqueue_wait_for_events(self.queue, [event.event])

    def synchronize(self):
        self.queue.finish()

backend=OpenCLBackend

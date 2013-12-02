"""Sailfish CUDA backend."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import operator
import time

import pycuda.compiler
import pycuda.tools
import pycuda.driver as cuda
import pycuda.gpuarray as cudaarray
import pycuda.reduction as reduction


def _expand_block(block):
    if type(block) is int:
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
        return tuple(grid)


class CUDABackend(object):
    name = 'cuda'
    array = cudaarray
    FatalError = pycuda.driver.LaunchError

    @classmethod
    def devices_count(cls):
        """Returns the number of CUDA devices on this host."""
        return pycuda.driver.Device.count

    @classmethod
    def add_options(cls, group):
        group.add_argument('--cuda-kernel-stats', dest='cuda_kernel_stats',
                help='print information about amount of memory and registers '
                     'used by the kernels', action='store_true',
                     default=False)
        group.add_argument('--cuda_nvcc', dest='cuda_nvcc',
                help='location of the NVCC compiler',
                type=str, default='nvcc')
        group.add_argument('--cuda-nvcc-opts', dest='cuda_nvcc_opts',
                help='additional parameters to pass to the CUDA compiler',
                type=str, default='')
        group.add_argument('--cuda-keep-temp', dest='cuda_keep_temp',
                help='keep intermediate CUDA files', action='store_true',
                default=False)
        group.add_argument('--cuda-fermi-highprec', dest='cuda_fermi_highprec',
                help='use high precision division and sqrt on Compute Capability 2.0+ '
                     ' devices', action='store_true', default=False)
        group.add_argument('--cuda-disable-l1', dest='cuda_disable_l1',
                           help='Disable L1 cache for global memory '
                           'reads/stores', action='store_true', default=False)
        group.add_argument('--nocuda_cache', dest='cuda_cache',
                           action='store_false', default=True,
                           help='Disable the use of the pycuda compiler cache.')
        group.add_argument('--cuda-sched-yield', dest='cuda_sched_yield',
                           action='store_true', default=False,
                           help='Yield to other threads when waiting for CUDA '
                           + 'calls to complete; improves performance of other '
                           + 'CPU threads under high load.')
        group.add_argument('--cuda-minimize-cpu-usage', dest='cuda_minimize_cpu',
                           action='store_true', default=False,
                           help='Minimize CPU usage when waiting for results ' +
                           'from the GPU. Might slightly degrade performance.')
        return 1

    def __init__(self, options, gpu_id):
        """Initializes the CUDA backend.

        :param options: LBConfig object
        :param gpu_id: number of the GPU to use
        """
        cuda.init()
        self.buffers = {}
        self.arrays = {}
        self._kern_stats = set()
        self.options = options
        self._device = cuda.Device(gpu_id)
        self._ctx = self._device.make_context(
            flags=cuda.ctx_flags.SCHED_AUTO if not options.cuda_sched_yield else
            cuda.ctx_flags.SCHED_YIELD)

        if (options.precision == 'double' and
            self._device.compute_capability()[0] >= 3):
            if hasattr(self._ctx, 'set_shared_config'):
                self._ctx.set_shared_config(cuda.shared_config.EIGHT_BYTE_BANK_SIZE)

        # To keep track of allocated memory.
        self._total_memory_bytes = 0

        self._iteration_kernels = []

    def __del__(self):
        self._ctx.pop()

    @property
    def supports_printf(self):
        return self._device.compute_capability()[0] >= 2

    @property
    def info(self):
        return '{0} / CC {1} / MEM {2}'.format(
                self._device.name(), self._device.compute_capability(),
                self.total_memory)

    @property
    def total_memory(self):
        return self._device.total_memory()

    def ipc_handle(self, addr):
        return cuda.mem_get_ipc_handle(addr)

    def ipc_handle_wrap(self, handle):
        return cuda.IPCMemoryHandle(handle)

    def set_iteration(self, it):
        for kernel in self._iteration_kernels:
            kernel.args[-1] = it

    def alloc_buf(self, size=None, like=None, wrap_in_array=False):
        """Allocates a buffer on the device."""
        if like is not None:
            # When calculating the total array size, take into account
            # any striding.
            # XXX: why does it even work?
            buf_size = like.shape[0] * like.strides[0]
            buf = cuda.mem_alloc(buf_size)
            self._total_memory_bytes += buf_size

            if like.base is not None and type(like.base) is not cuda.PagelockedHostAllocation:
                self.buffers[buf] = like.base
            else:
                self.buffers[buf] = like

            self.to_buf(buf)
            if wrap_in_array:
                self.arrays[buf] = cudaarray.GPUArray(like.shape, like.dtype, gpudata=buf)
        else:
            self._total_memory_bytes += size
            buf = cuda.mem_alloc(size)

        return buf

    def alloc_async_host_buf(self, shape, dtype):
        """Allocates a buffer that can be used for asynchronous data
        transfers."""
        return cuda.pagelocked_zeros(shape, dtype=dtype)

    def to_buf(self, cl_buf, source=None):
        """Copies data from the host to a device buffer."""
        if source is None:
            if cl_buf in self.buffers:
                cuda.memcpy_htod(cl_buf, self.buffers[cl_buf])
            else:
                raise ValueError('Unknown compute buffer and source not specified.')
        else:
            if source.base is not None:
                cuda.memcpy_htod(cl_buf, source.base)
            else:
                cuda.memcpy_htod(cl_buf, source)

    def from_buf(self, cl_buf, target=None):
        """Copies data from a device buffer to the host."""
        if target is None:
            if cl_buf in self.buffers:
                cuda.memcpy_dtoh(self.buffers[cl_buf], cl_buf)
            else:
                raise ValueError('Unknown compute buffer and target not specified.')
        else:
            if target.base is not None:
                cuda.memcpy_dtoh(target.base, cl_buf)
            else:
                cuda.memcpy_dtoh(target, cl_buf),

    def to_buf_async(self, cl_buf, stream=None):
        cuda.memcpy_htod_async(cl_buf, self.buffers[cl_buf], stream)

    def from_buf_async(self, cl_buf, stream=None):
        cuda.memcpy_dtoh_async(self.buffers[cl_buf], cl_buf, stream)

    def build(self, source):
        if self.options.cuda_nvcc_opts:
            import shlex
            options = shlex.split(self.options.cuda_nvcc_opts)
        else:
            options = []

        if not self.options.cuda_fermi_highprec and self._device.compute_capability()[0] >= 2:
            options.append('--prec-div=false')
            options.append('--prec-sqrt=false')

        if self.options.cuda_disable_l1:
            options.extend(['-Xptxas', '-dlcm=cg'])

        #if cuda.get_driver_version() >= 5000:
        #    # Generate annotated PTX code.
        #    options.append('-src-in-ptx')

        if self.options.cuda_cache:
            cache = None
        else:
            cache = False

        return pycuda.compiler.SourceModule(source, options=options,
                nvcc=self.options.cuda_nvcc, keep=self.options.cuda_keep_temp,
                cache_dir=cache)

    def get_kernel(self, prog, name, block, args, args_format, shared=0,
            needs_iteration=False, more_shared=False):
        """
        :param name: kernel name
        :param block: CUDA block size
        :param args: iterable of arguments to pass to the kernel
        :param args_format: string indicating the type of kernel arguments; see
            pycuda documentation for more info
        :param shared: number of bytes of shared memory to allocate for the
            kernel
        :param needs_iteration: if True, the kernel needs access to the current iteration
            number, which will be provided to it as the last argument
        """
        kern = prog.get_function(name)

        if more_shared:
            kern.set_cache_config(cuda.func_cache.PREFER_SHARED)
        else:
            # Use a larger L1 cache by default.
            kern.set_cache_config(cuda.func_cache.PREFER_L1)

        if needs_iteration:
            args_format += 'i'
            args = list(args)
            args.append(0)
            self._iteration_kernels.append(kern)

        kern.prepare(args_format)
        setattr(kern, 'args', args)
        setattr(kern, 'block_shape', _expand_block(block))
        setattr(kern, 'shared_size', shared)
        setattr(kern, 'needs_iteration', needs_iteration)

        if self.options.cuda_kernel_stats and name not in self._kern_stats:
            self._kern_stats.add(name)
            ddata = pycuda.tools.DeviceData()
            occ = pycuda.tools.OccupancyRecord(ddata, reduce(operator.mul, block), kern.shared_size_bytes, kern.num_regs)

            print '%s: l:%d  s:%d  r:%d  occ:(%f tb:%d w:%d l:%s)' % (name, kern.local_size_bytes, kern.shared_size_bytes,
                    kern.num_regs, occ.occupancy, occ.tb_per_mp, occ.warps_per_mp, occ.limited_by)

        return kern

    def run_kernel(self, kernel, grid_size, stream=None):
        kernel.prepared_async_call(_expand_grid(grid_size), kernel.block_shape,
                stream, *kernel.args, shared_size=kernel.shared_size)

    def get_reduction_kernel(self, reduce_expr, map_expr, neutral, *args):
        """Generate and return a reduction kernel; see PyCUDA documentation
        of pycuda.reduction.ReductionKernel for a detailed description.
        The eunction expects buffers that are in the device address space,
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
            arguments.append('const {0} *x{1}'.format(pycuda.tools.dtype_to_ctype(array.dtype), i))
        kernel = reduction.ReductionKernel(arrays[0].dtype, neutral=neutral,
                reduce_expr=reduce_expr, map_expr=map_expr,
                arguments=', '.join(arguments))
        return lambda : kernel(*arrays).get()

    def get_array(self, arg):
        return self.arrays[arg]

    def sync(self):
        self._ctx.synchronize()

    def make_stream(self):
        return cuda.Stream()

    def make_event(self, stream, timing=False):
        flags = 0
        if self.options.cuda_minimize_cpu:
            flags |= cuda.event_flags.BLOCKING_SYNC
        if not timing:
            flags |= cuda.event_flags.DISABLE_TIMING
        event = cuda.Event(flags)
        event.record(stream)
        return event

    def get_defines(self):
        return {
            'warp_size': self._device.get_attribute(cuda.device_attribute.WARP_SIZE),
            'supports_shuffle': self._device.compute_capability()[0] >= 3,
            'backend': 'cuda',
            'shared_var': '__shared__',
            'kernel': '__global__',
            'global_ptr': '',
            'const_ptr': 'const',
            'device_func': '__device__',
            'const_var': '__constant__',
        }

    def sync_stream(self, *streams):
        if self.options.cuda_minimize_cpu:
            for s in streams:
                self.make_event(s).synchronize()
        else:
            for s in streams:
                s.synchronize()

backend=CUDABackend

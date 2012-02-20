"""Sailfish CUDA backend."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import operator
from struct import calcsize, pack

import pycuda.compiler
import pycuda.tools
import pycuda.driver as cuda
import pycuda.gpuarray as cudaarray
import pycuda.reduction as reduction


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

def _set_txt_format(dsc, strides):
    # float
    if strides[-1] == 4:
        dsc.format = cuda.array_format.FLOAT
        dsc.num_channels = 1
    # double encoded as int2
    else:
        dsc.format = cuda.array_format.UNSIGNED_INT32
        dsc.num_channels = 2

class CUDABackend(object):
    name='cuda'

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
                help='use high precision division on Compute Capability 2.0+ '
                     ' devices', action='store_true', default=False)
        group.add_argument('--cuda_cache', type=bool, default=True,
                help='if True, use the pycuda compiler cache.')
        group.add_argument('--block_size', type=int, default=64,
                help='size of the block of threads on the compute device')
        return 1

    def __init__(self, options, gpu_id):
        """Initializes the CUDA backend.

        :param gpu_id: number of the GPU to use
        """
        cuda.init()
        self.buffers = {}
        self.arrays = {}
        self._kern_stats = set()
        self._tex_to_memcpy = {}
        self.options = options
        self._device = cuda.Device(gpu_id)
        self._ctx = self._device.make_context()

        # To keep track of allocated memory.
        self._total_memory_bytes = 0

    @property
    def total_memory(self):
        return self._device.total_memory()

    def alloc_buf(self, size=None, like=None, wrap_in_array=False):
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

    def nonlocal_field(self, prog, cl_buf, num, shape, strides):
        if len(shape) == 3:
            dsc = cuda.ArrayDescriptor()
            dsc.width = strides[0] / strides[2]
            dsc.height = shape[-3]
            _set_txt_format(dsc, strides)

            txt = prog.get_texref('img_f%d' % num)
            txt.set_address_2d(cl_buf, dsc, strides[-3])

            # It turns out that using 3D textures doesn't really make
            # much sense if it requires copying data around.  We therefore
            # access the 3D fields via a 2D texture, which still provides
            # some caching, while not requiring a separate copy of the
            # data.
            #
            # dsc = cuda.ArrayDescriptor3D()
            # dsc.depth, dsc.height, dsc.width = shape
            # dsc.format = cuda.array_format.FLOAT
            # dsc.num_channels = 1
            # ary = cuda.Array(dsc)

            # copy = cuda.Memcpy3D()
            # copy.set_src_device(cl_buf)
            # copy.set_dst_array(ary)
            # copy.width_in_bytes = copy.src_pitch = strides[-2]
            # copy.src_height = copy.height = dsc.height
            # copy.depth = dsc.depth

            # txt = prog.get_texref('img_f%d' % num)
            # txt.set_array(ary)
            # self._tex_to_memcpy[txt] = copy
        else:
            # 2D texture.
            dsc = cuda.ArrayDescriptor()
            dsc.width = shape[-1]
            dsc.height = shape[-2]
            _set_txt_format(dsc, strides)
            txt = prog.get_texref('img_f%d' % num)
            txt.set_address_2d(cl_buf, dsc, strides[-2])
        return txt

    def to_buf(self, cl_buf, source=None):
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

        if self.options.cuda_cache:
            cache = None
        else:
            cache = False

        return pycuda.compiler.SourceModule(source, options=options,
                nvcc=self.options.cuda_nvcc, keep=self.options.cuda_keep_temp,
                cache_dir=cache) #options=['-Xopencc', '-O0']) #, options=['--use_fast_math'])

    def get_kernel(self, prog, name, block, args, args_format, shared=None, fields=[]):
        kern = prog.get_function(name)
        kern.prepare(args_format, shared, texrefs=[x for x in fields if x is not None])
        setattr(kern, 'args', args)
        setattr(kern, 'block_shape', _expand_block(block))

        if self.options.cuda_kernel_stats and name not in self._kern_stats:
            self._kern_stats.add(name)
            ddata = pycuda.tools.DeviceData()
            occ = pycuda.tools.OccupancyRecord(ddata, reduce(operator.mul, block), kern.shared_size_bytes, kern.num_regs)

            print '%s: l:%d  s:%d  r:%d  occ:(%f tb:%d w:%d l:%s)' % (name, kern.local_size_bytes, kern.shared_size_bytes,
                    kern.num_regs, occ.occupancy, occ.tb_per_mp, occ.warps_per_mp, occ.limited_by)

        return kern

    def run_kernel(self, kernel, grid_size, stream=None):
        kernel.prepared_async_call(_expand_grid(grid_size), kernel.block_shape,
                stream, *kernel.args)

    def get_reduction_kernel(self, reduce_expr, map_expr, neutral, *args):
        """Generate and return reduction kernel; see PyCUDA documentation
        of pycuda.reduction.ReductionKernel for detailed description.
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
            arguments.append('const {0} *x{1}'.format(pycuda.tools.dtype_to_ctype(array.dtype), i))
        kernel = reduction.ReductionKernel(arrays[0].dtype, neutral=neutral,
                reduce_expr=reduce_expr, map_expr=map_expr,
                arguments=', '.join(arguments))
        return lambda : kernel(*arrays).get()

    def sync(self):
        self._ctx.synchronize()

    def make_stream(self):
        return cuda.Stream()

    def make_event(self, stream, timing=False):
        flags = 0
        if not timing:
            flags |= cuda.event_flags.DISABLE_TIMING
        event = cuda.Event(flags)
        event.record(stream)
        return event

    def get_defines(self):
        return {
            'backend': 'cuda',
            'shared_var': '__shared__',
            'kernel': '__global__',
            'global_ptr': '',
            'const_ptr': '',
            'device_func': '__device__',
            'const_var': '__constant__',
        }


backend=CUDABackend

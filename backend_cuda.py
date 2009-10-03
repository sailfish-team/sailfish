import pycuda.autoinit
import pycuda.driver as cuda

class CUDABackend(object):

	def __init__(self):
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
		return cuda.SourceModule(source, options=['--use_fast_math', '-Xptxas', '-v'])

	def get_kernel(self, prog, name, args, block, shared=None):
		kern = prog.get_function(name)
		kern.prepare(args, block=block, shared=shared)
		return kern

	def run_kernel(self, kernel, grid, *args):
		kernel.prepared_call(grid, *args)

	def sync(self):
		cuda.Context.synchronize()

	def get_defines(self):
		return {
			'shared_var': '__shared__',
			'kernel': '__global__',
			'global_ptr': '',
			'const_ptr': '',
			'device_func': '__device__'
		}


backend=CUDABackend

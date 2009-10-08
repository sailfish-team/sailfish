import pyopencl as cl

class CUDABackend(object):

	def __init__(self):
		self.ctx = cl.Context(dev_type=cl.device_type.ALL)
		self.queue = cl.CommandQueue(self.ctx)
		self.buffers = {}

	def alloc_buf(self, size=None, like=None):
		mf = cl.mem_flags
		if like is not None:
			buf = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=like)
			self.buffers[buf] = like
			self.to_buf(buf)
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
			cl.enqueue_write_buffer(self.queue, cl_buf, source)

	def from_buf(self, cl_buf, target=None):
		if target is None:
			if cl_buf in self.buffers:
				cl.enqueue_read_buffer(self.queue, cl_buf, self.buffers[cl_buf])
			else:
				raise ValueError('Unknown compute buffer and target not specified.')
		else:
			cl.enqueue_read_buffer(self.queue, cl_buf, target)

	def build(self, source):
		return cl.Program(self.ctx, source).build('-cl-single-precision-constant -cl-fast-relaxed-math')

	def get_kernel(self, prog, name, block, args, args_format, shared=0):
		kern = getattr(prog, name)
		for i, arg in enumerate(args):
			kern.set_arg(i, arg)
		setattr(kern, 'block', block)
		return kern

	def run_kernel(self, kernel, grid, *args):
		global_size = []
		for i, dim in enumerate(grid):
			global_size.append(dim * kernel.block[i])

		cl.enqueue_nd_range_kernel(self.queue, kernel, global_size, kernel.block)

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

backend=CUDABackend

import math
import numpy
import sys
import time
import geo
import vis2d

import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

from mako.template import Template
from mako.lookup import TemplateLookup

import sym

SUPPORTED_BACKENDS = {'cuda': 'backend_cuda', 'opencl': 'backend_opencl'}

for backend in SUPPORTED_BACKENDS.values():
	try:
		__import__(backend)
	except ImportError:
		pass

def get_backends():
	"""Get a list of available backends."""
	return sorted([k for k, v in SUPPORTED_BACKENDS.iteritems() if v in sys.modules])

class Values(optparse.Values):
	def __init__(self, *args):
		optparse.Values.__init__(self, *args)
		self.specified = set()

	def __setattr__(self, name, value):
		self.__dict__[name] = value
		if hasattr(self, 'specified'):
			self.specified.add(name)

def _convert_to_double(src):
	import re
	return re.sub('([0-9]+\.[0-9]*)f', '\\1', src.replace('float', 'double'))

class LBMSim(object):

	# The filename base for screenshots.
	filename = 'lbm_sim'

	def __init__(self, geo_class, misc_options=[], args=sys.argv[1:], defaults=None):
		parser = OptionParser()

		parser.add_option('-q', '--quiet', dest='quiet', help='reduce verbosity', action='store_true', default=False)

		group = OptionGroup(parser, 'LB engine settings')
		group.add_option('--lat_w', dest='lat_w', help='lattice width', type='int', action='store', default=128)
		group.add_option('--lat_h', dest='lat_h', help='lattice height', type='int', action='store', default=128)
		group.add_option('--lat_d', dest='lat_d', help='lattice depth', type='int', action='store', default=1)
		group.add_option('--visc', dest='visc', help='viscosity', type='float', action='store', default=0.01)
		group.add_option('--model', dest='model', help='LBE model to use', type='choice', choices=['bgk', 'mrt'], action='store', default='bgk')
		group.add_option('--accel_x', dest='accel_x', help='y component of the external acceleration', action='store', type='float', default=0.0)
		group.add_option('--accel_y', dest='accel_y', help='x component of the external acceleration', action='store', type='float', default=0.0)
		group.add_option('--accel_z', dest='accel_z', help='z component of the external acceleration', action='store', type='float', default=0.0)
		group.add_option('--periodic_x', dest='periodic_x', help='lattice periodic in the X direction', action='store_true', default=False)
		group.add_option('--periodic_y', dest='periodic_y', help='lattice periodic in the Y direction', action='store_true', default=False)
		group.add_option('--periodic_z', dest='periodic_z', help='lattice periodic in the Z direction', action='store_true', default=False)
		group.add_option('--precision', dest='precision', help='precision (single, double)', type='choice', choices=['single', 'double'], default='single')
		group.add_option('--boundary', dest='boundary', help='boundary condition implementation', type='choice', choices=[x.name for x in geo.SUPPORTED_BCS], default='fullbb')
		parser.add_option_group(group)

		group = OptionGroup(parser, 'Run mode settings')
		group.add_option('--backend', dest='backend', help='backend', type='choice', choices=get_backends(), default=get_backends()[0])
		group.add_option('--benchmark', dest='benchmark', help='benchmark mode, implies no visualization', action='store_true', default=False)
		group.add_option('--max_iters', dest='max_iters', help='number of iterations to run in benchmark/batch mode', action='store', type='int', default=0)
		group.add_option('--batch', dest='batch', help='run in batch mode, with no visualization', action='store_true', default=False)
		group.add_option('--nobatch', dest='batch', help='run in interactive mode', action='store_false')
		group.add_option('--save_src', dest='save_src', help='file to save the CUDA/OpenCL source code to', action='store', type='string', default='')
		group.add_option('--use_src', dest='use_src', help='CUDA/OpenCL source to use instead of the automatically generated one', action='store', type='string', default='')
		group.add_option('--output', dest='output', help='save simulation results to FILE', metavar='FILE', action='store', type='string', default='')
		group.add_option('--output_format', dest='output_format', help='output format', type='choice', choices=['h5nested', 'h5flat', 'vtk'], default='h5flat')
		parser.add_option_group(group)

		group = OptionGroup(parser, 'Visualization options')
		group.add_option('--scr_w', dest='scr_w', help='screen width', type='int', action='store', default=0)
		group.add_option('--scr_h', dest='scr_h', help='screen height', type='int', action='store', default=0)
		group.add_option('--scr_scale', dest='scr_scale', help='screen scale', type='float', action='store', default=3.0)
		group.add_option('--every', dest='every', help='update the visualization every N steps', metavar='N', type='int', action='store', default=100)
		group.add_option('--tracers', dest='tracers', help='number of tracer particles', type='int', action='store', default=32)
		group.add_option('--vismode', dest='vismode', help='visualization mode', type='choice', choices=vis2d.vis_map.keys(), action='store', default='std')
		parser.add_option_group(group)

		group = OptionGroup(parser, 'Simulation-specific options')
		for option in misc_options:
			group.add_option(option)

		parser.add_option_group(group)

		self.geo_class = geo_class
		self.options = Values(parser.defaults)
		parser.parse_args(args, self.options)

		# Set default command line values for unspecified options.  This is different
		# than the default values provided above, as these cannot be changed by
		# subclasses.
		if defaults is not None:
			for k, v in defaults.iteritems():
				if k not in self.options.specified:
					setattr(self.options, k, v)

		# Adjust workgroup size if necessary to ensure that we will be able to
		# successfully execute the main LBM kernel.
		if self.options.lat_w < 64:
			self.block_size = self.options.lat_w
		else:
			# TODO: This should be dynamically adjusted based on both the lat_w
			# value and device capabilities.
			self.block_size = 64

		self._mlups_calls = 0
		self._mlups = 0.0
		self.clear_hooks()
		self.backend = sys.modules[SUPPORTED_BACKENDS[self.options.backend]].backend()

		if not self.options.quiet:
			print 'Using the "%s" backend.' % self.options.backend

		if not self._is_double_precision():
			self.float = numpy.float32
		else:
			self.float = numpy.float64

	def _is_double_precision(self):
		return self.options.precision == 'double'

	def _calc_screen_size(self):
		# If the size of the window has not been explicitly defined, automatically adjust it
		# based on the size of the grid,
		if self.options.scr_w == 0:
			self.options.scr_w = self.options.lat_w * self.options.scr_scale

		if self.options.scr_h == 0:
			self.options.scr_h = self.options.lat_h * self.options.scr_scale

	def _init_vis(self):
		if not self.options.benchmark and not self.options.batch:
			if sym.GRID.dim == 2:
				self._init_vis_2d()
			elif sym.GRID.dim == 3:
				self._init_vis_3d()

	def _init_vis_2d(self):
		self.vis = vis2d.Fluid2DVis(self.options.scr_w, self.options.scr_h,
									self.options.lat_w, self.options.lat_h)

	def _init_vis_3d(self):
		import vis3d
		self.vis = vis3d.Fluid3DVis()

	def add_iter_hook(self, i, func, every=False):
		"""Add a hook that will be executed during the simulation.

		Args:
		  i: number of the time step after which the hook is to be run
		  func: a callable representing the hook
		  every: if True, the hook will be executed every i steps
		"""
		if every:
			self._iter_hooks_every.setdefault(i, []).append(func)
		else:
			self._iter_hooks.setdefault(i, []).append(func)

	def clear_hooks(self):
		"""Remove all hooks."""
		self._iter_hooks = {}
		self._iter_hooks_every = {}

	def get_tau(self):
		return self.float((6.0 * self.options.visc + 1.0)/2.0)

	def get_dist_size(self):
		return self.options.lat_w * self.options.lat_h * self.options.lat_d

	def _init_geo(self):
		# Particle distributions in host memory.
		if sym.GRID.dim == 2:
			self.shape = (self.options.lat_h, self.options.lat_w)
		else:
			self.shape = (self.options.lat_d, self.options.lat_h, self.options.lat_w)

		self.dist = numpy.zeros([len(sym.GRID.basis)] + list(self.shape), self.float)

		# Simulation geometry.
		self.geo = self.geo_class(list(reversed(self.shape)), self.options.model, self.options, self.float, self.backend)
		self.geo.init_dist(self.dist)
		self.geo_params = self.float(self.geo.get_params())
		# HACK: Prevent this method from being called again.
		self._init_geo = lambda: True

	def _init_code(self):
		lbm_tmpl = Template(filename='lbm.mako', lookup=TemplateLookup(directories=['.']))

		self.tau = self.get_tau()
		ctx = {}
		ctx['dim'] = sym.GRID.dim
		ctx['block_size'] = self.block_size
		ctx['lat_h'] = self.options.lat_h
		ctx['lat_w'] = self.options.lat_w
		ctx['lat_d'] = self.options.lat_d
		ctx['num_params'] = len(self.geo_params)
		ctx['model'] = self.options.model
		ctx['periodic_x'] = int(self.options.periodic_x)
		ctx['periodic_y'] = int(self.options.periodic_y)
		ctx['periodic_z'] = int(self.options.periodic_z)
		ctx['dist_size'] = self.get_dist_size()
		ctx['ext_accel_x'] = self.options.accel_x
		ctx['ext_accel_y'] = self.options.accel_y
		ctx['ext_accel_z'] = self.options.accel_z
		ctx['tau'] = self.tau
		ctx['visc'] = self.float(self.options.visc)
		ctx['backend'] = self.options.backend
		ctx['geo_params'] = self.geo_params
		ctx['boundary_type'] = self.options.boundary
		ctx['pbc_offsets'] = [{-1: self.options.lat_w,
								1: -self.options.lat_w},
							  {-1: self.options.lat_h*self.options.lat_w,
								1: -self.options.lat_h*self.options.lat_w},
							  {-1: self.options.lat_d*self.options.lat_h*self.options.lat_w,
								1: -self.options.lat_d*self.options.lat_h*self.options.lat_w}]
		ctx['bnd_limits'] = [self.options.lat_w, self.options.lat_h, self.options.lat_d]
		ctx['loc_names'] = ['gx', 'gy', 'gz']
		ctx['periodicity'] = [int(self.options.periodic_x), int(self.options.periodic_y),
							int(self.options.periodic_z)]

		ctx.update(self.geo.get_defines())
		ctx.update(self.backend.get_defines())

		src = lbm_tmpl.render(**ctx)

		if self._is_double_precision():
			src = _convert_to_double(src)

		if self.options.save_src:
			with open(self.options.save_src, 'w') as fsrc:
				print >>fsrc, src

		# If external source code was requested, ignore the code that we have
		# just generated above.
		if self.options.use_src:
			with open(self.options.use_src, 'r') as fsrc:
				src = fsrc.read()

		self.mod = self.backend.build(src)

	def _init_lbm(self):
		# Velocity.
		self.vx = numpy.zeros(self.shape, self.float)
		self.vy = numpy.zeros(self.shape, self.float)
		self.velocity = [self.vx, self.vy]
		self.gpu_vx = self.backend.alloc_buf(like=self.vx)
		self.gpu_vy = self.backend.alloc_buf(like=self.vy)
		self.gpu_velocity = [self.gpu_vx, self.gpu_vy]

		if sym.GRID.dim == 3:
			self.vz = numpy.zeros(self.shape, self.float)
			self.gpu_vz = self.backend.alloc_buf(like=self.vz)
			self.velocity.append(self.vz)
			self.gpu_velocity.append(self.gpu_vz)

		# Density.
		self.rho = numpy.zeros(self.shape, self.float)
		self.gpu_rho = self.backend.alloc_buf(like=self.rho)

		# Tracer particles.
		self.tracer_x = numpy.random.random_sample(self.options.tracers).astype(self.float) * self.options.lat_w
		self.tracer_y = numpy.random.random_sample(self.options.tracers).astype(self.float) * self.options.lat_h
		self.tracer_loc = [self.tracer_x, self.tracer_y]
		self.gpu_tracer_x = self.backend.alloc_buf(like=self.tracer_x)
		self.gpu_tracer_y = self.backend.alloc_buf(like=self.tracer_y)
		self.gpu_tracer_loc = [self.gpu_tracer_x, self.gpu_tracer_y]

		if sym.GRID.dim == 3:
			self.tracer_z = numpy.random.random_sample(self.options.tracers).astype(self.float) * self.options.lat_d
			self.gpu_tracer_z = self.backend.alloc_buf(like=self.tracer_z)
			self.tracer_loc.append(self.tracer_z)
			self.gpu_tracer_loc.append(self.gpu_tracer_z)

		# Particle distributions in device memory, A-B access pattern.
		self.gpu_dist1 = self.backend.alloc_buf(like=self.dist)
		self.gpu_dist2 = self.backend.alloc_buf(like=self.dist)

		# Kernel arguments.
		args_tracer2 = [self.gpu_dist1, self.geo.gpu_map] + self.gpu_tracer_loc
		args_tracer1 = [self.gpu_dist2, self.geo.gpu_map] + self.gpu_tracer_loc
		args1 = [self.geo.gpu_map, self.gpu_dist1, self.gpu_dist2, self.gpu_rho] + self.gpu_velocity + [numpy.uint32(0)]
		args2 = [self.geo.gpu_map, self.gpu_dist2, self.gpu_dist1, self.gpu_rho] + self.gpu_velocity + [numpy.uint32(0)]

		# Special argument list for the case where macroscopic quantities data is to be
		# saved in global memory, i.e. a visualization step.
		args1v = [self.geo.gpu_map, self.gpu_dist1, self.gpu_dist2, self.gpu_rho] + self.gpu_velocity + [numpy.uint32(1)]
		args2v = [self.geo.gpu_map, self.gpu_dist2, self.gpu_dist1, self.gpu_rho] + self.gpu_velocity + [numpy.uint32(1)]

		if sym.GRID.dim == 2:
			k_block_size = (self.block_size, 1)
		else:
			k_block_size = (self.block_size, 1, 1)

		kern_cnp1 = self.backend.get_kernel(self.mod,
					'LBMCollideAndPropagate',
					args=args1,
					args_format='P'*(len(args1)-1)+'i',
					block=k_block_size)
		kern_cnp2 = self.backend.get_kernel(self.mod,
					'LBMCollideAndPropagate',
					args=args2,
					args_format='P'*(len(args2)-1)+'i',
					block=k_block_size)
		kern_cnp1s = self.backend.get_kernel(self.mod,
					'LBMCollideAndPropagate',
					args=args1v,
					args_format='P'*(len(args1v)-1)+'i',
					block=k_block_size)
		kern_cnp2s = self.backend.get_kernel(self.mod,
					'LBMCollideAndPropagate',
					args=args2v,
					args_format='P'*(len(args2v)-1)+'i',
					block=k_block_size)
		kern_trac1 = self.backend.get_kernel(self.mod,
					'LBMUpdateTracerParticles',
					args=args_tracer1,
					args_format='P'*len(args_tracer1),
					block=(1,))
		kern_trac2 = self.backend.get_kernel(self.mod,
					'LBMUpdateTracerParticles',
					args=args_tracer2,
					args_format='P'*len(args_tracer2),
					block=(1,))

		# Map: iteration parity -> kernel arguments to use.
		self.kern_map = {
			0: (kern_cnp1, kern_cnp1s, kern_trac1),
			1: (kern_cnp2, kern_cnp2s, kern_trac2),
		}

		if sym.GRID.dim == 2:
			self.kern_grid_size = (self.options.lat_w/self.block_size, self.options.lat_h)
		else:
			self.kern_grid_size = (self.options.lat_w/self.block_size * self.options.lat_h, self.options.lat_d)

	def sim_step(self, i, tracers=True, get_data=False):
		"""Perform a single step of the simulation.

		Args:
		  i: current timestep
		  tracers: if True, the position of tracer particles will be updated
		  get_data: if True, macroscopic variables will be copied from the compute unit
		    and made available as properties of this class
		"""
		kerns = self.kern_map[i & 1]

		if (not self.options.benchmark and i % self.options.every == 0) or get_data:
			self.backend.run_kernel(kerns[1], self.kern_grid_size)
			if tracers:
				self.backend.run_kernel(kerns[2], (self.options.tracers,))
				for loc in self.gpu_tracer_loc:
					self.backend.from_buf(loc)

			for vel in self.gpu_velocity:
				self.backend.from_buf(vel)
			self.backend.from_buf(self.gpu_rho)

			if self.options.output:
				self._output_data(i)
		else:
			self.backend.run_kernel(kerns[0], self.kern_grid_size)
			if tracers:
				self.backend.run_kernel(kerns[2], (self.options.tracers,))

	def get_macro_quantities(self, gpu_dist):
		"""Compute global macroscopic quantities of the fluid directly from the distributions.

		Args:
		  gpu_dist: a compute unit memory object containing the distributions

		Returns:
		  A tuple of momentum, density.  Momentum is a 2- or 3-vector, depending on the
		  dimensionality of the grid used in the simulation.
		"""
		self.backend.from_buf(gpu_dist)

		mx = 0.0
		my = 0.0
		mz = 0.0

		for i, mval in enumerate(self.dist):
			mx += sym.GRID.basis[i][0] * numpy.sum(mval)
			my += sym.GRID.basis[i][1] * numpy.sum(mval)
			if sym.GRID.dim == 3:
				mz += sym.GRID>basis[i][2] * numpy.sum(mval)

		if sym.GRID.dim == 3:
			return (mx, my, mz), numpy.sum(self.dist)
		else:
			return (mx, my), numpy.sum(self.dist)

	def get_mlups(self, tdiff, iters=None):
		if iters is not None:
			it = iters
		else:
			it = self.options.every

		mlups = float(it) * self.options.lat_w * self.options.lat_h * self.options.lat_d * 1e-6 / tdiff
		self._mlups = (mlups + self._mlups * self._mlups_calls) / (self._mlups_calls + 1)
		self._mlups_calls += 1
		return (self._mlups, mlups)

	def _output_data(self, i):
		if self.options.output_format == 'h5flat':
			h5t = self.h5file.createGroup(self.h5grp, 'iter%d' % i, 'iteration %d' % i)
			self.h5file.createArray(h5t, 'v', numpy.dstack(self.velocity), 'velocity')
			self.h5file.createArray(h5t, 'rho', self.rho, 'density')
		elif self.options.output_format == 'vtk':
			from enthought.tvtk.api import tvtk
			id = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))
			id.point_data.scalars = self.rho.flatten()
			id.point_data.scalars.name = 'density'
			id.point_data.vectors = numpy.c_[self.vx.flatten(), self.vy.flatten(), self.vz.flatten()]
			id.point_data.vectors.name = 'velocity'
			id.dimensions = self.rho.shape
			w = tvtk.XMLPImageDataWriter(input=id, file_name='%s%05d.xml' % (self.options.output, i))
			w.write()
		else:
			record = self.h5tbl.row
			record['iter'] = i
			record['vx'] = self.vx
			record['vy'] = self.vy
			if sym.GRID.dim == 3:
				record['vz'] = self.vz
			record['rho'] = self.rho
			record.append()
			self.h5tbl.flush()

	def _init_output(self):
		if self.options.output and self.options.output_format != 'vtk':
			import tables
			self.h5file = tables.openFile(self.options.output, mode='w')
			self.h5grp = self.h5file.createGroup('/', 'results', 'simulation results')
			self.h5file.setNodeAttr(self.h5grp, 'viscosity', self.options.visc)
			self.h5file.setNodeAttr(self.h5grp, 'accel_x', self.options.accel_x)
			self.h5file.setNodeAttr(self.h5grp, 'accel_y', self.options.accel_y)
			self.h5file.setNodeAttr(self.h5grp, 'accel_z', self.options.accel_z)
			self.h5file.setNodeAttr(self.h5grp, 'sample_rate', self.options.every)
			self.h5file.setNodeAttr(self.h5grp, 'model', self.options.model)

			if self.options.output_format == 'h5nested':
				desc = {
					'iter': tables.Float32Col(pos=0),
					'vx': tables.Float32Col(pos=1, shape=self.vx.shape),
					'vy': tables.Float32Col(pos=2, shape=self.vy.shape),
					'rho': tables.Float32Col(pos=4, shape=self.rho.shape)
				}

				if sym.GRID.dim == 3:
					desc['vz'] = tables.Float32Col(pos=2, shape=self.vz.shape)

				self.h5tbl = self.h5file.createTable(self.h5grp, 'results', desc, 'results')

	def _run_benchmark(self):
		self.iter = 0

		if self.options.max_iters:
			cycles = self.options.max_iters
		else:
			cycles = 1000

		print '# iters mlups_avg mlups_curr'

		import time

		while True:
			t_prev = time.time()

			for iter in range(0, cycles):
				self.sim_step(self.iter, tracers=False)
				self.iter += 1

			self.backend.sync()
			t_now = time.time()
			print self.iter,
			print '%.2f %.2f' % self.get_mlups(t_now - t_prev, cycles)

			if self.options.max_iters:
				break

	def _run_batch(self):
		assert self.options.max_iters > 0

		for self.iter in range(0, self.options.max_iters):
			need_data = False

			if self.iter in self._iter_hooks:
				need_data = True

			if not need_data:
				for k in self._iter_hooks_every:
					if self.iter % k == 0:
						need_data = True
						break

			self.sim_step(self.iter, tracers=False, get_data=need_data)

			if need_data:
				for hook in self._iter_hooks.get(self.iter, []):
					hook()
				for k, v in self._iter_hooks_every.iteritems():
					if self.iter % k == 0:
						for hook in v:
							hook()


	def run(self):
		self._calc_screen_size()
		self._init_vis()
		self._init_geo()
		self._init_code()
		self._init_lbm()
		self._init_output()

		if self.options.benchmark:
			self._run_benchmark()
		elif self.options.batch:
			self._run_batch()
		else:
			self.vis.main(self)



"""Core Lattice Boltzmann simulation classes for Sailfish."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'
__version__ = '0.2-alpha1'

import inspect
import math
import numpy
import os
import sys
import time

import optparse
from optparse import OptionGroup, OptionParser

from mako.lookup import TemplateLookup

from sailfish import geo, io, sym, vis

SUPPORTED_BACKENDS = {'cuda': 'backend_cuda', 'opencl': 'backend_opencl'}
VIS_MODULES = ['vis2d', 'vis3d', 'vis_surf']

for backend in SUPPORTED_BACKENDS.values():
    try:
        __import__('sailfish', fromlist=[backend])
    except ImportError:
        pass

for visbackend in VIS_MODULES:
    try:
        __import__('sailfish', fromlist=[visbackend])
    except ImportError:
        pass

def get_backends():
    """Get a list of available backends."""
    return sorted([k for k, v in SUPPORTED_BACKENDS.iteritems()
        if ('sailfish.%s' % v) in sys.modules])

def get_backend_module(backend):
    return sys.modules['sailfish.%s' % SUPPORTED_BACKENDS[backend]]

def get_vis_engines():
    """Get a list of available visaulization engines."""

    ret = {}

    for v in VIS_MODULES:
        module = 'sailfish.%s' % v
        if module not in sys.modules:
            continue

        for class_name, obj in inspect.getmembers(sys.modules[module]):
            try:
                if vis.FluidVis in inspect.getmro(obj):
                    ret[obj.name] = obj
            except AttributeError:
                pass

    return ret

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
    t = re.sub('([0-9]+\.[0-9]*(e-?[0-9]*)?)f([^a-zA-Z0-9\.])', '\\1\\3',
               src.replace('float', 'double'))
    t = t.replace('logf(', 'log(')
    t = t.replace('expf(', 'exp(')
    t = t.replace('powf(', 'pow(')
    return t

class Hook(object):
    """Class wrapping hook called after simulation iteration.
    :param function: function to execute when hook is called
    :param need_data: whether hook needs transmitting data from device to host
    """
    def __init__(self, function, need_data=True):
        self.need_data = need_data
        self.function = function
    def __call__(self):
        self.function()

class LBMSim(object):
    """Base class for LBM simulations. Descendant classes should be declared for specific simulations."""

    #: The filename base for screenshots.
    filename = 'lbm_sim'

    #: The command to use to automatically format the compute unit source code.
    format_cmd = (r"sed -i -e '{{:s;N;\#//#{{p ;d}}; \!#!{{p;d}} ; s/\n//g;t s}}' {file} ; "
                  r"sed -i -e 's/}}/}}\n\n/g' {file} ; indent -linux -sob -l120 {file} ; "
                  r"sed -i -e '/^$/{{N; s/\n\([\t ]*}}\)$/\1/}}' -e '/{{$/{{N; s/{{\n$/{{/}}' {file}")
    # The first sed call removes all newline characters except for those terminating lines
    # that are preprocessor directives (starting with #) or single line comments (//).

    #: File name of the mako template containing the kernel code.
    kernel_file = 'single_fluid.mako'

    @property
    def time(self):
        """The current simulation time in simulation units."""
        # We take the size of the time step to be proportional to the square
        # of the discrete space interval, which in turn is
        # 1/(smallest dimension of the lattice).
        return self.iter_ * self.dt

    @property
    def dt(self):
        """Size of the time step in simulation units."""
        return self.geo.dx**2

    @property
    def dist(self):
        """The current distributions array.

        .. warning:: use :meth:`hostsync_dist` before accessing this property
        """
        return self.dist1

    @property
    def constants(self):
        return []

    def _add_options(self, parser, lb_group):
        """Add simulation options common to a class of simulations.

        Descendant classes (e.g. free surface, single fluid, etc) should use
        this method to provide their own generic options common to all
        simulations.

        :param parser: instance of the optparse.OptionParser class
        :param lb_group: instance of optparser.OptionGroup class representing
            core LB engine settings
        :rtype: iterable of optparse.OptionGroup instances or None
        """
        pass

    def __init__(self, geo_class, options=[], args=None, defaults=None):
        """
        :param geo_class: geometry class to use for the simulation
        :param options: iterable of ``optparse.Option`` instances representing additional
          options accepted by this simulation
        :param args: command line arguments
        :param defaults: a dictionary specifying the default values for any simulation options.
          These take precedence over the default values specified in ``optparse.Option`` objects.
        """
        self._t_start = time.time()

        if args is None:
            args = sys.argv[1:]

        supported_backends = get_backends()
        supported_vis = get_vis_engines()

        if not supported_backends:
            raise ValueError('There are no supported compute backends on your system. Make sure pycuda or pyopencl are correctly installed.')

        self.geo_class = geo_class

        parser = OptionParser()
        parser.add_option('-q', '--quiet', dest='quiet', help='reduce verbosity', action='store_true', default=False)
        parser.add_option('-v', '--verbose', dest='verbose', help='print additional info about the simulation', action='store_true', default=False)

        group = OptionGroup(parser, 'LB engine settings')
        group.add_option('--precision', dest='precision', help='precision (single, double)', type='choice', choices=['single', 'double'], default='single')
        group.add_option('--block_size', dest='block_size', help='size of the block of threds', type='int', action='store', default=64)
        group.add_option('--lat_nx', dest='lat_nx', help='lattice width', type='int', action='store', default=128)
        group.add_option('--lat_ny', dest='lat_ny', help='lattice height', type='int', action='store', default=128)
        group.add_option('--lat_nz', dest='lat_nz', help='lattice depth', type='int', action='store', default=1)
        group.add_option('--periodic_x', dest='periodic_x', help='lattice periodic in the X direction', action='store_true', default=False)
        group.add_option('--periodic_y', dest='periodic_y', help='lattice periodic in the Y direction', action='store_true', default=False)
        group.add_option('--periodic_z', dest='periodic_z', help='lattice periodic in the Z direction', action='store_true', default=False)
        group.add_option('--tracers', dest='tracers', help='number of tracer particles', type='int', action='store', default=32)
        group.add_option('--visc', dest='visc', help='viscosity', type='float', action='store', default=0.01)
        group.add_option('--every', dest='every',
            help='update the data on the host every N steps', metavar='N',
            type='int', action='store', default=100)

        group.add_option('--from', dest='from_',
            help='update the data on the host from N steps', metavar='N',
            type='int', action='store', default=0)


        class_options = self._add_options(parser, group)
        parser.add_option_group(group)

        for backend in supported_backends:
            group = OptionGroup(parser, '"%s" backend settings' % backend)
            opts = get_backend_module(backend).backend.add_options(group)
            if opts:
                parser.add_option_group(group)

        for name, cls in supported_vis.iteritems():
            group = OptionGroup(parser, '"%s" visualizaton engine settings' % (name))
            opts = cls.add_options(group)
            if opts:
                parser.add_option_group(group)

        group = OptionGroup(parser, 'Run mode settings')
        group.add_option('--backend', dest='backend', help='backend', type='choice', choices=supported_backends, default=supported_backends[0])
        group.add_option('--benchmark', dest='benchmark', help='benchmark mode, implies no visualization', action='store_true', default=False)
        group.add_option('--max_iters', dest='max_iters', help='number of iterations to run in benchmark/batch mode', action='store', type='int', default=0)
        group.add_option('--batch', dest='batch', help='run in batch mode, with no visualization', action='store_true', default=False)
        group.add_option('--nobatch', dest='batch', help='run in interactive mode', action='store_false')
        group.add_option('--vis', dest='vis', help='visualization module to use', type='choice',
                choices=supported_vis.keys(), default='pygame')
        group.add_option('--save_src', dest='save_src', help='file to save the CUDA/OpenCL source code to', action='store', type='string', default='')
        group.add_option('--use_src', dest='use_src', help='CUDA/OpenCL source to use instead of the automatically generated one', action='store', type='string', default='')
        group.add_option('--noformat_src', dest='format_src', help='do not format the generated source code', action='store_false', default=True)
        group.add_option('--output', dest='output', help='save simulation results to FILE', metavar='FILE', action='store', type='string', default='')
        group.add_option('--output_format', dest='output_format', help='output format', type='choice',
                choices=io.format_name_to_cls.keys(), default='npy')
        group.add_option('--use_mako_cache', dest='mako_cache',
                help='cache the generated Mako templates in /tmp/sailfish_modules-$USER', action='store_true',
                default=False)
        group.add_option('--save_checkpoint', help='base name of the checkpoint file', type='string', default='')
        group.add_option('--restore_checkpoint', help='name of the checkpoint file from which to restore the starting state of the simulation', type='string', default='')
        parser.add_option_group(group)

        if class_options is not None:
            for group in class_options:
                parser.add_option_group(group)

        group = OptionGroup(parser, 'Simulation-specific options')
        for option in options:
            group.add_option(option)

        if options:
            parser.add_option_group(group)

        self.options = Values(parser.defaults)
        parser.parse_args(args, self.options)

        if not supported_vis:
            if not self.options.batch:
                print 'Warning: no visualization modules are available, switching to batch mode.'
            self.options.batch = True

        # Set default command line values for unspecified options.  This is different
        # than the default values provided above, as these cannot be changed by
        # subclasses.
        if defaults is not None:
            for k, v in defaults.iteritems():
                if k not in self.options.specified:
                    setattr(self.options, k, v)

        # Whether to use the macroscopic fields to set the initial distributions.
        self._ic_fields = False
        self.num_tracers = 0
        self.iter_ = 0
        self._mlups_calls = 0
        self._mlups = 0.0
        self.clear_hooks()
        self.backend = get_backend_module(self.options.backend).backend(self.options)
        if not self.options.quiet:
            print 'Using the "%s" backend.' % self.options.backend

        if not self._is_double_precision():
            self.float = numpy.float32
        else:
            self.float = numpy.float64

        self.S = sym.S()
        self.forces = {}
        self._force_couplings = {}
        self._force_term_for_eq = {}
        self.vis = vis.FluidVis()
        self.output_fields = {}
        self.output_vectors = {}

        # Fields accessed via image/texture references to speed-up
        # non-local data access.
        self.image_fields = set()

    def _set_grid(self, name):
        for x in sym.KNOWN_GRIDS:
            if x.__name__ == name:
                self.grid = x
                break

    def _set_model(self, *models):
        for x in models:
            if self.grid.model_supported(x):
                self.lbm_model = x
                break

    def curr_dists(self):
        if self.iter_ & 1:
            return [self.gpu_dist1b]
        else:
            return [self.gpu_dist1a]

    def hostsync_dist(self):
        """Copy the current distributions from the compute unit to the host.

        The distributions are then available in :attr:`dist`.
        """
        for dist in self.curr_dists():
            self.backend.from_buf(dist)
        self.backend.sync()

    def hostsync_velocity(self):
        """Copy the current velocity field from the compute unit to the host.

        The velocity field is then availble in :attr:`vx`, :attr:`vy` and :attr:`vz`.
        """
        for vel in self.gpu_velocity:
            self.backend.from_buf(vel)
        self.backend.sync()

    def hostsync_density(self):
        """Copy the current density field from the compute unit to the host.

        The density field is then available in :attr:`rho`.
        """
        self.backend.from_buf(self.gpu_rho)
        self.backend.sync()

    def hostsync_tracers(self):
        """Copy the tracer positions from the compute unit to the host.

        The distributions are then available in :attr:`tracer_x`, :attr:`tracer_y` and :attr:`tracer_z`.
        """
        for loc in self.gpu_tracer_loc:
            self.backend.from_buf(loc)
        self.backend.sync()

    @property
    def sim_info(self):
        """A dictionary of simulation settings."""
        ret = {}
        ret['grid'] = self.grid.__name__
        ret['precision'] = self.options.precision
        ret['size'] = tuple(reversed(self.shape))
        ret['visc'] = self.options.visc
        ret['dx'] = self.geo.dx
        ret['dt'] = self.dt
        return ret

    def _is_double_precision(self):
        return self.options.precision == 'double'

    def _init_vis(self):
        self._timed_print('Initializing visualization engine.')

        if not self.options.benchmark and not self.options.batch:

            engines = get_vis_engines()

            if self.grid.dim in engines[self.options.vis].dims:
                self.vis = (engines[self.options.vis])(self)
            else:
                self._timed_print('Warning: Selected visualization engine "%s" does not support %dD data visualization.' %
                        (self.options.vis, self.grid.dim))
                for name, cls in engines.iteritems():
                    if self.grid.dim in cls.dims:
                        self._timed_print('Warning: Falling back to the "%s" visualization engine.' % name)
                        self.vis = cls(self)
                        break
                else:
                    self._timed_print('Warning: Falling back to batch mode.')
                    self.options.batch = True

    def add_iter_hook(self, i, func, every=False, need_data=True):
        """Add a hook that will be executed during the simulation.

        :param i: number of the time step after which the hook is to be run
        :param func: callable representing the hook
        :param every: if ``True``, the hook will be executed every *i* steps
        """
        if every:
            self.iter_hooks_every.setdefault(i, []).append(Hook(func, need_data))
        else:
            self.iter_hooks.setdefault(i, []).append(Hook(func, need_data))

    def clear_hooks(self):
        """Remove all hooks."""
        self.iter_hooks = {}
        self.iter_hooks_every = {}

    def get_reduction(self, reduce_expr, map_expr, neutral, *args):
        """Generate and return reduction kernel; see PyCUDA and PyOpenCL
        documentation of ReductionKernel for detailed description.
        Function expects buffers that are in the host address space,
        e.g. sim.rho, sim.velocity, etc.

        :param reduce_expr: expression used to reduce two values into one,
            must use a and b as values names, e.g. 'a+b'
        :param map_expr: expression used to map value from input array,
            arrays are named x0, x1, etc., e.g. 'x0[i]*x1[i]
        :param neutral: neutral value in reduce_expr, e.g. '0'
        :param args: buffers on which to calculate reduction, e.g. sim.rho
        """
        arrays = []
        for arg in args:
            if arg.base is not None:
                array = arg.base
            else:
                array = arg
            for dev_buf, host_buf in self.backend.buffers.iteritems():
                if array is host_buf:
                    arrays.append(dev_buf)
                    break
        return self.backend.get_reduction_kernel(reduce_expr, map_expr, neutral, *arrays)

    def get_tau(self):
        return self.float((6.0 * self.options.visc + 1.0)/2.0)

    def get_dist_size(self):
        return self.arr_nx * self.arr_ny * self.arr_nz

    def _timed_print(self, info):
        if self.options.verbose:
            print '[{0:07.2f}] {1}'.format(time.time() - self._t_start, info)

    def _init_shape(self):
        self.arr_nx = int(math.ceil(float(self.options.lat_nx) / self.options.block_size)) * self.options.block_size
        self.arr_ny = self.options.lat_ny
        self.arr_nz = self.options.lat_nz

        # Particle distributions in host memory.
        if self.grid.dim == 2:
            self.shape = (self.options.lat_ny, self.options.lat_nx)
        else:
            self.shape = (self.options.lat_nz, self.options.lat_ny, self.options.lat_nx)

    def _init_geo(self):
        self._timed_print('Initializing geometry.')

        # Simulation geometry.
        self.geo = self.geo_class(list(reversed(self.shape)), self.options,
                self.float, self.backend, self)

        self._init_fields(not self.geo.ic_fields)

        if self.geo.ic_fields:
            self.geo.init_fields()
            self._ic_fields = True
        else:
            self.geo.init_dist(self.dist1)
        self.geo_params = self.float(self.geo.params)

    def _init_post_geo(self):
        pass

    def _update_ctx(self, ctx):
        pass

    def _init_code(self):
        self._timed_print('Preparing compute device code.')

        # Clear all locale settings, we do not want them affecting the
        # generated code in any way.
        import locale
        locale.setlocale(locale.LC_ALL, 'C')

        if self.options.mako_cache:
            import pwd
            lookup = TemplateLookup(directories=sys.path,
                    module_directory='/tmp/sailfish_modules-%s' % (pwd.getpwuid(os.getuid())[0]))
        else:
            lookup = TemplateLookup(directories=sys.path)

        lbm_tmpl = lookup.get_template(os.path.join('sailfish/templates', self.kernel_file))

        self.tau = self.get_tau()
        ctx = {}
        ctx['dim'] = self.grid.dim
        ctx['block_size'] = self.options.block_size

        # Size of the lattice.
        ctx['lat_ny'] = self.options.lat_ny
        ctx['lat_nx'] = self.options.lat_nx
        ctx['lat_nz'] = self.options.lat_nz

        # Actual size of the array, including any padding.
        ctx['arr_nx'] = self.arr_nx
        ctx['arr_ny'] = self.arr_ny
        ctx['arr_nz'] = self.arr_nz
        ctx['periodic_x'] = int(self.options.periodic_x)
        ctx['periodic_y'] = int(self.options.periodic_y)
        ctx['periodic_z'] = int(self.options.periodic_z)
        ctx['num_params'] = len(self.geo_params)
        ctx['geo_params'] = self.geo_params
        ctx['tau'] = self.tau
        ctx['visc'] = self.float(self.options.visc)
        ctx['backend'] = self.options.backend
        ctx['dist_size'] = self.get_dist_size()
        ctx['pbc_offsets'] = [{-1: self.options.lat_nx,
                                1: -self.options.lat_nx},
                              {-1: self.options.lat_ny*self.arr_nx,
                                1: -self.options.lat_ny*self.arr_nx},
                              {-1: self.options.lat_nz*self.arr_ny*self.arr_nx,
                                1: -self.options.lat_nz*self.arr_ny*self.arr_nx}]
        ctx['bnd_limits'] = [self.options.lat_nx, self.options.lat_ny, self.options.lat_nz]
        ctx['loc_names'] = ['gx', 'gy', 'gz']
        ctx['periodicity'] = [int(self.options.periodic_x), int(self.options.periodic_y),
                            int(self.options.periodic_z)]
        ctx['grid'] = self.grid
        ctx['sim'] = self
        ctx['model'] = self.lbm_model
        ctx['bgk_equilibrium'] = self.equilibrium
        ctx['bgk_equilibrium_vars'] = self.equilibrium_vars
        ctx['constants'] = self.constants
        ctx['grids'] = [self.grid]

        ctx['simtype'] = 'lbm'
        ctx['forces'] = self.forces
        ctx['force_couplings'] = self._force_couplings
        ctx['force_for_eq'] = self._force_term_for_eq
        ctx['image_fields'] = self.image_fields
        ctx['precision'] = self.options.precision

        # TODO: Find a more general way of specifying whether sentinels are
        # necessary.
        ctx['propagation_sentinels'] = (self.options.bc_wall == 'halfbb')

        self._update_ctx(ctx)
        ctx.update(self.geo.get_defines())
        ctx.update(self.backend.get_defines())
        src = lbm_tmpl.render(**ctx)

        if self._is_double_precision():
            src = _convert_to_double(src)

        if self.options.save_src:
            with open(self.options.save_src, 'w') as fsrc:
                print >>fsrc, src

            if self.options.format_src:
                os.system(self.format_cmd.format(file=self.options.save_src))

        # If external source code was requested, ignore the code that we have
        # just generated above.
        if self.options.use_src:
            with open(self.options.use_src, 'r') as fsrc:
                src = fsrc.read()

        self.mod = self.backend.build(src)

    def _get_strides(self, type_):
        t = type_().nbytes
        if self.grid.dim == 3:
            strides = (self.arr_ny * self.arr_nx * t, self.arr_nx * t, t)
            size = self.arr_nx * self.arr_ny * self.arr_nz
        else:
            strides = (self.arr_nx * t, t)
            size = self.arr_nx * self.arr_ny
        return (strides, size)

    def make_field(self, name=None, output=False):
        """Create a new numpy array representing a scalar field used in the simulation.

        This method automatically takes care of the field type, shape and strides.
        """
        strides, size = self._get_strides(self.float)
        field = numpy.ndarray(self.shape, buffer=numpy.zeros(size, dtype=self.float),
                              dtype=self.float, strides=strides)
        if output and name is not None:
            self.output_fields[name] = field
        return field

    def make_vector_field(self, name=None, output=False):
        components = []

        for x in range(0, self.grid.dim):
            components.append(self.make_field())

        if output and name is not None:
            self.output_vectors[name] = components

        return components

    def make_int_field(self):
        strides, size = self._get_strides(numpy.uint32)
        return numpy.ndarray(self.shape, buffer=numpy.zeros(size, dtype=numpy.uint32),
                             dtype=numpy.uint32, strides=strides)

    def make_dist(self, grid):
        strides, size = self._get_strides(self.float)
        shape = [len(grid.basis)] + list(self.shape)
        strides = [strides[-1]*size] + list(strides)
        size *= len(grid.basis)
        return numpy.ndarray(shape, buffer=numpy.zeros(size, dtype=self.float),
                             dtype=self.float, strides=strides)

    def get_dist_bytes(self, grid):
        strides, size = self._get_strides(self.float)
        size *= grid.Q * strides[-1]
        return size

    def _init_fields(self, need_dist):
        """Initialize the data fields used in the simulation.

        All the data field arrays are first allocated on the host, and filled with
        default values.  These can then be overridden when the distributions for the
        simulation are initialized.  Afterwards, the fields are copied to the compute
        unit in :meth:`_init_compute`.

        :param need_dist: if True, allocate the particle distributions on the host.
            Otherwise they are stored on the compute unit only.
        """
        self._timed_print('Preparing the data fields.')

        self.velocity = self.make_vector_field('v', True)

        if self.grid.dim == 2:
            self.vx, self.vy = self.velocity
        else:
            self.vx, self.vy, self.vz = self.velocity

        self.rho = self.make_field('rho', True)

        self.vis.add_field(lambda: numpy.sqrt(numpy.square(self.vx) + numpy.square(self.vy)),
                'velocity magnitude')
        self.vis.add_field(self.vx, 'X velocity component', True)
        self.vis.add_field(self.vy, 'Y velocity component', True)
        if self.grid.dim == 3:
            self.vis.add_field(self.vz, 'Z velocity component', True)

        self.vis.add_field(lambda:
                (numpy.roll(self.vy, 1, 1) - numpy.roll(self.vy, -1, 1)) -
                (numpy.roll(self.vx, 1, 0) - numpy.roll(self.vx, -1, 0)),
                'vorticity from the X and Y velocity components', True)

        # Tracer particles.
        if self.num_tracers:
            self.tracer_x = numpy.random.random_sample(self.num_tracers).astype(self.float) * self.options.lat_nx
            self.tracer_y = numpy.random.random_sample(self.num_tracers).astype(self.float) * self.options.lat_ny
            self.tracer_loc = [self.tracer_x, self.tracer_y]
            if self.grid.dim == 3:
                self.tracer_z = numpy.random.random_sample(self.num_tracers).astype(self.float) * self.options.lat_nz
                self.tracer_loc.append(self.tracer_z)

        else:
            self.tracer_loc = []

        if need_dist:
            self.dist1 = self.make_dist(self.grid)

    def _init_compute_fields(self):
        self._timed_print('Preparing the compute unit data fields.')
        # Velocity.
        self.gpu_vx = self.backend.alloc_buf(like=self.vx)
        self.gpu_vy = self.backend.alloc_buf(like=self.vy)
        self.gpu_velocity = [self.gpu_vx, self.gpu_vy]

        if self.grid.dim == 3:
            self.gpu_vz = self.backend.alloc_buf(like=self.vz)
            self.gpu_velocity.append(self.gpu_vz)

        # Density.
        self.gpu_rho = self.backend.alloc_buf(like=self.rho)
        self.gpu_mom0 = [self.gpu_rho]

        # Tracer particles.
        if self.num_tracers:
            self.gpu_tracer_x = self.backend.alloc_buf(like=self.tracer_x)
            self.gpu_tracer_y = self.backend.alloc_buf(like=self.tracer_y)
            self.gpu_tracer_loc = [self.gpu_tracer_x, self.gpu_tracer_y]
            if self.grid.dim == 3:
                self.gpu_tracer_z = self.backend.alloc_buf(like=self.tracer_z)
                self.gpu_tracer_loc.append(self.gpu_tracer_z)
        else:
            self.gpu_tracer_loc = []

        # Particle distributions in device memory, A-B access pattern.
        if not self._ic_fields:
            self.gpu_dist1a = self.backend.alloc_buf(like=self.dist1)
            self.gpu_dist1b = self.backend.alloc_buf(like=self.dist1)
        else:
            self.gpu_dist1a = self.backend.alloc_buf(size=self.get_dist_bytes(self.grid), wrap_in_array=False)
            self.gpu_dist1b = self.backend.alloc_buf(size=self.get_dist_bytes(self.grid), wrap_in_array=False)

    def _init_compute_ic(self):
        if not self._ic_fields:
            # Nothing to do, the initial distributions have already been
            # set and copied to the GPU in _init_compute_fields.
            return

        self._timed_print('Initializing distributions on compute unit.')

        args1 = [self.gpu_dist1a] + self.gpu_velocity + [self.gpu_rho]
        args2 = [self.gpu_dist1b] + self.gpu_velocity + [self.gpu_rho]

        kern1 = self.backend.get_kernel(self.mod, 'SetInitialConditions',
                    args=args1,
                    args_format='P'*len(args1),
                    block=self._kernel_block_size())

        kern2 = self.backend.get_kernel(self.mod, 'SetInitialConditions',
                    args=args2,
                    args_format='P'*len(args2),
                    block=self._kernel_block_size())

        self.backend.run_kernel(kern1, self.kern_grid_size)
        self.backend.run_kernel(kern2, self.kern_grid_size)
        self.backend.sync()

    def _kernel_block_size(self):
        if self.grid.dim == 2:
            return (self.options.block_size, 1)
        else:
            return (self.options.block_size, 1, 1)

    def _init_compute_kernels(self):
        self._timed_print('Preparing the compute unit kernels.')

        # Kernel arguments.
        args_tracer2 = [self.gpu_dist1a, self.geo.gpu_map] + self.gpu_tracer_loc
        args_tracer1 = [self.gpu_dist1b, self.geo.gpu_map] + self.gpu_tracer_loc
        args1 = ([self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_rho] + self.gpu_velocity +
                 [numpy.uint32(0)])
        args2 = ([self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_rho] + self.gpu_velocity +
                 [numpy.uint32(0)])

        # Special argument list for the case where macroscopic quantities data is to be
        # saved in global memory, i.e. a visualization step.
        args1v = ([self.geo.gpu_map, self.gpu_dist1a, self.gpu_dist1b, self.gpu_rho] + self.gpu_velocity +
                  [numpy.uint32(1)])
        args2v = ([self.geo.gpu_map, self.gpu_dist1b, self.gpu_dist1a, self.gpu_rho] + self.gpu_velocity +
                  [numpy.uint32(1)])

        k_block_size = self._kernel_block_size()
        kernel_name = 'CollideAndPropagate'

        kern_cnp1 = self.backend.get_kernel(self.mod, kernel_name,
                    args=args1,
                    args_format='P'*(len(args1)-1)+'i',
                    block=k_block_size)
        kern_cnp2 = self.backend.get_kernel(self.mod, kernel_name,
                    args=args2,
                    args_format='P'*(len(args2)-1)+'i',
                    block=k_block_size)
        kern_cnp1s = self.backend.get_kernel(self.mod, kernel_name,
                    args=args1v,
                    args_format='P'*(len(args1v)-1)+'i',
                    block=k_block_size)
        kern_cnp2s = self.backend.get_kernel(self.mod, kernel_name,
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

        # For occupancy analysis in performance tests.
        self._lb_kernel = kern_cnp1

        # Map: iteration parity -> kernel arguments to use.
        self.kern_map = {
            0: (kern_cnp1, kern_cnp1s, kern_trac1),
            1: (kern_cnp2, kern_cnp2s, kern_trac2),
        }

        if self.grid.dim == 2:
            self.kern_grid_size = (self.arr_nx/self.options.block_size, self.arr_ny)
        else:
            self.kern_grid_size = (self.arr_nx/self.options.block_size * self.arr_ny, self.arr_nz)

    def _lbm_step(self, get_data, **kwargs):
        kerns = self.kern_map[self.iter_ & 1]

        if get_data:
            self.backend.run_kernel(kerns[1], self.kern_grid_size)
            if kwargs.get('tracers'):
                self.backend.run_kernel(kerns[2], (self.num_tracers,))
                self.hostsync_tracers()
            self.hostsync_velocity()
            self.hostsync_density()
        else:
            self.backend.run_kernel(kerns[0], self.kern_grid_size)
            if kwargs.get('tracers'):
                self.backend.run_kernel(kerns[2], (self.num_tracers,))

    def sim_step(self, tracers=False, get_data=False, **kwargs):
        """Perform a single step of the simulation.

        :param tracers: if ``True``, the position of tracer particles will be updated
        :param get_data: if ``True``, macroscopic variables will be copied from the compute unit
          and made available as properties of this class
        """
        i = self.iter_

        if (not self.options.benchmark and (not self.options.batch or
            (self.options.batch and self.options.output)) and
            i % self.options.every == 0 and i >= self.options.from_) or get_data:

            self._lbm_step(True, tracers=tracers, **kwargs)

            if self.options.output and i % self.options.every == 0 and i >= self.options.from_:
                self.output.save(i)
        else:
            self._lbm_step(False, tracers=tracers, **kwargs)

        self.iter_ += 1

    def _init_output(self):
        if self.options.output:
            self.output = io.format_name_to_cls[self.options.output_format](self.options.output, self)

    def get_mlups(self, tdiff, iters=None):
        if iters is not None:
            it = iters
        else:
            it = self.options.every

        mlups = float(it) * self.geo.count_active_nodes() * 1e-6 / tdiff
        self._mlups = (mlups + self._mlups * self._mlups_calls) / (self._mlups_calls + 1)
        self._mlups_calls += 1
        return (self._mlups, mlups)

    # TODO: Move this to a separate class.
    def output_ascii(self, file):
        if self.grid.dim == 3:
            rho = self.geo.mask_array_by_fluid(self.rho)
            vx = self.geo.mask_array_by_fluid(self.vx)
            vy = self.geo.mask_array_by_fluid(self.vy)
            vz = self.geo.mask_array_by_fluid(self.vz)

            for z in range(0, vx.shape[0]):
                for y in range(0, vx.shape[1]):
                    for x in range(0, vx.shape[2]):
                        print >>file, rho[z,y,x], vx[z,y,x], vy[z,y,x], vz[z,y,x]
                    print >>file, ''
        else:
            rho = self.geo.mask_array_by_fluid(self.rho)
            vx = self.geo.mask_array_by_fluid(self.vx)
            vy = self.geo.mask_array_by_fluid(self.vy)

            for y in range(0, vx.shape[0]):
                for x in range(0, vx.shape[1]):
                    print >>file, rho[y,x], vx[y,x], vy[y,x]
                print >>file, ''

    def _run_benchmark(self):
        cycles = self.options.every

        print '# iters mlups_avg mlups_curr'

        import time

        while True:
            t_prev = time.time()

            for i in xrange(0, cycles):
                self.sim_step(tracers=False)

            self.backend.sync()
            t_now = time.time()
            print self.iter_,

            avg, curr = self.get_mlups(t_now - t_prev, cycles)
            print '%.2f %.2f' % (avg, curr)
            self._bench_avg = avg

            if self.options.max_iters <= self.iter_:
                break

    def _run_batch(self):
        assert self.options.max_iters > 0

        for i in range(0, self.options.max_iters):
            need_data = False

            if self.iter_ in self.iter_hooks:
                for hook in self.iter_hooks.get(self.iter_, []):
                    need_data = need_data or hook.need_data

            if not need_data:
                for k, hooks in self.iter_hooks_every.iteritems():
                    if self.iter_ % k == 0:
                        for hook in hooks:
                            need_data = need_data or hook.need_data

            self.sim_step(tracers=False, get_data=need_data)

            for hook in self.iter_hooks.get(self.iter_-1, []):
                hook()
            for k, v in self.iter_hooks_every.iteritems():
                if (self.iter_-1) % k == 0:
                    for hook in v:
                        hook()


    def run(self):
        """Run the simulation.

        This automatically handles any options related to visualization and the benchmark and batch modes.
        """
        if not self.grid.model_supported(self.lbm_model):
            raise ValueError('The LBM model "%s" is not supported with '
                    'grid type %s' % (self.lbm_model, self.grid.__name__))

        self._init_shape()
        self._init_vis()
        self._init_geo()
        self._init_post_geo()
        self._init_code()
        self._init_compute_fields()
        self._init_compute_kernels()
        if self.options.restore_checkpoint:
            self.restore_checkpoint(self.options.restore_checkpoint)
        else:
            self._init_compute_ic()
        self._init_output()

        self._timed_print('Starting the simulation...')
        self._timed_print('Simulation parameters:')

        if self.options.verbose:
            for k, v in sorted(self.sim_info.iteritems()):
                print '  {0}: {1}'.format(k, v)

        if self.options.benchmark:
            self._run_benchmark()
        elif self.options.batch:
            self._run_batch()
        else:
            self.vis.main()

        if self.options.save_checkpoint:
            self.save_checkpoint(self.options.save_checkpoint)

    def save_checkpoint(self, base_fname):
        self._timed_print('Creating checkpoint file...')

        if not self._ic_fields:
            raise NotImplementedError()

        state = {}
        state['cmdline'] = ' '.join(sys.argv)
        state['iteration'] = self.iter_

        for i, dist in enumerate(self.curr_dists()):
            arr = self.make_dist(self.grid)
            self.backend.from_buf(dist, arr)
            state['grid%d' % i] = arr

        numpy.savez(base_fname, **state)

    def restore_checkpoint(self, fname):
        self._timed_print('Restoring state from a checkpoint file...')
        state = numpy.load(fname)
        self._timed_print('Previous invocation was: %s' % state['cmdline'])
        self.iter_ = 1
        dist_arrays = []
        for i, dist in enumerate(self.curr_dists()):
            arr = self.make_dist(self.grid)
            arr[:] = state['grid%d' % i]
            dist_arrays.append(arr)
            self.backend.to_buf(dist, arr)
        self.iter_ = 0
        for i, dist in enumerate(self.curr_dists()):
            self.backend.to_buf(dist, dist_arrays[i])

    def add_body_force(self, force, grid=0, accel=True):
        """Add a constant global force field acting on the fluid.

        :param force: n-vector of the force values
        :param grid: grid number on which this force is acting
        :param accel: if ``True``, the added field is an acceleration field, otherwise
            it is an actual force field
        """
        if len(force) != self.grid.dim:
            raise ValueError('The dimensionality of the force vector needs to be the same as that of the grid.')

        if grid not in self.forces:
            self.forces[grid] = {}

        self.forces.setdefault(grid, {}).setdefault(
                accel, numpy.zeros(self.grid.dim, dtype=self.float))
        a = self.forces[grid][accel] + numpy.float64(force)
        self.forces[grid][accel] = a

    def use_force_for_eq(self, force, grid=0):
        self._force_term_for_eq[grid] = force

    def add_force_coupling(self, i, j, g):
        self._force_couplings[(i, j)] = g

    def add_nonlocal_field(self, num):
        if self.options.backend == 'cuda':
            self.image_fields.add(num)

    def bind_nonlocal_field(self, gpu_buf, num):
        if self.options.backend == 'cuda':
            strides, _ = self._get_strides(self.float)
            return self.backend.nonlocal_field(self.mod, gpu_buf, num, self.shape, strides)


class FreeSurfaceLBMSim(LBMSim):
    @property
    def sim_info(self):
        ret = LBMSim.sim_info.fget(self)
        ret['gravity'] = self.gravity
        return ret

    def __init__(self, geo_class, options=[], args=None, defaults=None):
        LBMSim.__init__(self, geo_class, options, args, defaults)
        self._set_grid('D2Q9')
        self._set_model('bgk')
        self.equilibrium, self.equilibrium_vars = sym.shallow_water_equilibrium(self.grid)
        self.gravity = self.options.gravity

    def _add_options(self, parser, lb_group):
        lb_group.add_option('--gravity', dest='gravity',
            help='gravitational acceleration', action='store', type='float',
            default=0.001)
        return []

    def _update_ctx(self, ctx):
        ctx['gravity'] = self.gravity
        ctx['bc_wall'] = 'fullbb'
        ctx['bc_velocity'] = None
        ctx['bc_pressure'] = None
        ctx['bc_wall_'] = geo.get_bc('fullbb')
        ctx['bc_velocity_'] = geo.get_bc('fullbb')
        ctx['bc_pressure_'] = geo.get_bc('fullbb')


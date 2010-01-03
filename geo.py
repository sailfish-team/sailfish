import cPickle as pickle
import os
import sys
import numpy

import sym

# Abstract class implementation, from Peter's Norvig site.
def abstract():
	import inspect
	caller = inspect.getouterframes(inspect.currentframe())[1][3]
	raise NotImplementedError('%s must be implemented in subclass' % caller)

class LBMGeo(object):
	"""Abstract class for the LBM geometry."""

	# Dimensionality, needs to be overridden in child classes.
	dim = 0

	# Node types.
	NODE_FLUID = 0
	NODE_WALL = 1
	NODE_VELOCITY = 2
	NODE_PRESSURE = 3

	NODE_TYPES = [NODE_WALL, NODE_VELOCITY, NODE_PRESSURE]

	# Constants to specify node orientation.  This needs to match the order
	# in sym.basis.
	NODE_DIR_E = 0
	NODE_DIR_N = 1
	NODE_DIR_W = 2
	NODE_DIR_S = 3
	NODE_DIR_NE = 4
	NODE_DIR_NW = 5
	NODE_DIR_SW = 6
	NODE_DIR_SE = 7
	NODE_DIR_OTHER = 8

	NODE_TYPE_MASK = 0xfffffff0
	NODE_ORIENTATION_SHIFT = 4
	NODE_ORIENTATION_MASK = 0xf

	@classmethod
	def _encode_node(cls, orientation, type):
		"""Encode a node entry for the map of nodes.

		:param orientation: node orientation
		:param type: node type

		:rtype: node code
		"""
		return orientation | (type << cls.NODE_ORIENTATION_SHIFT)

	@classmethod
	def _decode_node_type(cls, code):
		return (code & cls.NODE_TYPE_MASK) >> cls.NODE_ORIENTATION_SHIFT

	@classmethod
	def _decode_node_orientation(cls, code):
		return (code & cls.NODE_ORIENTATION_MASK)

	@classmethod
	def _decode_node(cls, code):
		"""Decode an entry from the map of nodes.

		:param code: node code from the map

		:rtype: tuple of orientation, type
		"""
		return cls._decode_node_orientation(code), cls._decode_node_type(code)

	def __init__(self, shape, options, float, backend, sim, save_cache=True, use_cache=True):
		self.sim = sim
		self.shape = shape
		self.options = options
		self.backend = backend
		self.map = numpy.zeros(shape, numpy.int32)
		self.gpu_map = backend.alloc_buf(like=self.map)
		self.float = float
		self.save_cache = save_cache
		self.use_cache = use_cache

		# Map: object_id -> [(location, direction)]
		self._force_nodes = {}
		self.reset()

		# Cache for equilibrium distributions.  Sympy numerical evaluation
		# is expensive, so we try to avoid unnecessary recomputations.
		self.feq_cache = {}

	def _get_state(self):
		rdict = {
			'map': self.map,
			'_params': self._params,
			'_force_nodes': self._force_nodes,
			'_num_velocities': self._num_velocities,
			'_num_pressures': self._num_pressures
		}
		return rdict

	def _set_state(self, rdict):
		for k, v in rdict.iteritems():
			setattr(self, k, v)
		self._update_map()

	@property
	def dx(self):
		"""Lattice spacing in simulation units."""
		return 1.0/min(self.shape)

	def define_nodes(self):
		"""Define the types of all nodes in the simulation domain.

		Subclasses need to override this method to specify the geometry to be used
		in the simulation.

		Use :meth:`set_geo` and :meth:`fill_geo` to set the type of nodes.  By default,
		all nodes are set as fluid nodes.
		"""
		abstract

	def init_dist(self, dist):
		"""Initialize the particle distributions in the whole simulation domain.

		Subclasses need to override this method to provide initial conditions for
		the simulation.
		"""
		abstract

	def get_reynolds(self, viscosity):
		"""Get the Reynolds number for this geometry."""
		abstract

	def get_defines(self):
		abstract

	@property
	def cache_file(self):
		return '.sailfish_%s_%s_%s_%s' % (
				os.path.basename(sys.argv[0]), self.sim.grid.__name__,
				'-'.join(map(str, self.shape)), str(self.float().dtype))

	def _clear_state(self):
		self._params = None
		self.map = numpy.zeros(tuple(reversed(self.shape)), numpy.int32)
		self._velocity_map = numpy.zeros(shape=([self.dim] + list(self.map.shape)), dtype=self.float)
		self._pressure_map = numpy.zeros(shape=self.map.shape, dtype=self.float)
		self._num_velocities = 0
		self._num_pressures = 0

	@property
	def has_pressure_nodes(self):
		return self._num_pressures > 0

	@property
	def has_velocity_nodes(self):
		return self._num_velocities > 0

	def reset(self):
		"""Perform a full reset of the geometry."""

		if self.use_cache and os.path.exists(self.cache_file):
			with open(self.cache_file, 'r') as f:
				self._set_state(pickle.load(f))
			return

		self._clear_state()
		self.define_nodes()
		a = self.params
		self._postprocess_nodes()
		self._update_map()

		if self.save_cache:
			with open(self.cache_file, 'w') as f:
				pickle.dump(self._get_state(), f, pickle.HIGHEST_PROTOCOL)

	def _get_map(self, location):
		"""Get a node map entry.

		:param location: a tuple specifying the node location
		"""
		return self.map[tuple(reversed(location))]

	def _set_map(self, location, value):
		"""Set a node map entry.

		:param location: a tuple specifying the node location
		:param value: the node code as returned by _encode_node()
		"""
		self.map[tuple(reversed(location))] = value

	def _update_map(self):
		"""Copy the node map to the compute unit."""
		self.backend.to_buf(self.gpu_map, self.map)

	def set_geo(self, location, type, val=None, update=False):
		"""Set the type of a grid node.

		:param location: location of the node
		:param type: type of the node, one of the NODE_* constants
		:param val: optional argument for the node, e.g. the value of velocity or pressure
		:param update: whether to automatically update the geometry for the simulation
		"""
		self._set_map(location, numpy.int32(type))

		rloc = tuple(reversed(location))
		rloc2 = tuple([slice(None)] + list(rloc))

		if val is not None:
			if type == self.NODE_VELOCITY:
				if len(val) == self.dim:
					self._velocity_map[rloc2] = val
				else:
					raise ValueError('Invalid velocity specified.')
			elif type == self.NODE_PRESSURE:
				self._pressure_map[rloc] = val

		if update:
			self._postprocess_nodes(nodes=[location])
			self._update_map()

	def mask_array_by_fluid(self, array):
		"""Mask an array so that only fluid nodes are active.

		:param array: a numpy array of the same dimensionality as the simulation domain.
		  This will usually be an array containing the macroscopic variables (velocity, density).
		"""
		if get_bc(self.options.bc_wall).wet_nodes:
			return array
		mask = (self._decode_node_type(self.map) == self.NODE_WALL)
		return numpy.ma.array(array, mask=mask)

	def _prep_array_fill(self, out, location, target, shift=0):
		loc = list(reversed(location))

		if target is None:
			tg = [slice(None)] * len(location)

			# In order for numpy's array broadcasting to work correctly, the dimensions
			# indexed with the ':' operator must be to the right of any dimensions with
			# a specific numerical index, i.e. t[:,:,:] = b[0,0,:] works, but
			# t[:,:,:] = b[0,:,0] does not.
			#
			# To work around this, we swap the axes of the dist array to get the correct
			# form for broadcasting.
			start = None
			for	i in reversed(range(0, len(loc))):
				if start is None and loc[i] != slice(None):
					start = i
				elif start is not None and loc[i] == slice(None):
					t = loc[start]
					loc[start] = loc[i]
					loc[i] = t

					# shift = +1 if the first dimension is for different velocity directions.
					out = numpy.swapaxes(out, i+shift, start+shift)
					start -= 1
		else:
			tg = list(reversed(target))

		return out, loc, tg

	def fill_dist(self, location, dist, target=None):
		"""Fill (a part of) the simulation domain with distributions from a specific node(s).

		:param location: location of the node, a n-tuple.  The location can also be a row,
		    in which case the coordinate spanning the row should be set to slice(None).
		:param dist: the distribution array
		:param target: if not None, a n-tuple representing the area to which the data from
		    the specified node is to be propagated
		"""
		out, loc, tg = self._prep_array_fill(dist, location, target, shift=1)

		for i in range(0, len(self.sim.grid.basis)):
			addr = tuple([i] + loc)
			dest = tuple([i] + tg)
			out[dest] = out[addr]

	def fill_geo(self, location, target=None):
		"""Fill (a part of) the simulation domain with boundary conditions from a specific node(s).

		See :meth:`fill_dist` for a description of the parameters."""

		out, loc, tg = self._prep_array_fill(self.map, location, target)
		loc = tuple(loc)
		out[tg] = out[loc]

		out, loc, tg = self._prep_array_fill(self._pressure_map, location, target)
		loc = tuple(loc)
		out[tg] = out[loc]

		out, loc, tg = self._prep_array_fill(self._velocity_map, location, target, shift=1)
		for i in range(0, self.dim):
			addr = tuple([i] + loc)
			dest = tuple([i] + tg)
			out[dest] = out[addr]

	def velocity_to_dist(self, location, velocity, dist, rho=1.0):
		"""Set the distributions for a node so that the fluid there has a specific velocity.

		This function is used to set the initial conditions for the simulation.

		:param location: location of the node, a n-tuple
		:param velocity: velocity to set, a n-tuple
		:param dist: the distribution array
		:param rho: the density to use for the node
		"""
		if (rho, velocity) not in self.feq_cache:
			vals = []
			self.feq_cache[(rho, velocity)] = map(self.float,
					sym.eval_bgk_equilibrium(self.sim.grid, self.options.incompressible, velocity, rho))

		for i, val in enumerate(self.feq_cache[(rho, velocity)]):
			dist[i][tuple(reversed(location))] = val

	def _postprocess_nodes(self, nodes=None):
		"""Detect types of wall nodes and mark them appropriately.

		:param nodes: optional iterable of locations to postprocess
		"""
		abstract

	@property
	def params(self):
		if self._params is not None:
			return self._params

		ret = []
		i = 0

		if self.dim == 3:
			v1, i1 = numpy.unique1d(self._velocity_map[0,:,:,:], return_inverse=True)
			v2, i2 = numpy.unique1d(self._velocity_map[1,:,:,:], return_inverse=True)
			v3, i3 = numpy.unique1d(self._velocity_map[2,:,:,:], return_inverse=True)
		else:
			v1, i1 = numpy.unique1d(self._velocity_map[0,:,:], return_inverse=True)
			v2, i2 = numpy.unique1d(self._velocity_map[1,:,:], return_inverse=True)

		i1m = numpy.max(i1) + 1
		i2m = numpy.max(i2) + 1
		idx = i1 + i1m*i2

		if self.dim == 3:
			idx += i1m*i3*i2m

		vfin, ifin = numpy.unique1d(idx, return_inverse=True)
		ifin = ifin.reshape(self._velocity_map.shape[1:])

		for j, v in enumerate(vfin):
			if j == 0:
				continue

			at = v / i1m
			a1 = v % i1m
			a2 = at % i2m
			a3 = at / i2m

			midx = (ifin == j)
			ret.extend((v1[a1], v2[a2]))

			if self.dim == 3:
				ret.append(v3[a3])

			self.map[midx] = self.map[midx] + i
			i += 1

		self._num_velocities = i
		i -= 1

		pressure, ifin = numpy.unique1d(self._pressure_map, return_inverse=True)
		ifin = ifin.reshape(self._pressure_map.shape)

		for j, v in enumerate(pressure):
			if j == 0:
				continue

			midx = (ifin == j)
			ret.append(v)
			self.map[midx] = self.map[midx] + i
			i += 1

		self._num_pressures = i + 1 - self._num_velocities
		self._params = ret
		return ret

	# FIXME: This method implicitly assumes that the object can be enclosed in a box
	# which does not intersect any other objects or boundaries.
	def add_force_object(self, obj_id, location, size):
		"""Scan the box defined by location and size for nodes and links that cross
		the fluid-solid interface. This function should be called from :meth:`define_nodes`.

		:param obj_id: object ID (any hashable)
		:param location: n-tuple specifying the top corner of the box to scan
		:param size: n-tuple specifying the dimensions of the box to scan
		"""
		self._force_nodes[obj_id] = []

		if self.dim == 2:
			mask = numpy.ones_like(self.map)
			mask[location[1]:location[1]+size[1],location[0]:location[0]+size[0]] = 0

			for i, vec in enumerate(self.sim.grid.basis):
				a = numpy.roll(self.map, -vec[0], axis=1)
				a = numpy.roll(a, -vec[1], axis=0)

				b = numpy.logical_and((self.map == self.NODE_FLUID), (a == self.NODE_WALL))
				b = numpy.ma.masked_array(b, mask)

				c = numpy.roll(b, vec[0], axis=1)
				c = numpy.roll(c, vec[1], axis=0)

				self._force_nodes[obj_id].append((numpy.nonzero(b), numpy.nonzero(c)))
		else:
			mask = numpy.ones_like(self.map)
			mask[location[2]:location[2]+size[2],location[1]:location[1]+size[1],location[0]:location[0]+size[0]] = 0

			for i, vec in enumerate(self.sim.grid.basis):
				a = numpy.roll(self.map, -vec[0], axis=2)
				a = numpy.roll(a, -vec[1], axis=1)
				a = numpy.roll(a, -vec[2], axis=0)

				b = numpy.logical_and((self.map == self.NODE_FLUID), (a == self.NODE_WALL))
				b = numpy.ma.masked_array(b, mask)

				c = numpy.roll(b, vec[0], axis=2)
				c = numpy.roll(c, vec[1], axis=1)
				c = numpy.roll(c, vec[2], axis=0)

				# For each diretion, save a map of source (fluid, b array) nodes and
				# target (solid, c array) nodes.
				self._force_nodes[obj_id].append((numpy.nonzero(b), numpy.nonzero(c)))

	def force(self, obj_id, dist):
		"""Calculate the force the fluid exerts on a solid object.

		The force is calculated for a time t - \Delta t / 2, given distributions
		at time t.

		To illustrate how the force is calculated, consider the following simplified
		case of momemntum transfer across the boundary (|)::

		  t = 0    <-y-- S --x->  |  <-a-- F -b-->
		  t = 1    <-a'- S --z->  |  <-c-- F -x'->

		Primes denote quantities after relaxation.  The amount of momentum transferred
		from the fluid node (F) to the solid node (S) is equal to a' - x'.

		:param obj_id: object ID
		:param dist: the distribution array for the current time step

		:rtype: force exterted on the selected object (a n-vector)
		"""
		force = numpy.float64([0.0] * self.dim)

		for dir, (fluid_map, solid_map) in enumerate(self._force_nodes[obj_id]):
			force += numpy.float64(list(self.sim.grid.basis[dir])) * (
					numpy.sum(dist[dir][solid_map]) +
					numpy.sum(dist[self.sim.grid.idx_opposite[dir]][fluid_map]))
		return force

class LBMGeo2D(LBMGeo):

	dim = 2

	def __init__(self, shape, *args, **kwargs):
		self.lat_w, self.lat_h = shape
		LBMGeo.__init__(self, shape, *args, **kwargs)

	def _get_map(self, location):
		return self.map[location[1], location[0]]

	def _set_map(self, location, value):
		self.map[location[1], location[0]] = value

	def get_defines(self):
		return {'geo_fluid': self.NODE_FLUID,
				'geo_wall': self.NODE_WALL,
				'geo_wall_e': self.NODE_DIR_E,
				'geo_wall_w': self.NODE_DIR_W,
				'geo_wall_s': self.NODE_DIR_S,
				'geo_wall_n': self.NODE_DIR_N,
				'geo_wall_ne': self.NODE_DIR_NE,
				'geo_wall_se': self.NODE_DIR_SE,
				'geo_wall_nw': self.NODE_DIR_NW,
				'geo_wall_sw': self.NODE_DIR_SW,
				'geo_dir_other': self.NODE_DIR_OTHER,
				'geo_bcv': self.NODE_VELOCITY,
				'geo_bcp': self.NODE_PRESSURE + self._num_velocities - 1,
				'geo_orientation_mask': self.NODE_ORIENTATION_MASK,
				'geo_orientation_shift': self.NODE_ORIENTATION_SHIFT,
				}

	def _postprocess_nodes(self, nodes=None):
		lat_w, lat_h = self.shape

		if nodes is None:
			nodes_ = ((x, y) for x in range(0, lat_w) for y in range(0, lat_h))
		else:
			nodes_ = nodes

		for x, y in nodes_:
			if self.map[y][x] != self.NODE_FLUID:
				# If the bool corresponding to a specific direction is True, the
				# distributions in this direction are undefined.
				north = y < lat_h-1 and self.map[y+1][x] == self.NODE_FLUID
				south = y > 0 and self.map[y-1][x] == self.NODE_FLUID
				west  = x > 0 and self.map[y][x-1] == self.NODE_FLUID
				east  = x < lat_w-1 and self.map[y][x+1] == self.NODE_FLUID

				# Walls aligned with the grid.
				if north and not west and not east:
					self.map[y][x] = self._encode_node(self.NODE_DIR_N, self.map[y][x])
				elif south and not west and not east:
					self.map[y][x] = self._encode_node(self.NODE_DIR_S, self.map[y][x])
				elif west and not south and not north:
					self.map[y][x] = self._encode_node(self.NODE_DIR_W, self.map[y][x])
				elif east and not south and not north:
					self.map[y][x] = self._encode_node(self.NODE_DIR_E, self.map[y][x])
				# Corners.
				elif y > 0 and x > 0 and self.map[y-1][x-1] == self.NODE_FLUID:
					self.map[y][x] = self._encode_node(self.NODE_DIR_SW, self.map[y][x])
				elif y > 0 and x < lat_w-1 and self.map[y-1][x+1] == self.NODE_FLUID:
					self.map[y][x] = self._encode_node(self.NODE_DIR_SE, self.map[y][x])
				elif y < lat_h-1 and x > 0 and self.map[y+1][x-1] == self.NODE_FLUID:
					self.map[y][x] = self._encode_node(self.NODE_DIR_NW, self.map[y][x])
				elif y < lat_h-1 and x < lat_w-1 and self.map[y+1][x+1] == self.NODE_FLUID:
					self.map[y][x] = self._encode_node(self.NODE_DIR_NE, self.map[y][x])
				else:
					self.map[y][x] = self._encode_node(self.NODE_DIR_OTHER, self.map[y][x])

class LBMGeo3D(LBMGeo):

	dim = 3

	def __init__(self, shape, *args, **kwargs):
		self.lat_w, self.lat_h, self.lat_d = shape
		LBMGeo.__init__(self, shape, *args, **kwargs)

	def _get_map(self, location):
		return self.map[location[2], location[1], location[0]]

	def _set_map(self, location, value):
		self.map[location[2], location[1], location[0]] = value

	def get_defines(self):
		return {'geo_fluid': self.NODE_FLUID,
				'geo_wall': self.NODE_WALL,
				'geo_bcv': self.NODE_VELOCITY,
				'geo_bcp': self.NODE_PRESSURE + self._num_velocities - 1,
				'geo_orientation_mask': self.NODE_ORIENTATION_MASK,
				'geo_orientation_shift': self.NODE_ORIENTATION_SHIFT,
				'geo_dir_other': self.NODE_DIR_OTHER,
				}

	def _postprocess_nodes(self, nodes=None):
		lat_w, lat_h, lat_d = self.shape

		# FIXME: Eventually, we will need to postprocess nodes in 3D grids as well.
		if nodes is None:
			# Postprocess the whole domain here.
			orientation = numpy.empty(shape=self.map.shape, dtype=numpy.int32)
			orientation[:,:,:] = self.NODE_DIR_OTHER
			self.map = self._encode_node(orientation, self.map)
		else:
			nodes_ = nodes

			for loc in nodes_:
				cnode_type = self._get_map(loc)

				if cnode_type != self.NODE_FLUID:
					self._set_map(loc, self._encode_node(self.NODE_DIR_OTHER, cnode_type))

#
# Boundary conditions
#
class LBMBC(object):
	"""Generic boundary condition class."""
	def __init__(self, name, supported_types=set(LBMGeo.NODE_TYPES), dims=set([2,3]), midgrid=False, wet_nodes=False):
		"""
		:param name: a string representing the boundary condition
		:param midgrid: if ``True``, the location of the boundary condition in the real
			domain between the grid nodes
		:param wet_nodes: if ``True``, the boundary condition nodes represent fluid particles
		    and undergo standard collisions
		"""
		self.name = name
		self.midgrid = midgrid
		self.wet_nodes = wet_nodes
		self.supported_types = supported_types
		self.supported_dims = dims

	def supports_dim(self, dim):
		return dim in self.supported_dims

def get_bc(type_):
	return BCS_MAP[type_]

SUPPORTED_BCS = [LBMBC('fullbb', midgrid=True, supported_types=set([LBMGeo.NODE_WALL, LBMGeo.NODE_VELOCITY])),
				 LBMBC('equilibrium', midgrid=False, supported_types=set([LBMGeo.NODE_VELOCITY, LBMGeo.NODE_PRESSURE])),
				 LBMBC('zouhe', midgrid=False, wet_nodes=True, supported_types=set([LBMGeo.NODE_WALL, LBMGeo.NODE_VELOCITY, LBMGeo.NODE_PRESSURE]), dims=set([2]))
				 ]

BCS_MAP = dict((x.name, x) for x in SUPPORTED_BCS)


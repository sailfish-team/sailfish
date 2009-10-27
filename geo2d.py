import numpy
import sym
import sympy

# Abstract class implementation, from Peter's Norvig site.
def abstract():
	import inspect
	caller = inspect.getouterframes(inspect.currentframe())[1][3]
	raise NotImplementedError(caller + ' must be implemented in subclass')

#
# Boundary conditions
#
class LBMBC(object):
	def __init__(self, name, midgrid=False, local=True):
		self.name = name
		self.midgrid = midgrid
		self.local = local

SUPPORTED_BCS = [LBMBC('fullbb', midgrid=True),
				 LBMBC('halfbb', midgrid=True),
				 LBMBC('zouhe', midgrid=False)
				 ]
BCS_MAP = dict((x.name, x) for x in SUPPORTED_BCS)

class LBMGeo(object):
	"""Abstract class for the LBM geometry."""

	NODE_FLUID = 0
	NODE_WALL = 1
	NODE_VELOCITY = 2
	NODE_PRESSURE = 3

	# Constants to specify node orientation.  This needs to match the order
	# in sym.basis.
	NODE_WALL_E = 0
	NODE_WALL_N = 1
	NODE_WALL_W = 2
	NODE_WALL_S = 3
	NODE_WALL_NE = 4
	NODE_WALL_NW = 5
	NODE_WALL_SW = 6
	NODE_WALL_SE = 7

	NODE_TYPE_MASK = 0xfffffff8
	NODE_ORIENTATION_SHIFT = 3
	NODE_ORIENTATION_MASK = 0x7

	@classmethod
	def _encode_node(cls, orientation, type):
		return orientation | (type << cls.NODE_ORIENTATION_SHIFT)

	@classmethod
	def _decode_node(cls, code):
		return (code & cls.NODE_ORIENTATION_MASK,
			    (code & cls.NODE_TYPE_MASK) >> cls.NODE_ORIENTATION_SHIFT)

	@classmethod
	def map_to_node_type(cls, node_map):
			return ((node_map & cls.NODE_TYPE_MASK) >> cls.NODE_ORIENTATION_SHIFT)

	def __init__(self, lat_w, lat_h, model, options, float, backend):
		self.lat_w = lat_w
		self.lat_h = lat_h
		self.model = model
		self.options = options
		self.backend = backend
		self.map = numpy.zeros((lat_h, lat_w), numpy.int32)
		self.gpu_map = backend.alloc_buf(like=self.map)
		self._vel_map = {}
		self._pressure_map = {}
		self._define_nodes()
		self._postprocess_nodes()
		self.float = float

		# Cache for equilibrium distributions.  Sympy numerical evaluation
		# is expensive.
		self.feq_cache = {}

	def _define_nodes(self):
		"""Define the types of all nodes."""
		abstract

	def init_dist(self, dist):
		"""Initialize the particle distributions in the whole simulation domain.

		Subclasses need to override this method to provide initial conditions for
		the simulation.
		"""
		abstract

	def get_reynolds(self, viscosity):
		"""Return the Reynolds number for this geometry."""
		abstract

	def reset(self):
		"""Perform a full reset of the geometry."""
		self._vel_map = {}
		self._pressure_map = {}
		self._define_nodes()
		self._postprocess_nodes()
		self.get_params()

	def update_map(self):
		self.backend.to_buf(self.gpu_map, self.map)

	def set_geo(self, x, y, type, val=None, update=False):
		"""Set the type of a grid node.

		Args:
		  x, y: location of the node
		  type: type of the node, one of the LBMGeo.NODE_* constants
		  val: optional argument for the node, e.g. the value of velocity or pressure
		  update: whether to automatically update the geometry for the simulation
		"""
		self.map[y][x] = numpy.int32(type)

		if val is not None:
			if type == LBMGeo.NODE_VELOCITY:
				if len(val) == 2:
					self._vel_map.setdefault(val, []).append((x,y))
				else:
					raise ValueError('Invalid velocity specified')
			elif type == LBMGeo.NODE_PRESSURE:
				self._pressure_map.setdefault(val, []).append((x,y))

		if update:
			self._postprocess_nodes(nodes=[(x, y)])
			self.update_map()

	def mask_array_by_fluid(self, array):
		# FIXME
		mask = self.map == LBMGeo.NODE_WALL
		return numpy.ma.array(array, mask=mask)

	def velocity_to_dist(self, vx, vy, dist, x, y):
		"""Set the distributions at node (x,y) so that the fluid there has a
		specific velocity (vx,vy).
		"""
		if (vx, vy) not in self.feq_cache:
			vals = []
			eq_rho = 1.0
			eq_dist = sym.bgk_equilibrium()
			for i, (eqd, idx) in enumerate(eq_dist):
				vals.append(self.float(sympy.N(eqd,
						subs={sym.rho: eq_rho, sym.vx: vx, sym.vy: vy})))
			self.feq_cache[(vx, vy)] = vals

		for i, val in enumerate(self.feq_cache[(vx, vy)]):
			dist[i][y][x] = val

	def _postprocess_nodes(self, nodes=None):
		"""Detect types of wall nodes and mark them appropriately.

		Args:
		  nodes: optional iterable of locations to postprocess
		"""

		if nodes is None:
			nodes_ = ((x, y) for x in range(0, self.lat_w) for y in range(0, self.lat_h))
		else:
			nodes_ = nodes

		for x, y in nodes_:
			if self.map[y][x] != LBMGeo.NODE_FLUID:
				# If the bool corresponding to a specific direction is True, the
				# distributions in this direction are undefined.
				north = y < self.lat_h-1 and self.map[y+1][x] == LBMGeo.NODE_FLUID
				south = y > 0 and self.map[y-1][x] == LBMGeo.NODE_FLUID
				west  = x > 0 and self.map[y][x-1] == LBMGeo.NODE_FLUID
				east  = x < self.lat_w-1 and self.map[y][x+1] == LBMGeo.NODE_FLUID

				if north and not west and not east:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_N, self.map[y][x])
				elif south and not west and not east:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_S, self.map[y][x])
				elif west and not south and not north:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_W, self.map[y][x])
				elif east and not south and not north:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_E, self.map[y][x])
				elif y > 0 and x > 0 and self.map[y-1][x-1] == LBMGeo.NODE_FLUID:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_SW, self.map[y][x])
				elif y > 0 and x < self.lat_w-1 and self.map[y-1][x+1] == LBMGeo.NODE_FLUID:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_SE, self.map[y][x])
				elif y < self.lat_h-1 and x > 0 and self.map[y+1][x-1] == LBMGeo.NODE_FLUID:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_NW, self.map[y][x])
				elif y < self.lat_h-1 and x < self.lat_w-1 and self.map[y+1][x+1] == LBMGeo.NODE_FLUID:
					self.map[y][x] = self._encode_node(LBMGeo.NODE_WALL_NE, self.map[y][x])

	def get_params(self):
		ret = []
		i = 0
		for v, pos_list in self._vel_map.iteritems():
			ret.extend(v)
			for x, y in pos_list:
				orientation, type = self._decode_node(self.map[y][x])
				self.map[y][x] = self._encode_node(orientation, type + i)

			i += 1

		i -= 1

		for p, pos_list in self._pressure_map.iteritems():
			ret.append(p)
			for x, y in pos_list:
				orientation, type = self._decode_node(self.map[y][x])
				self.map[y][x] = self._encode_node(orientation, type + i)

			i += 1

		self.update_map()
		return ret

	def get_defines(self):
		return {'geo_fluid': LBMGeo.NODE_FLUID,
				'geo_wall': LBMGeo.NODE_WALL,
				'geo_wall_e': LBMGeo.NODE_WALL_E,
				'geo_wall_w': LBMGeo.NODE_WALL_W,
				'geo_wall_s': LBMGeo.NODE_WALL_S,
				'geo_wall_n': LBMGeo.NODE_WALL_N,
				'geo_wall_ne': LBMGeo.NODE_WALL_NE,
				'geo_wall_se': LBMGeo.NODE_WALL_SE,
				'geo_wall_nw': LBMGeo.NODE_WALL_NW,
				'geo_wall_sw': LBMGeo.NODE_WALL_SW,
				'geo_bcv': LBMGeo.NODE_VELOCITY,
				'geo_bcp': LBMGeo.NODE_PRESSURE + len(self._vel_map) - 1,
				'geo_orientation_mask': LBMGeo.NODE_ORIENTATION_MASK,
				'geo_orientation_shift': LBMGeo.NODE_ORIENTATION_SHIFT,
				}


	def get_bc(self):
		return BCS_MAP[self.options.boundary]



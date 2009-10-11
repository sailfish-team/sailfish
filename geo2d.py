import numpy

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
				 LBMBC('halfbb', midgrid=True)]
BCS_MAP = dict((x.name, x) for x in SUPPORTED_BCS)

class LBMGeo(object):
	"""Abstract class for the LBM geometry."""

	NODE_FLUID = 0
	NODE_WALL = 1
	NODE_VELOCITY = 6
	NODE_PRESSURE = 7

	# Internal constants.
	_NODE_WALL_E = 2
	_NODE_WALL_W = 3
	_NODE_WALL_N = 4
	_NODE_WALL_S = 5

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
		self.float = float

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

	def set_geo(self, x, y, type, val=None):
		"""Set the type of a grid node.

		Args:
		  x, y: location of the node
		  type: type of the node, one of the LBMGeo.NODE_* constants
		  val: optional argument for the node, e.g. the value of velocity or pressure
		"""
		self.map[y][x] = numpy.int32(type)

		if val is not None:
			if type == LBMGeo.NODE_VELOCITY and len(val) == 2:
				self._vel_map.setdefault(val, []).append((x,y))
			elif type == LBMGeo.NODE_PRESSURE:
				self._pressure_map.setdefault(val, []).append((x,y))

	def mask_array_by_fluid(self, array):
		mask = self.map == LBMGeo.NODE_WALL
		return numpy.ma.array(array, mask=mask)

	def velocity_to_dist(self, vx, vy, dist, x, y):
		"""Set the distributions at node (x,y) so that the fluid there has a
		specific velocity (vx,vy).
		"""
		cusq = -1.5 * (vx*vx + vy*vy)
		eq_rho = 1.0
		dist[0][y][x] = self.float((1.0 + cusq) * 4.0/9.0 * eq_rho)
		dist[4][y][x] = self.float((1.0 + cusq + 3.0*vy + 4.5*vy*vy) / 9.0 * eq_rho)
		dist[1][y][x] = self.float((1.0 + cusq + 3.0*vx + 4.5*vx*vx) / 9.0 * eq_rho)
		dist[3][y][x] = self.float((1.0 + cusq - 3.0*vy + 4.5*vy*vy) / 9.0 * eq_rho)
		dist[2][y][x] = self.float((1.0 + cusq - 3.0*vx + 4.5*vx*vx) / 9.0 * eq_rho)
		dist[7][y][x] = self.float((1.0 + cusq + 3.0*(vx+vy) + 4.5*(vx+vy)*(vx+vy)) / 36.0 * eq_rho)
		dist[5][y][x] = self.float((1.0 + cusq + 3.0*(vx-vy) + 4.5*(vx-vy)*(vx-vy)) / 36.0 * eq_rho)
		dist[6][y][x] = self.float((1.0 + cusq + 3.0*(-vx-vy) + 4.5*(vx+vy)*(vx+vy)) / 36.0 * eq_rho)
		dist[8][y][x] = self.float((1.0 + cusq + 3.0*(-vx+vy) + 4.5*(-vx+vy)*(-vx+vy)) / 36.0 * eq_rho)

	def _postprocess_nodes(self):
		"""Detect types of wall nodes and mark them appropriately."""

		for x in range(0, self.lat_w):
			for y in range(0, self.lat_h):
				if self.map[y][x] == LBMGeo.NODE_WALL:
					# If the bool corresponding to a specific direction is True, the
					# distributions in this direction are undefined.
					north = y < self.lat_h-1 and self.map[y+1][x] == LBMGeo.NODE_FLUID
					south = y > 0 and self.map[y-1][x] == LBMGeo.NODE_FLUID
					west  = x > 0 and self.map[y][x-1] == LBMGeo.NODE_FLUID
					east  = x < self.lat_w-1 and self.map[y][x+1] == LBMGeo.NODE_FLUID

					if north and not west and not east:
						self.map[y][x] = LBMGeo._NODE_WALL_N
					elif south and not west and not east:
						self.map[y][x] = LBMGeo._NODE_WALL_S
					elif west and not south and not north:
						self.map[y][x] = LBMGeo._NODE_WALL_W
					elif east and not south and not north:
						self.map[y][x] = LBMGeo._NODE_WALL_E

	def get_params(self):
		ret = []
		i = 0
		for v, pos_list in self._vel_map.iteritems():
			ret.extend(v)
			for x, y in pos_list:
				self.map[y][x] += i

			i += 1

		i -= 1

		for p, pos_list in self._pressure_map.iteritems():
			ret.append(p)
			for x, y in pos_list:
				self.map[y][x] += i
			i += 1

		self.update_map()
		return ret

	def get_defines(self):
		return {'geo_fluid': LBMGeo.NODE_FLUID,
				'geo_wall': LBMGeo.NODE_WALL,
				'geo_wall_e': LBMGeo._NODE_WALL_E,
				'geo_wall_w': LBMGeo._NODE_WALL_W,
				'geo_wall_s': LBMGeo._NODE_WALL_S,
				'geo_wall_n': LBMGeo._NODE_WALL_N,
				'geo_bcv': LBMGeo.NODE_VELOCITY,
				'geo_bcp': LBMGeo.NODE_PRESSURE + len(self._vel_map) - 1}

	def get_bc(self):
		return BCS_MAP[self.options.boundary]



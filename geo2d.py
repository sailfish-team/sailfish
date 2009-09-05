import numpy
import pycuda.driver as cuda

# Abstract class implementation, from Peter's Norvig site.
def abstract():
	import inspect
	caller = inspect.getouterframes(inspect.currentframe())[1][3]
	raise NotImplementedError(caller + ' must be implemented in subclass')


class LBMGeo(object):
	"""Abstract class for the LBM geometry."""

	NODE_FLUID = 0
	NODE_WALL = 1
	NODE_VELOCITY = 5
	NODE_PRESSURE = 6

	# Internal constants.
	_NODE_WALL_E = 1
	_NODE_WALL_W = 2
	_NODE_WALL_N = 3
	_NODE_WALL_S = 4

	def __init__(self, lat_w, lat_h, model, options, float):
		self.lat_w = lat_w
		self.lat_h = lat_h
		self.model = model
		self.options = options
		self.map = numpy.zeros((lat_h, lat_w), numpy.int32)
		self.gpu_map = cuda.mem_alloc(self.map.size * self.map.dtype.itemsize)
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
		cuda.memcpy_htod(self.gpu_map, self.map)

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
					if y < self.lat_h-1 and self.map[y+1][x] == LBMGeo.NODE_FLUID:
						self.map[y][x] = LBMGeo._NODE_WALL_N
					elif y > 0 and self.map[y-1][x] == LBMGeo.NODE_FLUID:
						self.map[y][x] = LBMGeo._NODE_WALL_S
					elif x > 0 and self.map[y][x-1] == LBMGeo.NODE_FLUID:
						self.map[y][x] = LBMGeo._NODE_WALL_W
					elif x < self.lat_w-1 and self.map[x+1][y] == LBMGeo._NODE_WALL_E:
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
		return ('#define GEO_FLUID %d\n'
				'#define GEO_WALL_E %d\n#define GEO_WALL_W %d\n'
				'#define GEO_WALL_N %d\n#define GEO_WALL_S %d\n'
				'#define GEO_BCV %d\n#define GEO_BCP %d\n' %
				(LBMGeo.NODE_FLUID,
				 LBMGeo._NODE_WALL_E, LBMGeo._NODE_WALL_W,
				 LBMGeo._NODE_WALL_N, LBMGeo._NODE_WALL_S,
				 LBMGeo.NODE_VELOCITY, LBMGeo.NODE_PRESSURE + len(self._vel_map) - 1))

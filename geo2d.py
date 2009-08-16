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
	NODE_VELOCITY = 2
	NODE_PRESSURE = 3

	def __init__(self, lat_w, lat_h, model, options):
		self.lat_w = lat_w
		self.lat_h = lat_h
		self.model = model
		self.options = options
		self.map = numpy.zeros((lat_h, lat_w), numpy.int32)
		self.gpu_map = cuda.mem_alloc(self.map.size * self.map.dtype.itemsize)
		self._vel_map = {}
		self._pressure_map = {}
		self._reset()

	def update_map(self):
		cuda.memcpy_htod(self.gpu_map, self.map)

	def _reset(self): abstract

	def reset(self):
		self._vel_map = {}
		self._pressure_map = {}
		self._reset()
		self.get_params()

	def init_dist(self, dist): abstract

	def get_reynolds(self, viscosity):
		"""Returns the Reynolds number for this geometry."""
		abstract

	def set_geo(self, x, y, type, val=None):
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
		"""Set the distributions at node (x,y) so that the fluid there has a specific velocity (vx,vy)."""
		cusq = -1.5 * (vx*vx + vy*vy)
		eq_rho = 1.0
		dist[0][y][x] = numpy.float32((1.0 + cusq) * 4.0/9.0 * eq_rho)
		dist[4][y][x] = numpy.float32((1.0 + cusq + 3.0*vy + 4.5*vy*vy) / 9.0 * eq_rho)
		dist[1][y][x] = numpy.float32((1.0 + cusq + 3.0*vx + 4.5*vx*vx) / 9.0 * eq_rho)
		dist[3][y][x] = numpy.float32((1.0 + cusq - 3.0*vy + 4.5*vy*vy) / 9.0 * eq_rho)
		dist[2][y][x] = numpy.float32((1.0 + cusq - 3.0*vx + 4.5*vx*vx) / 9.0 * eq_rho)
		dist[7][y][x] = numpy.float32((1.0 + cusq + 3.0*(vx+vy) + 4.5*(vx+vy)*(vx+vy)) / 36.0 * eq_rho)
		dist[5][y][x] = numpy.float32((1.0 + cusq + 3.0*(vx-vy) + 4.5*(vx-vy)*(vx-vy)) / 36.0 * eq_rho)
		dist[6][y][x] = numpy.float32((1.0 + cusq + 3.0*(-vx-vy) + 4.5*(vx+vy)*(vx+vy)) / 36.0 * eq_rho)
		dist[8][y][x] = numpy.float32((1.0 + cusq + 3.0*(-vx+vy) + 4.5*(-vx+vy)*(-vx+vy)) / 36.0 * eq_rho)

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
		return ('#define GEO_FLUID %d\n#define GEO_WALL %d\n#define GEO_BCV %d\n#define GEO_BCP %d\n' %
				(LBMGeo.NODE_FLUID, LBMGeo.NODE_WALL, LBMGeo.NODE_VELOCITY, LBMGeo.NODE_PRESSURE + len(self._vel_map) - 1))

from operator import itemgetter
import math
import sympy
from sympy import Matrix, Rational, Symbol, Poly
import re

#set_main(sys.modules[__name__])

#
# Classes for different grid types.
#
# These classes are not supposed to be instantiated -- they serve as containers for
# information about the grid.
#

class DxQy(object):
	rho = Symbol('rho')
	vx = Symbol('vx')
	vy = Symbol('vy')
	vz = Symbol('vz')
	mx = Symbol('mx')
	my = Symbol('my')
	mz = Symbol('mz')
	visc = Symbol('visc')

	# Square of the sound velocity.
	# TODO: Before we can ever start using different values of the sound speed,
	# make sure that the sound speed is not hardcoded in the formulas below.
	cssq = Rational(1,3)

	@classmethod
	def model_supported(cls, model):
		if model == 'mrt':
			return hasattr(cls, 'mrt_matrix')
		# The D3Q13 grid only supports MRT.
		elif model == 'bgk':
			return (cls.Q != 13 or cls.dim != 3)
		else:
			return True

class D2Q9(DxQy):
	dim = 2
	Q = 9

	# Discretized velocities.
	basis = map(lambda x: Matrix((x,)),
				[(0,0), (1,0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])

	# BGK weights.
	weights = map(lambda x: Rational(*x),
			[(4,9), (1,9), (1,9), (1,9), (1,9), (1,36), (1,36), (1,36), (1,36)])

	# Names of the moments.
	mrt_names = ['rho', 'en', 'ens', 'mx', 'ex', 'my', 'ey', 'pxx', 'pxy']

	# The factor 0 is used for conserved moments.
	# en -- bulk viscosity
	# sd -- shear viscosity
	mrt_collision = [0, 1.63, 1.14, 0, 1.9, 0, 1.9, -1, -1]

	@classmethod
	def _init_mrt_basis(cls):
		cls.mrt_basis = map(lambda x: Matrix(x), [[1]*9,
				[x.dot(x) for x in cls.basis],
				[(x.dot(x))**2 for x in cls.basis],
				[x[0] for x in cls.basis],
				[x[0] * x.dot(x) for x in cls.basis],
				[x[1] for x in cls.basis],
				[x[1] * x.dot(x) for x in cls.basis],
				[x[0]*x[0] - x[1]*x[1] for x in cls.basis],
				[x[0]*x[1] for x in cls.basis]])

	@classmethod
	def _init_mrt_equilibrium(cls):
		cls.mrt_equilibrium = []

		c1 = -2

		# Name -> index map.
		n2i = {}
		for i, name in enumerate(cls.mrt_names):
			n2i[name] = i

		cls.mrt_collision[n2i['pxx']] = 1 / (0.5 + cls.visc * Rational(12, 2-c1))
		cls.mrt_collision[n2i['pxy']] = cls.mrt_collision[n2i['pxx']]

		vec_rho = cls.mrt_matrix[n2i['rho'],:]
		vec_mx = cls.mrt_matrix[n2i['mx'],:]
		vec_my = cls.mrt_matrix[n2i['my'],:]

		# We choose the form of the equilibrium distributions and the
		# optimal parameters as shows in the PhysRevE.61.6546 paper about
		# MRT in 2D.
		for i, name in enumerate(cls.mrt_names):
			if cls.mrt_collision[i] == 0:
				cls.mrt_equilibrium.append(0)
				continue

			vec_e = cls.mrt_matrix[i,:]
			if name == 'en':
				t = (Rational(1, vec_e.dot(vec_e)) *
						(-8*vec_rho.dot(vec_rho)*cls.rho +
							18*(vec_mx.dot(vec_mx)*cls.mx**2 + vec_my.dot(vec_my)*cls.my**2)))
			elif name == 'ens':
				# The 4 and -18 below are freely adjustable parameters.
				t = (Rational(1, vec_e.dot(vec_e)) *
						(4*vec_rho.dot(vec_rho)*cls.rho +
							-18*(vec_mx.dot(vec_mx)*cls.mx**2 + vec_my.dot(vec_my)*cls.my**2)))
			elif name == 'ex':
				t = Rational(1, vec_e.dot(vec_e)) * (c1 * vec_mx.dot(vec_mx)*cls.mx)
			elif name == 'ey':
				t = Rational(1, vec_e.dot(vec_e)) * (c1 * vec_my.dot(vec_my)*cls.my)
			elif name == 'pxx':
				t = (Rational(1, vec_e.dot(vec_e)) * Rational(2, 3) *
						(vec_mx.dot(vec_mx)*cls.mx**2 - vec_my.dot(vec_my)*cls.my**2))
			elif name == 'pxy':
				t = (Rational(1, vec_e.dot(vec_e)) * Rational(2, 3) *
						(math.sqrt(vec_mx.dot(vec_mx) * vec_my.dot(vec_my)) * cls.mx * cls.my))

			t = poly_factorize(t)
			t = expand_powers(str(t))
			cls.mrt_equilibrium.append(t)

class D3Q13(DxQy):
	dim = 3
	Q = 13

	basis = map(lambda x: Matrix((x, )),
				[(0,0,0), (1,1,0), (1,-1,0), (1,0,1), (1,0,-1), (0,1,1), (0,1,-1),
				 (-1,-1,0), (-1,1,0), (-1,0,-1), (-1,0,1), (0,-1,-1), (0,-1,1)])

	weights = map(lambda x: Rational(*x),
			[(1,2), (1,24), (1,24), (1,24), (1,24), (1,24), (1,24),
			 (1,24), (1,24), (1,24), (1,24), (1,24), (1,24)])

	mrt_names = ['rho', 'en', 'mx', 'my', 'mz',
				 'pww', 'pxx', 'pxy', 'pyz', 'pzx', 'm3x', 'm3y', 'm3z']

	# This choice of relaxation rates should make the simulation stable
	# for viscosities in the range [0.00255, 0.125] and flow speeds up to 0.1.
	mrt_collision = [0, 1.5, 0, 0, 0, -1, -1, -1, -1, -1, 1.8, 1.8, 1.8]

	@classmethod
	def _init_mrt_basis(cls):
		cls.mrt_basis = map(lambda x: Matrix(x), [
			[1]*13,
			[13*x.dot(x)/2 - 12 for x in cls.basis],
			[x[0] for x in cls.basis],
			[x[1] for x in cls.basis],
			[x[2] for x in cls.basis],
			[x[1]*x[1] - x[2]*x[2] for x in cls.basis],
			[3*x[0]*x[0] - x.dot(x) for x in cls.basis],
			[x[0]*x[1] for x in cls.basis],
			[x[1]*x[2] for x in cls.basis],
			[x[0]*x[2] for x in cls.basis],
			[(x[1]*x[1] - x[2]*x[2])*x[0] for x in cls.basis],
			[(x[2]*x[2] - x[0]*x[0])*x[1] for x in cls.basis],
			[(x[0]*x[0] - x[1]*x[1])*x[2] for x in cls.basis]])

	@classmethod
	def _init_mrt_equilibrium(cls):
		cls.mrt_equilibrium = []

		# Name -> index map.
		n2i = {}
		for i, name in enumerate(cls.mrt_names):
			n2i[name] = i

		cls.mrt_collision[n2i['pxx']] = 2 / (8 * cls.visc + 1)
		cls.mrt_collision[n2i['pww']] = cls.mrt_collision[n2i['pxx']]
		cls.mrt_collision[n2i['pxy']] = 2 / (4 * cls.visc + 1)
		cls.mrt_collision[n2i['pyz']] = cls.mrt_collision[n2i['pxy']]
		cls.mrt_collision[n2i['pzx']] = cls.mrt_collision[n2i['pxy']]

		vec_rho = cls.mrt_matrix[n2i['rho'],:]
		vec_mx = cls.mrt_matrix[n2i['mx'],:]
		vec_my = cls.mrt_matrix[n2i['my'],:]
		vec_my = cls.mrt_matrix[n2i['mz'],:]

		# Form of the equilibrium functions follows that from
		# PhysRevE.63.066702.
		for i, name in enumerate(cls.mrt_names):
			if cls.mrt_collision[i] == 0:
				cls.mrt_equilibrium.append(0)
				continue

			vec_e = cls.mrt_matrix[i,:]

			if name == 'm3x' or name == 'm3y' or name == 'm3z':
				t = 0
			elif name == 'pxy':
				t = 1/cls.rho * (cls.mx * cls.my)
			elif name == 'pyz':
				t = 1/cls.rho * (cls.my * cls.mz)
			elif name == 'pzx':
				t = 1/cls.rho * (cls.mx * cls.mz)
			elif name == 'pxx':
				# times rho_0 = 1
				t = 1/cls.rho * (2 * cls.mx**2 - cls.my**2 - cls.mz**2)
			elif name == 'pww':
				# times rho_0 = 1
				t = 1/cls.rho * (cls.my**2 - cls.mz**2)
			elif name == 'en':
				t = 3*cls.rho*(13*cls.cssq - 8)/2 + 13/(2 * cls.rho)*(cls.mx**2 + cls.my**2 + cls.mz**2)

			t = expand_powers(str(t))
			cls.mrt_equilibrium.append(t)


class D3Q15(DxQy):
	dim = 3
	Q = 15

	basis = map(lambda x: Matrix((x, )),
				[(0,0,0), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
				 (1,1,1), (-1,1,1), (1,-1,1), (-1,-1,1),
				 (1,1,-1), (-1,1,-1), (1,-1,-1), (-1,-1,-1)])

	weights = map(lambda x: Rational(*x),
			[(2,9), (1,9), (1,9), (1,9), (1,9), (1,9), (1,9),
				(1,72), (1,72), (1,72), (1,72), (1,72), (1,72), (1,72), (1,72)])

	mrt_names = ['rho', 'en', 'ens', 'mx', 'ex', 'my', 'ey', 'mz', 'ez',
				 'pww', 'pxx', 'pxy', 'pyz', 'pzx', 'mxyz']

	mrt_collision = [0.0, 1.6, 1.2, 0.0, 1.6, 0.0, 1.6, 0.0, 1.6,
				-1, -1, -1, -1, -1, 1.2]

	@classmethod
	def _init_mrt_basis(cls):
		cls.mrt_basis = map(lambda x: Matrix(x), [
			[1]*15,
			[x.dot(x) for x in cls.basis],
			[(x.dot(x))**2 for x in cls.basis],
			[x[0] for x in cls.basis],
			[x[0] * x.dot(x) for x in cls.basis],
			[x[1] for x in cls.basis],
			[x[1] * x.dot(x) for x in cls.basis],
			[x[2] for x in cls.basis],
			[x[2] * x.dot(x) for x in cls.basis],
			[x[1]*x[1] - x[2]*x[2] for x in cls.basis],
			[x[0]*x[0] - x[1]*x[1] for x in cls.basis],
			[x[0]*x[1] for x in cls.basis],
			[x[1]*x[2] for x in cls.basis],
			[x[0]*x[2] for x in cls.basis],
			[x[0]*x[1]*x[2] for x in cls.basis]])

	@classmethod
	def _init_mrt_equilibrium(cls):
		cls.mrt_equilibrium = []

		# Name -> index map.
		n2i = {}
		for i, name in enumerate(cls.mrt_names):
			n2i[name] = i

		cls.mrt_collision[n2i['pxx']] = 1 / (0.5 + 3*cls.visc)
		cls.mrt_collision[n2i['pww']] = cls.mrt_collision[n2i['pxx']]
		cls.mrt_collision[n2i['pxy']] = cls.mrt_collision[n2i['pxx']]
		cls.mrt_collision[n2i['pyz']] = cls.mrt_collision[n2i['pxx']]
		cls.mrt_collision[n2i['pzx']] = cls.mrt_collision[n2i['pxx']]

		vec_rho = cls.mrt_matrix[n2i['rho'],:]
		vec_mx = cls.mrt_matrix[n2i['mx'],:]
		vec_my = cls.mrt_matrix[n2i['my'],:]

		# Form of the equilibrium functions follows that from
		# dHumieres, PhilTranA, 2002.
		for i, name in enumerate(cls.mrt_names):
			if cls.mrt_collision[i] == 0:
				cls.mrt_equilibrium.append(0)
				continue

			vec_e = cls.mrt_matrix[i,:]
			if name == 'en':
				t = -cls.rho + 1/cls.rho * (cls.mx**2 + cls.my**2 + cls.mz**2)
			elif name == 'ens':
				t = -cls.rho
			elif name == 'ex':
				t = -Rational(7,3)*cls.mx
			elif name == 'ey':
				t = -Rational(7,3)*cls.my
			elif name == 'ez':
				t = -Rational(7,3)*cls.mz
			elif name == 'pxx':
				t = 1/(3*cls.rho) * (2*cls.mx**2 - (cls.my**2 + cls.mz**2))
			elif name == 'pww':
				t = 1/cls.rho * (cls.my**2 - cls.mz**2)
			elif name == 'pxy':
				t = 1/cls.rho * (cls.mx * cls.my)
			elif name == 'pyz':
				t = 1/cls.rho * (cls.my * cls.mz)
			elif name == 'pzx':
				t = 1/cls.rho * (cls.mx * cls.mz)
			elif name == 'myz':
				t = 0

#			t = poly_factorize(t)
			t = expand_powers(str(t))
			cls.mrt_equilibrium.append(t)


class D3Q19(DxQy):
	dim = 3
	Q = 19

	basis = map(lambda x: Matrix((x, )),
				[(0,0,0),
				(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
				(1,1,0), (-1,1,0), (1,-1,0), (-1,-1,0),
				(1,0,1), (-1,0,1), (1,0,-1), (-1,0,-1),
				(0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1)])

	weights = map(lambda x: Rational(*x),
			[(1,3), (1,18), (1,18), (1,18), (1,18), (1,18), (1,18),
				(1,36), (1,36), (1,36), (1,36), (1,36), (1,36),
				(1,36), (1,36), (1,36), (1,36), (1,36), (1,36)])

	mrt_names = ['rho', 'en', 'ens', 'mx', 'ex', 'my', 'ey', 'mz', 'ez',
				 'pww', 'piww', 'pxx', 'pixx', 'pxy', 'pyz', 'pzx', 'm3x', 'm3y', 'm3z']

	mrt_collision = [0.0, 1.19, 1.24, 0.0, 1.2, 0.0, 1.2, 0.0, 1.2,
				-1, 1.4, -1, 1.4, -1, -1, -1, 1.98, 1.98, 1.98]

	@classmethod
	def _init_mrt_basis(cls):
		cls.mrt_basis = map(lambda x: Matrix(x), [
			[1]*19,
			[x.dot(x) for x in cls.basis],
			[(x.dot(x))**2 for x in cls.basis],
			[x[0] for x in cls.basis],
			[x[0] * x.dot(x) for x in cls.basis],
			[x[1] for x in cls.basis],
			[x[1] * x.dot(x) for x in cls.basis],
			[x[2] for x in cls.basis],
			[x[2] * x.dot(x) for x in cls.basis],
			[x[1]*x[1] - x[2]*x[2] for x in cls.basis],
			[x.dot(x) * (x[1]*x[1] - x[2]*x[2]) for x in cls.basis],
			[x[0]*x[0] - x[1]*x[1] for x in cls.basis],
			[x.dot(x) * (x[0]*x[0] - x[1]*x[1]) for x in cls.basis],
			[x[0]*x[1] for x in cls.basis],
			[x[1]*x[2] for x in cls.basis],
			[x[0]*x[2] for x in cls.basis],
			[(x[1]*x[1] - x[2]*x[2])*x[0] for x in cls.basis],
			[(x[2]*x[2] - x[0]*x[0])*x[1] for x in cls.basis],
			[(x[0]*x[0] - x[1]*x[1])*x[2] for x in cls.basis]])

	@classmethod
	def _init_mrt_equilibrium(cls):
		cls.mrt_equilibrium = []

		# Name -> index map.
		n2i = {}
		for i, name in enumerate(cls.mrt_names):
			n2i[name] = i

		cls.mrt_collision[n2i['pxx']] = 1 / (0.5 + 3*cls.visc)
		cls.mrt_collision[n2i['pww']] = cls.mrt_collision[n2i['pxx']]
		cls.mrt_collision[n2i['pxy']] = cls.mrt_collision[n2i['pxx']]
		cls.mrt_collision[n2i['pyz']] = cls.mrt_collision[n2i['pxx']]
		cls.mrt_collision[n2i['pzx']] = cls.mrt_collision[n2i['pxx']]

		vec_rho = cls.mrt_matrix[n2i['rho'],:]
		vec_mx = cls.mrt_matrix[n2i['mx'],:]
		vec_my = cls.mrt_matrix[n2i['my'],:]

		# Form of the equilibrium functions follows that from
		# dHumieres, PhilTranA, 2002.
		for i, name in enumerate(cls.mrt_names):
			if cls.mrt_collision[i] == 0:
				cls.mrt_equilibrium.append(0)
				continue

			vec_e = cls.mrt_matrix[i,:]
			if name == 'en':
				t = -11 * cls.rho + 19/cls.rho * (cls.mx**2 + cls.my**2 + cls.mz**2)
			elif name == 'ens':
				t = - Rational(475,63)/cls.rho*(cls.mx**2 + cls.my**2 + cls.mz**2)
			elif name == 'ex':
				t = -Rational(2,3)*cls.mx
			elif name == 'ey':
				t = -Rational(2,3)*cls.my
			elif name == 'ez':
				t = -Rational(2,3)*cls.mz
			elif name == 'pxx':
				t = 1/(3*cls.rho) * (2*cls.mx**2 - (cls.my**2 + cls.mz**2))
			elif name == 'pww':
				t = 1/cls.rho * (cls.my**2 - cls.mz**2)
			elif name == 'pxy':
				t = 1/cls.rho * (cls.mx * cls.my)
			elif name == 'pyz':
				t = 1/cls.rho * (cls.my * cls.mz)
			elif name == 'pzx':
				t = 1/cls.rho * (cls.mx * cls.mz)
			elif name == 'm3x' or name == 'm3y' or name == 'm3z' or name == 'pixx' or name == 'piww':
				t = 0

#			t = poly_factorize(t)
			t = expand_powers(str(t))
			cls.mrt_equilibrium.append(t)


def bgk_equilibrium(as_string=True):
	"""Get expressions for the BGK equilibrium distribution.

	Args:
	  as_string: if True, this function will return strings, if False,
	             sympy expressions

	Returns:
	  a list of strings or sympy epxressions representing the equilibrium
	  distribution functions
	"""
	out = []

	for i, ei in enumerate(GRID.basis):
		t = (GRID.rho * GRID.weights[i] *
					(1 + poly_factorize(
						3*ei.dot(GRID.v) +
						Rational(9, 2) * (ei.dot(GRID.v))**2 -
						Rational(3, 2) * GRID.v.dot(GRID.v))))
		if as_string:
			t = expand_powers(str(t))

		out.append((t, GRID.idx_name[i]))

	return out

def eval_bgk_equilibrium(velocity, rho):
	"""Get BGK equilibrium distributions for a specific velocity and density.

	Args:
	  velocity: a n-tuple of velocity components
	  rho: density

	Returns:
	  a list of values of the distributions (in the same order as the basis
	  vectors for the current grid)
	"""
	vals = []

	subs={GRID.rho: rho}

	for i, v in enumerate(velocity):
		subs[GRID.v[i]] = v

	for i, (eqd, idx) in enumerate(EQ_DIST):
		vals.append(sympy.N(eqd, subs=subs))

	return vals

def bgk_external_force():
	"""Get expressions for the external body force correction in the BGK model.

	Returns:
	  a list of sympy expressions (in the same order as the current grid's basis)
	"""
	eax = Symbol('eax')
	eay = Symbol('eay')
	eaz = Symbol('eaz')
	pref = Symbol('pref')

	if GRID.dim == 2:
		ea = Matrix(([eax, eay],))
	else:
		ea = Matrix(([eax, eay, eaz],))

	ret = []

	for i, ei in enumerate(GRID.basis):
		t = expand_powers(str(pref * GRID.weights[i] *
			poly_factorize( (ei - GRID.v + ei.dot(GRID.v)*ei*3).dot(ea) )))
		ret.append((t, GRID.idx_name[i]))

	return ret

def bgk_external_force_pref():
	# This includes a factor of c_s^2.
	return 'rho * (3.0f - 3.0f/(2.0f * tau))'

def bb_swap_pairs():
	"""Get a set of indices which have to be swapped for a full bounce-back."""
	ret = set()

	for i, j in enumerate(GRID.idx_opposite):
		# Nothing to swap with.
		if i == j:
			continue

		ret.add(min(i,j))

	return ret

def fill_missing_dists(distp, missing_dir):
	syms = [Symbol('%s->%s' % (distp, x)) for x in GRID.idx_name]
	ret = []

	for i, sym in enumerate(syms):
		sp = GRID.basis[i].dot(GRID.basis[missing_dir+1])

		if sp < 0:
			ret.append((syms[GRID.idx_opposite[i]], sym))

	return ret

def ex_rho(distp, missing_dir=None, rho=None):
	"""Express density as a function of the distibutions.

	Args:
	  distp: name of the pointer to the distribution structure

	Returns:
	  a sympy expression for the density
	"""
	syms = [Symbol('%s->%s' % (distp, x)) for x in GRID.idx_name]
	ret = 0

	if missing_dir is None:
		for sym in syms:
			ret += sym
		return ret

	srho = Symbol(rho)
	return srho / (GRID.basis[missing_dir+1].dot(GRID.v) + 1)

def ex_velocity(distp, comp, rho, momentum=False, missing_dir=None, par_rho=None):
	"""Express velocity as a function of the distributions.

	Args:
	  distp: name of the pointer to the distribution structure
	  comp: velocity component number: 0, 1 or 2 (for 3D lattices)
	  rho: name of the density variable

	Returns:
	  a sympy expression for the velocity in a given direction
	"""
	syms = [Symbol('%s->%s' % (distp, x)) for x in GRID.idx_name]
	srho = Symbol(rho)
	ret = 0

	if missing_dir is None:
		for i, sym in enumerate(syms):
			ret += GRID.basis[i][comp] * sym

		if momentum:
			return ret
		else:
			return ret / srho

	prho = Symbol(par_rho)

	for i, sym in enumerate(syms):
		sp = GRID.basis[i].dot(GRID.basis[missing_dir+1])
		if sp <= 0:
			ret = 1

	ret = ret * (srho - prho)
	ret *= -GRID.basis[missing_dir+1][comp]
	if momentum:
		return ret
	else:
		return ret / prho


def bgk_to_mrt(bgk_dist, mrt_dist):
	bgk_syms = Matrix([Symbol('%s->%s' % (bgk_dist, x)) for x in GRID.idx_name])
	mrt_syms = [Symbol('%s.%s' % (mrt_dist, x)) for x in GRID.mrt_names]

	ret = []

	for i, rhs in enumerate(GRID.mrt_matrix * bgk_syms):
		ret.append((mrt_syms[i], rhs))

	return ret

def mrt_to_bgk(bgk_dist, mrt_dist):
	bgk_syms = [Symbol('%s->%s' % (bgk_dist, x)) for x in GRID.idx_name]
	mrt_syms = Matrix([Symbol('%s.%s' % (mrt_dist, x)) for x in GRID.mrt_names])

	ret = []

	for i, rhs in enumerate(GRID.mrt_matrix.inv() * mrt_syms):
		ret.append((bgk_syms[i], rhs))
	return ret

def _get_known_dists(normal):
	unknown = []
	known = []

	for i, vec in enumerate(GRID.basis):
		if normal.dot(vec) > 0:
			unknown.append(i)
		else:
			known.append(i)

	return known, unknown

def zouhe_bb(orientation):
	idx = orientation + 1
	normal = GRID.basis[idx]
	known, unknown = _get_known_dists(normal)
	ret = []

	eq = bgk_equilibrium(as_string=False)

	# Bounce-back of the non-equilibrium parts.
	for i in unknown:
		oi = GRID.idx_opposite[i]
		ret.append((Symbol('fi->%s' % GRID.idx_name[i]),
					Symbol('fi->%s' % GRID.idx_name[oi]) - eq[oi][0] + eq[i][0]))

	for i in range(0, len(ret)):
		t = poly_factorize(ret[i][1])
		t = expand_powers(str(t))

		ret[i] = (ret[i][0], t)

	return ret

def zouhe_fixup(orientation):
	idx = orientation + 1
	normal = GRID.basis[idx]
	known, unknown = _get_known_dists(normal)

	unknown_not_normal = set(unknown)
	unknown_not_normal.remove(idx)

	ret = []

	# Momentum differences.
	mdiff = [Symbol('nvx'), Symbol('nvy')]
	if GRID.dim == 3:
		mdiff.append(Symbol('nvz'))
		basis = [Matrix([1,0,0]), Matrix([0,1,0]), Matrix([0,0,1])]
	else:
		basis = [Matrix([1,0]), Matrix([0,1])]

	# Scale by number of adjustable distributions.
	for i, md in enumerate(mdiff):
		if basis[i].dot(normal) != 0:
			mdiff[i] = 0
			continue

		cnt = 0
		for idir in unknown_not_normal:
			if GRID.basis[idir].dot(basis[i]) != 0:
				cnt += 1

		ret.append((md, md / cnt))

	# Adjust distributions to conserve momentum in directions other than
	# the wall normal.
	for i in unknown_not_normal:
		val = 0
		ei = GRID.basis[i]
		if ei[0] != 0 and mdiff[0] != 0:
			val += mdiff[0] * ei[0]
		if ei[1] != 0 and mdiff[1] != 0:
			val += mdiff[1] * ei[1]
		if GRID.dim == 3 and ei[2] != 0 and mdiff[2] != 0:
			val += mdiff[2] * ei[2]

		if val != 0:
			csym = Symbol('fi->%s' % GRID.idx_name[i])
			ret.append((csym, csym + val))

	return ret

def zouhe_velocity(orientation):
	# TODO: Add some code to factor out the common factors in the
	# expressions returned by this function.
	idx = orientation + 1
	normal = GRID.basis[idx]
	known, unknown = _get_known_dists(normal)

	# First, compute an expression for the density.
	vrho = 0
	for didx in known:
		if GRID.basis[didx].dot(normal) == -1:
			vrho += 2 * Symbol('fi->%s' % GRID.idx_name[didx])
		else:
			vrho += Symbol('fi->%s' % GRID.idx_name[didx])
	vrho /= (1 - GRID.v.dot(normal))

	ret = []
	ret.append((Symbol('rho'), vrho))

	# Bounce-back of the non-equilibrium part of the distributions
	# in the direction of the normal vector.
	oidx = GRID.idx_opposite[idx]
	sym_norm = Symbol('fi->%s' % GRID.idx_name[idx])
	sym_opp  = Symbol('fi->%s' % GRID.idx_name[oidx])

	val_norm = sympy.solve(bgk_equilibrium(as_string=False)[idx][0] - sym_norm -
					  bgk_equilibrium(as_string=False)[oidx][0] + sym_opp, sym_norm)[0]

	ret.append((sym_norm, poly_factorize(val_norm)))

	# Compute expressions for the remaining distributions.
	remaining = [Symbol('fi->%s' % GRID.idx_name[x]) for x in unknown if x != idx]

	vxe = ex_velocity('fi', 0, 'rho')
	vye = ex_velocity('fi', 1, 'rho')

	# Substitute the distribution calculated from the bounce-back procedure above.
	vx2 = vxe.subs({sym_norm: val_norm})
	vy2 = vye.subs({sym_norm: val_norm})

	for sym, val in sympy.solve((GRID.vx - vx2, GRID.vy - vy2), *remaining).iteritems():
		ret.append((sym, poly_factorize(val)))

	return ret

def get_prop_dists(dir):
	"""Compute a list of base vectors with a specific value of the X component (`dir`)."""
	ret = []

	for i, ei in enumerate(GRID.basis):
		if ei[0] == dir and i > 0:
			ret.append((i, GRID.idx_name[i]))

	return ret

#
# Sympy stuff.
#

def poly_factorize(poly):
	"""Factorize multivariate polynomials into a sum of products of monomials.

	This function can be used to decompose polynomials into a form which
	minimizes the number of additions and multiplications, and which thus
	can be evaluated efficently."""
	max_deg = {}

	if not isinstance(poly, Poly):
		poly = Poly(sympy.expand(poly), *poly.atoms(Symbol))

	denom, poly = poly.as_integer()

	# Determine the order of factorization.  We proceed through the
	# symbols, starting with the one present in the highest order
	# in the polynomial.
	for i, sym in enumerate(poly.symbols):
		max_deg[i] = 0

	for monom in poly.monoms:
		for i, symvar in enumerate(monom):
			max_deg[i] = max(max_deg[i], symvar)

	ret_poly = 0
	monoms = list(poly.monoms)

	for isym, maxdeg in sorted(max_deg.items(), key=itemgetter(1), reverse=True):
		drop_idx = []
		new_poly = []

		for i, monom in enumerate(monoms):
			if monom[isym] > 0:
				drop_idx.append(i)
				new_poly.append((poly.coeff(*monom), monom))

		if not new_poly:
			continue

		ret_poly += sympy.factor(Poly(new_poly, *poly.symbols))

		for idx in reversed(drop_idx):
			del monoms[idx]

	# Add any remaining O(1) terms.
	new_poly = []
	for i, monom in enumerate(monoms):
		new_poly.append((poly.coeff(*monom), monom))

	if new_poly:
		ret_poly += Poly(new_poly, *poly.symbols)

	return ret_poly / denom

def expand_powers(t):
	# FIXME: This should work for powers other than 2.
	return re.sub('([a-z]+)\*\*2', '\\1*\\1', t)

def use_pointers(str):
	ret = str.replace('rho', ' *rho')
	ret = ret.replace('vx', 'v[0]')
	ret = ret.replace('vy', 'v[1]')
	ret = ret.replace('vz', 'v[2]')
	return ret

def make_float(t):
	return re.sub(r'([0-9]+\.[0-9]*)', r'\1f', str(t))

def _gcd(a,b):
	while b:
		a, b = b, a % b
	return a

def gcd(*terms):
	return reduce(lambda a,b: _gcd(a,b), terms)

def orthogonalize(*vectors):
	"""Ortogonalize a set of vectors.

	Given a set of vectors orthogonalize them using the GramSchmidt procedure.
	The vectors are then simplified (common factors are removed to keep their
	norm small).

	Args:
	  vectors: a collection of vectors to orthogonalize

	Returns:
	  orthogonalized vectors
	"""
	ret = []
	for x in sympy.GramSchmidt(vectors):
		fact = 1
		for z in x:
			if isinstance(z, Rational) and z.q != 1 and fact % z.q != 0:
				fact = fact * z.q

		x = x * fact
		cd = abs(gcd(*x))
		if cd > 1:
			x /= cd

		ret.append(x)
	return ret

def _prepare_grids():
	"""Decorate grid classes with useful info computable from the basic definitions
	of the grids.

	This approach saves the programmer's time and automatically ensures correctness
	of the computed values."""

	for grid in KNOWN_GRIDS:
		if len(grid.basis) != len(grid.weights):
			raise TypeError('Grid %s is ill-defined: not all BGK weights have been specified.' % grid.__name__)

		if len(grid.basis) != grid.Q:
			raise TypeError('Grid {0} has an ill-defined Q factor.'.format(grid.__name__))

		if sum(grid.weights) != 1:
			raise TypeError('BGK weights for grid %s do not sum up to unity.' % grid.__name__)

		grid.idx_name = []
		grid.idx_opposite = []

		if grid.dim == 2:
			names = [{-1: 'S', 1: 'N', 0: ''},
					{-1: 'W', 1: 'E', 0: ''}]
			grid.v = Matrix(([grid.vx, grid.vy],))
		else:
			names = [{-1: 'B', 1: 'T', 0: ''},
					 {-1: 'S', 1: 'N', 0: ''},
					 {-1: 'W', 1: 'E', 0: ''}]
			grid.v = Matrix(([grid.vx, grid.vy, grid.vz],))

		for ei in grid.basis:
			# Compute direction names.
			name = 'f'
			for i, comp in enumerate(reversed(ei.tolist()[0])):
				name += names[i][int(comp)]

			if name == 'f':
				name += 'C'

			grid.idx_name.append(name)

			# Find opposite directions.
			for j, ej in enumerate(grid.basis):
				if ej == -1 * ei:
					grid.idx_opposite.append(j)
					break
			else:
				raise TypeError('Opposite vector for %s not found.' % ei)

		# If MRT is supported for the current grid, compute the transformation
		# matrix from the velocity space to moment space.  The procedure is as
		# follows:
		#  - _init_mrt_basis computes the moment vectors
		#  - the moment vectors are orthogonalized using the Gram-Schmidt procedure
		#  - the othogonal vectors form the transformation matrix
		#  - the equilibrium expressions are computed and saved
		if hasattr(grid, '_init_mrt_basis'):
			grid._init_mrt_basis()

			if len(grid.mrt_basis) != len(grid.basis):
				raise TypeError('The number of moment vectors for grid %s is different '
					'than the number of vectors in velocity space.' % grid.__name__)

			if len(grid.mrt_basis) != len(grid.mrt_names):
				raise TypeError('The number of MRT names for grid %s is different '
					'than the number of moments.' % grid.__name__)

			grid.mrt_matrix = Matrix([x.transpose().tolist()[0] for x in orthogonalize(*grid.mrt_basis)])
			grid._init_mrt_equilibrium()

def use_grid(grid):
	"""Select the current grid.

	Args:
	  grid: one of the grid classes listed in sym.KNOWN_GRIDS.
	"""
	global GRID, EQ_DIST
	if grid not in KNOWN_GRIDS:
		raise ValueError('Unknown grid type "%s"' % grid)

	GRID = grid
	EQ_DIST = bgk_equilibrium()

KNOWN_GRIDS = (D2Q9, D3Q13, D3Q15, D3Q19)

_prepare_grids()
use_grid(D2Q9)

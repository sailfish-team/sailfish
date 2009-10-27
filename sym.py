from operator import itemgetter
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

class D2Q9(DxQy):
	dim = 2

	# Discretized velocities.
	basis = map(lambda x: Matrix((x,)),
				[(0,0), (1,0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])

	# BGK weights.
	weights = map(lambda x: Rational(*x),
			[(4,9), (1,9), (1,9), (1,9), (1,9), (1,36), (1,36), (1,36), (1,36)])

# TODO: Finish this.
class D3Q13(DxQy):
	dim = 3
	basis = map(lambda x: Matrix((x, )),
				[(0,0,0), (1,1,0), (-1,-1,0), (1,-1,0), (-1,1,0), (1,0,1),
				 (-1,0,-1), (1,0,-1), (-1,0,1), (0,1,1), (0,-1,-1), (0,1,-1), (0,-1,1)])

class D3Q15(DxQy):
	dim = 3
	basis = map(lambda x: Matrix((x, )),
				[(0,0,0), (1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (0,0,1), (0,0,-1),
				 (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1), (-1,1,1), (-1,1,-1), (-1,-1,1),
				 (-1,-1,-1)])

	weights = map(lambda x: Rational(*x),
			[(2,9), (1,9), (1,9), (1,9), (1,9), (1,9), (1,9),
				(1,72), (1,72), (1,72), (1,72), (1,72), (1,72), (1,72), (1,72)])


class D3Q19(DxQy):
	dim = 3
	basis = map(lambda x: Matrix((x, )),
				[(0,0,0),
				(1,0,0), (0,1,0), (-1,0,0), (0,-1,0), (0,0,1), (0,0,-1),
				(1,1,0), (-1,1,0), (-1,-1,0), (1,-1,0),
				 (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1),
				 (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1)])

	weights = map(lambda x: Rational(*x),
			[(1,3), (1,18), (1,18), (1,18), (1,18), (1,18), (1,18),
				(1,36), (1,36), (1,36), (1,36), (1,36), (1,36),
				(1,36), (1,36), (1,36), (1,36), (1,36), (1,36)])

KNOWN_GRIDS = [D2Q9, D3Q15, D3Q19]

def _prepare_grids():
	"""Decorate grid classes with useful info computable from the basic definitions
	of the grids.

	This approach saves the programmer's time and automatically ensures correctness
	of the computed values."""

	for grid in KNOWN_GRIDS:
		if len(grid.basis) != len(grid.weights):
			raise TypeError('Grid %s is ill-defined: not all BGK weights have been specified.' % grid.__name__)

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

			# Find opposite direction.
			for j, ej in enumerate(grid.basis):
				if ej == -1 * ei:
					grid.idx_opposite.append(j)
					break
			else:
				raise TypeError('Opposite vector for %s not found.' % ei)

_prepare_grids()

# The grid type used in the simulation.  One of the classes defined above.
GRID = D2Q9

def use_grid(grid):
	if grid not in KNOWN_GRIDS:
		raise ValueError('Unknown grid type "%s"' % grid)

	GRID = grid

def bgk_equilibrium(as_string=True):
	"""Get expressions for the BGK equilibrium distribution.

	Returns:
	   a list of strings representing the equilibrium distribution functions
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

def bgk_external_force():
	eax = Symbol('eax')
	eay = Symbol('eay')
	pref = Symbol('pref')
	ea = Matrix(([eax, eay],))
	ret = []

	for i, ei in enumerate(basis):
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

def ex_rho(distp):
	syms = [Symbol('%s->%s' % (distp, x)) for x in GRID.idx_name]
	ret = 0

	for sym in syms:
		ret += sym

	return ret

def ex_velocity(distp, comp, rho):
	syms = [Symbol('%s->%s' % (distp, x)) for x in GRID.idx_name]
	srho = Symbol(rho)
	ret = 0

	for i, sym in enumerate(syms):
		 ret += GRID.basis[i][comp] * sym

	return ret / srho

def _get_known_dists(normal):
	unknown = []
	known = []

	for i, vec in enumerate(GRID.basis):
		if normal.dot(vec) == 1:
			unknown.append(i)
		else:
			known.append(i)

	return known, unknown

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

	val_norm = solve(bgk_equilibrium(as_string=False)[idx][0] - sym_norm -
					  bgk_equilibrium(as_string=False)[oidx][0] + sym_opp, sym_norm)[0]

	ret.append((sym_norm, poly_factorize(val_norm)))

	# Compute expressions for the remaining distributions.
	remaining = [Symbol('fi->%s' % GRID.idx_name[x]) for x in unknown if x != idx]

	vxe = ex_velocity('fi', 0, 'rho')
	vye = ex_velocity('fi', 1, 'rho')

	# Substitute the distribution calculated from the bounce-back procedure above.
	vx2 = vxe.subs({sym_norm: val_norm})
	vy2 = vye.subs({sym_norm: val_norm})

	for sym, val in solve((vx - vx2, vy - vy2), *remaining).iteritems():
		ret.append((sym, poly_factorize(val)))

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
	ret = ret.replace('vx', ' *vx')
	return ret.replace('vy', ' *vy')



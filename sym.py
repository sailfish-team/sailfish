from operator import itemgetter
from sympy import *
import re

#set_main(sys.modules[__name__])

# Discretized velocities.
basis = map(lambda x: Matrix((x,)),
			[(0,0), (1,0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])

# BGK weights.
weights = map(lambda x: Rational(*x),
		[(4, 9), (1, 9), (1, 9), (1, 9), (1, 9), (1, 36), (1, 36), (1, 36), (1, 36)])

# Names of the distributions to use in the code.
idx_name = ['fC', 'fE', 'fN', 'fW', 'fS', 'fNE', 'fNW', 'fSW', 'fSE']
idx_opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]

rho = Symbol('rho')
vx = Symbol('vx')
vy = Symbol('vy')

v = Matrix(([vx, vy],))

def poly_factorize(poly):
	"""Factorize multivariate polynomials into a sum of products of monomials.

	This function can be used to decompose polynomials into a form which
	minimizes the number of additions and multiplications, and which thus
	can be evaluated efficently."""
	max_deg = {}

	if not isinstance(poly, Poly):
		poly = Poly(expand(poly), *poly.atoms(Symbol))

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

		ret_poly += factor(Poly(new_poly, *poly.symbols))

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

def bgk_equilibrium(as_string=True):
	"""Get expressions for the BGK equilibrium distribution.

	Returns:
	   a list of strings representing the equilibrium distribution functions
	"""
	out = []

	for i, ei in enumerate(basis):
		t = (rho * weights[i] *
					(1 + poly_factorize(
						3*ei.dot(v) +
						Rational(9, 2) * (ei.dot(v))**2 -
						Rational(3, 2) * v.dot(v))))
		if as_string:
			t = expand_powers(str(t))

		out.append((t, idx_name[i]))

	return out

def bgk_external_force():
	eax = Symbol('eax')
	eay = Symbol('eay')
	pref = Symbol('pref')
	ea = Matrix(([eax, eay],))
	ret = []

	for i, ei in enumerate(basis):
		t = expand_powers(str(pref * weights[i] *
			poly_factorize( (ei - v + ei.dot(v)*ei*3).dot(ea) )))
		ret.append((t, idx_name[i]))

	return ret

def bgk_external_force_pref():
	# This includes a factor of c_s^2.
	return 'rho * (3.0f - 3.0f/(2.0f * tau))'

def bb_swap_pairs():
	"""Get a set of indices which have to be swapped for a full bounce-back."""
	ret = set()

	for i, j in enumerate(idx_opposite):
		# Nothing to swap with.
		if i == j:
			continue

		ret.add(min(i,j))

	return ret

def ex_rho(distp):
	syms = [Symbol('%s->%s' % (distp, x)) for x in idx_name]
	ret = 0

	for sym in syms:
		ret += sym

	return ret

def ex_velocity(distp, dim, rho):
	syms = [Symbol('%s->%s' % (distp, x)) for x in idx_name]
	srho = Symbol(rho)
	ret = 0

	for i, sym in enumerate(syms):
		 ret += basis[i][dim] * sym

	return ret / srho

def _get_known_dists(normal):
	unknown = []
	known = []

	for i, vec in enumerate(basis):
		if normal.dot(vec) == 1:
			unknown.append(i)
		else:
			known.append(i)

	return known, unknown

def zouhe_velocity(orientation):
	# TODO: Add some code to factor out the common factors in the
	# expressions returned by this function.
	idx = orientation + 1
	normal = basis[idx]
	known, unknown = _get_known_dists(normal)

	# First, compute an expression for the density.
	vrho = 0
	for didx in known:
		if basis[didx].dot(normal) == -1:
			vrho += 2 * Symbol('fi->%s' % idx_name[didx])
		else:
			vrho += Symbol('fi->%s' % idx_name[didx])
	vrho /= (1 - v.dot(normal))

	ret = []
	ret.append((Symbol('rho'), vrho))

	# Bounce-back of the non-equilibrium part of the distributions
	# in the direction of the normal vector.
	oidx = idx_opposite[idx]
	sym_norm = Symbol('fi->%s' % idx_name[idx])
	sym_opp  = Symbol('fi->%s' % idx_name[oidx])

	val_norm = solve(bgk_equilibrium(as_string=False)[idx][0] - sym_norm -
					  bgk_equilibrium(as_string=False)[oidx][0] + sym_opp, sym_norm)[0]

	ret.append((sym_norm, poly_factorize(val_norm)))

	# Compute expressions for the remaining distributions.
	remaining = [Symbol('fi->%s' % idx_name[x]) for x in unknown if x != idx]

	vxe = ex_velocity('fi', 0, 'rho')
	vye = ex_velocity('fi', 1, 'rho')

	# Substitute the distribution calculated from the bounce-back procedure above.
	vx2 = vxe.subs({sym_norm: val_norm})
	vy2 = vye.subs({sym_norm: val_norm})

	for sym, val in solve((vx - vx2, vy - vy2), *remaining).iteritems():
		ret.append((sym, poly_factorize(val)))

	return ret

def use_pointers(str):
	ret = str.replace('rho', ' *rho')
	ret = ret.replace('vx', ' *vx')
	return ret.replace('vy', ' *vy')



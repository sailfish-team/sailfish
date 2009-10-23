from operator import itemgetter
from sympy import *
import re

#set_main(sys.modules[__name__])

basis = map(lambda x: Matrix((x,)),
			[(0,0), (1,0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, 1), (-1, -1), (1, -1)])

weights = map(lambda x: Rational(*x),
		[(4, 9), (1, 9), (1, 9), (1, 9), (1, 9), (1, 36), (1, 36), (1, 36), (1, 36)])

idx_name = ['fC', 'fE', 'fN', 'fW', 'fS', 'fNE', 'fNW', 'fSW', 'fSE']
idx_opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]

rho = Symbol('rho')
vx = Symbol('vx')
vy = Symbol('vy')
v = Matrix(([vx, vy],))
csq = Rational(1, 3)  # fluid velocity

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
	 return re.sub('([a-z]+)\*\*2', '\\1*\\1', t)

def get_bgk_collision():
	out = []

	for i, ei in enumerate(basis):
		t = expand_powers(str(
				rho * weights[i] *
					(1 + poly_factorize(
						3*ei.dot(v) +
						Rational(9, 2) * (ei.dot(v))**2 -
						Rational(3, 2) * v.dot(v)))))
		out.append((t, idx_name[i]))

	return out


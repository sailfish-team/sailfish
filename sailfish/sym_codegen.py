import operator
import re
try:
    from sympy.core import singleton
    NegativeOne = singleton.S.NegativeOne
except ImportError:
    from sympy.core import basic
    NegativeOne = basic.S.NegativeOne
from sympy.printing.ccode import CCodePrinter
from sympy.printing.precedence import precedence
from sympy import Symbol, Poly

import sympy
import numpy as np
#
# Sympy stuff.
#

def _truncate_add(add, order=2):
    ret = 0
    from sailfish.sym import S
    for mul in add.args:
        o = 0
        for a in mul.args:
            if a in (S.g0m1x, S.g0m1y, S.g0m1z):
                o += 1
            elif str(a) == 'vsq':
                o += 2
            elif type(a) is sympy.Pow:
                if a.base in (S.g0m1x, S.g0m1y, S.g0m1z):
                    o += a.exp
                elif str(a.base) == 'vsq':
                    o += a.exp * 2

        if o <= order:
            ret += mul
    return ret

def truncate_velocity(poly, order=2):
    """Truncates a polynomial to terms of desired order in velocity.

    :param poly: Poly object to truncate
    :param order: desired order
    """
    degs = []
    from sailfish.sym import S
    for x in poly.gens:
        if x in (S.g0m1x, S.g0m1y, S.g0m1z):
            degs.append(1)
        elif str(x) == 'vsq':
            degs.append(2)
        else:
            degs.append(0)

    gens = []

    for g in poly.gens:
        if type(g) is sympy.Pow and g.exp == -1:
            g = g.series(S.g0m1x, n=order+1).removeO()
            g = g.series(S.g0m1y, n=order+1).removeO()
            g = g.series(S.g0m1z, n=order+1).removeO()
            gens.append(g)
        else:
            gens.append(g)

    truncated = 0
    for powers, coeff in poly.terms():
        o = 0
        for power, deg in zip(powers, degs):
            o += deg * power

        if o > order:
            continue
        truncated += coeff * reduce(operator.mul, (
            x**p for p, x in zip(powers, gens)))

    truncated = truncated.series(S.g0m1x, n=order+1).removeO()
    truncated = truncated.series(S.g0m1y, n=order+1).removeO()
    truncated = truncated.series(S.g0m1z, n=order+1).removeO()

    if type(truncated) is sympy.Add:
        return _truncate_add(truncated.expand(), order)
    else:
        return truncated


def poly_factorize(poly):
    """Factorize multivariate polynomials into a sum of products of monomials.

    This function can be used to decompose polynomials into a form which
    minimizes the number of additions and multiplications, and which thus
    can be evaluated efficently."""
    max_deg = {}

    if 'horner' in dir(sympy):
        return sympy.horner(poly)

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

def _use_pointers(str):
    ret = re.sub(r'([^_a-z0-9A-Z]|^)rho', r'\1(*rho)', str)
    return ret

def _use_vectors(str):
    ret = str.replace('vx', 'v0[0]').replace('vy', 'v0[1]').replace('vz', 'v0[2]')
    for dist in range(1,9):
        ret = ret.replace('g%seax' % dist, 'ea%s[0]' % dist)
        ret = ret.replace('g%seay' % dist, 'ea%s[1]' % dist)
        ret = ret.replace('g%seaz' % dist, 'ea%s[2]' % dist)
    ret = ret.replace('eax', 'ea0[0]').replace('eay', 'ea0[1]').replace('eaz', 'ea0[2]')

    for dist in range(0, 9):
        ret = ret.replace('g%sd1m0x' % dist, 'grad%s[0]' % dist)
        ret = ret.replace('g%sd1m0y' % dist, 'grad%s[1]' % dist)
        ret = ret.replace('g%sd1m0z' % dist, 'grad%s[2]' % dist)

    return ret

def make_float(t):
    return re.sub(r'((^|[^a-zA-Z])[0-9]+\.[0-9]*(e(\+|-)[0-9]*)?)',
                  r'\1f', str(t))

def _int2float(t):
    return re.sub(r'([^eE]|^)([\(\+\-\*/ ]|^)([0-9]+)([\+\-\*/\) ]|$)',
                  r'\1\2\3.0\4', str(t))

class KernelCodePrinter(CCodePrinter):

    def _print_Pow(self, expr):
        PREC = precedence(expr)
        if expr.exp is NegativeOne:
            return '(1.0/%s)' % (self.parenthesize(expr.base, PREC))
        # For the kernel code, it's better to calculate the power
        # here explicitly by multiplication.
        elif expr.exp == 2:
            return '(%s*%s)' % (self.parenthesize(expr.base, PREC),
                              self.parenthesize(expr.base, PREC))
        else:
            return _int2float('powf(%s,%s)' % (self.parenthesize(expr.base, PREC),
                                               self.parenthesize(expr.exp, PREC)))

    def _print_Function(self, expr):
        if expr.func.__name__ in ('log', 'exp', 'sin', 'cos', 'tanh'):
            return '{0}f({1})'.format(expr.func.__name__,
                    self.stringify(expr.args, ', '))
        else:
            return super(KernelCodePrinter, self)._print_Function(expr)

    def _print_Rational(self, expr):
        p, q = int(expr.p), int(expr.q)
        return '%d.0/%d.0' % (p, q)

    # Replaces expressions: x / F, with x * (1.0 / F), where F is a floating
    # point or integer number. This allows the CUDA compiler to replace the
    # division operation by a multiplication when the code runs in double
    # precision.
    def _print_Mul(self, expr):
        s = super(KernelCodePrinter, self)._print_Mul(expr)
        return re.sub(r'/ ?([0-9\.]+)', r'* (1.0 / \1)', s)


def cexpr(sim, incompressible, pointers, ex, rho, aliases=True, vectors=True,
          phi=None, vel=None):
    """Convert a SymPy expression into a string containing valid C code.

    :param sim: the main simulation class (descendant of :class:`LBMSim`)
    :param incompressible: if ``True``, use the incompressible model
    :param pointers: if ``True``, macroscopic variables (density and velocities)
        will be converted to pointers in the output
    :param ex: the sympy expression to convert
    :param vectors: if ``True``, references to vector components (velocity,
        acceleration) will be replaced by references to C arrays
    :param rho: density symbol (sympy Symbol, string).  If ``None`` the
        standard rho symbol for the grid will be used.
    :param vel: symbol to use for velocity
    :rtype: string representing the C code
    """

    t = ex
    S = sim.S

    if type(t) is int or type(t) is float or isinstance(t, np.generic):
        t = '%.20e' % t
        return make_float(t)

    if type(rho) is str:
        rho = Symbol(rho)
        t = t.subs(S.rho, rho)
    if rho is None:
        rho = S.rho

    if type(phi) is str:
        phi = Symbol(phi)
        t = t.subs(S.phi, phi)

    if incompressible:
        t = t.subs(S.rho0, 1)
    else:
        t = t.subs(S.rho0, rho)

    if aliases:
        for src, dst in S.aliases.iteritems():
            t = t.subs(src, dst)

    t = KernelCodePrinter().doprint(t)
    if pointers:
        t = _use_pointers(t)
        t = _use_vectors(t)  # FIXME

    if vectors:
        t = _use_vectors(t)

    if vel:
        t = t.replace('v0', vel)

    t = make_float(_int2float(t))
    return t

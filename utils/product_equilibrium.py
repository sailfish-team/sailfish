"""Computes the product form of equilibrium minimizing the entropy function.

Works for arbitrary lattices supporting an entropy function.
"""

from sailfish import sym
from sympy import *
import operator

g = sym.D3Q15
As = []
Bxs = []
Bys = []
Bzs = []

for i in range(0, 9):
    if i % 2 == 0:
        As.append(Symbol('A%d' % i))
    else:
        As.append(0)

for i in range(0, 9):
    if i == 0:
        Bxs.append(1)
        Bys.append(1)
        Bzs.append(1)
    else:
        Bxs.append(Symbol('Bx%d' % i))
        Bys.append(Symbol('By%d' % i))
        Bzs.append(Symbol('Bz%d' % i))

k = Symbol('k')
rho = sym.S.rho
ux = k * sym.S.vx
uy = k * sym.S.vy
uz = k * sym.S.vz
v = (ux**2 + uy**2 + uz**2)**(Rational(1, 2))

# Series expansion of factors.
A = lambda o: reduce(operator.add, [As[i] * v**i for i in range(0, o + 1)], 0)
Bx = lambda o: reduce(operator.add, [Bxs[i] * ux**i for i in range(0, o + 1)], 0)
By = lambda o: reduce(operator.add, [Bys[i] * uy**i for i in range(0, o + 1)], 0)
Bz = lambda o: reduce(operator.add, [Bzs[i] * uz**i for i in range(0, o + 1)], 0)

# Equilibrium in product form.
F = lambda o, w, ei: w * rho * A(o) * Bx(o)**ei[0] * By(o)**ei[1] * Bz(o)**ei[2]

# Results.
res = {}

# Compute results to 8th order. High order terms are slow due to the large
# number of terms in the series expansion.
for o in range(0, 9):
    rho_ex = sum([F(o, w, ei) for w, ei in zip(g.weights, g.basis)]).subs(res)
    mx_ex = sum([F(o, w, ei) * ei[0] for w, ei in zip(g.weights, g.basis)]).subs(res)
    my_ex = sum([F(o, w, ei) * ei[1] for w, ei in zip(g.weights, g.basis)]).subs(res)
    mz_ex = sum([F(o, w, ei) * ei[2] for w, ei in zip(g.weights, g.basis)]).subs(res)

    args = []

    if type(As[o]) is Symbol:
        args.append(As[o])

    if o > 0:
        args.extend([Bxs[o], Bys[o], Bzs[o]])

    if o > 0:
        rho_ex = rho_ex.series(k, n=o+1).removeO() - rho_ex.series(k, n=o).removeO()
        mx_ex = mx_ex.series(k, n=o+1).removeO() - mx_ex.series(k, n=o).removeO()
        my_ex = my_ex.series(k, n=o+1).removeO() - my_ex.series(k, n=o).removeO()
        mz_ex = mz_ex.series(k, n=o+1).removeO() - mz_ex.series(k, n=o).removeO()

    if o == 1:
        mx_ex -= rho * sym.S.vx
        my_ex -= rho * sym.S.vy
        mz_ex -= rho * sym.S.vz
    elif o == 0:
        rho_ex -= rho

    rho_ex = simplify(rho_ex)
    mx_ex = simplify(mx_ex)
    my_ex = simplify(my_ex)
    mz_ex = simplify(mz_ex)

    print 'Solving o = %d' % o
    print ' *', rho_ex
    print ' *', mx_ex
    print ' *', my_ex
    print ' *', mz_ex
    print '.. for: ', args

    sol = solve((rho_ex, mx_ex, my_ex, mz_ex), *args)
    if sol:
        for symbol, h in sol.iteritems():
            res[symbol] = h.subs({k: 1})
    print sol
    print '---'

print res

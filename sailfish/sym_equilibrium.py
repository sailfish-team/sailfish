"""Symbolic expressions for various LB equilibria."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import namedtuple
import sympy
from sympy import Rational, Symbol, Eq
from sailfish.sym import poly_factorize, S

EqDef = namedtuple('EqDef', 'expr local_vars')

# Form of the equilibrium function taken from Phys Rev E 78, 056709.
def free_energy_binary_liquid_equilibrium(sim):
    grid = sim.grid
    if (grid.dim == 3 and grid.Q != 19) or (grid.dim == 2 and grid.Q != 9):
        raise TypeError('The binary liquid model requires the D2Q9 or D3Q19 grid.')

    S = sim.S
    pb = Symbol('pb')
    mu = Symbol('mu')

    # This is zero for grids with square of sound speed = 1/3.
    lambda_ = S.visc * (1 - grid.cssq * 3)

    out = []
    lvars = []
    lvars.append(Eq(pb, S.rho / 3 + S.A * (- (S.phi**2) / 2 + Rational(3,4) * S.phi**4)))
    lvars.append(Eq(mu, S.A * (-S.phi + S.phi**3) - S.kappa * S.g1d2m0))

    t_sum = 0

    for i, ei in enumerate(grid.basis[1:]):
        t = (S.wi[i] * (pb - S.kappa * S.phi * S.g1d2m0 + S.rho * ei.dot(grid.v) +
            Rational(3,2) * (
                (ei.dot(grid.v))**2 * S.rho + lambda_ *
                    (2 * ei.dot(grid.v) * ei.dot(S.grad0) + ei.dot(ei) * grid.v.dot(S.grad0))
                 - Rational(1,3) * (S.rho * grid.v.dot(grid.v) + lambda_ * 3 * grid.v.dot(S.grad0))
            )) +
            S.kappa * (S.wxx[i] * S.g1d1m0x**2 + S.wyy[i] * S.g1d1m0y**2 + S.wzz[i] * S.g1d1m0z**2 +
                       S.wyz[i] * S.g1d1m0y * S.g1d1m0z +
                       S.wxy[i] * S.g1d1m0x * S.g1d1m0y +
                       S.wxz[i] * S.g1d1m0x * S.g1d1m0z))

        t_sum += t
        out.append(t)

    # The first term is chosen so that rho is conserved.
    out1 = [sympy.simplify(S.rho - t_sum)] + out

    out = []
    t_sum = 0
    for i, ei in enumerate(grid.basis[1:]):
        t = S.wi[i] * (S.Gamma * mu + ei.dot(grid.v) * S.phi + Rational(3,2) * S.phi * (
                -Rational(1,3) * grid.v.dot(grid.v) + (ei.dot(grid.v))**2))

        t_sum += t
        out.append(t)

    # The first term is chosen so that the order parameter is conserved.
    out = [sympy.simplify(S.phi - t_sum)] + out
    return EqDef([out1, out], lvars)

def shallow_water_equilibrium(grid):
    """Get expressions for the BGK equilibrium distribution for the shallow
    water equation."""

    if grid.dim != 2 or grid.Q != 9:
        raise TypeError('Shallow water equation requires the D2Q9 grid.')

    out = [S.rho - grid.weights[0] * S.rho * (
        Rational(15, 8) * S.gravity * S.rho - 3 * grid.v.dot(grid.v))]

    for ei, weight in zip(grid.basis[1:], grid.weights[1:]):
        out.append(weight * (
                S.rho * poly_factorize(Rational(3,2) * S.rho * S.gravity + 3*ei.dot(grid.v) +
                    Rational(9,2) * (ei.dot(grid.v))**2 - Rational(3, 2) * grid.v.dot(grid.v))))

    return EqDef([out], local_vars=[])

def bgk_equilibrium(grid, rho=None, rho0=None):
    """Get expressions for the BGK equilibrium distribution.

    :param grid: the grid class to be used
    """
    out = []

    if rho is None:
        rho = S.rho

    if rho0 is None:
        rho0 = S.rho0

    for ei, weight in zip(grid.basis, grid.weights):
        out.append(weight * (rho + rho0 * poly_factorize(
            3 * ei.dot(grid.v) + Rational(9, 2) * (ei.dot(grid.v))**2 -
            Rational(3, 2) * grid.v.dot(grid.v))))

    return EqDef([out], local_vars=[])


def elbm_equilibrium(grid, rho=None):
    """
    Form of the equilibrium defined in Europhys. Lett., 63 (6) pp. 798-804
    (2003).
    """
    prefactor = Symbol('prefactor')
    coeff1 = Symbol('coeff1')
    coeff2 = Symbol('coeff2')

    if rho is None:
        rho = S.rho

    out = []
    lvars = []

    sqrt = sympy.sqrt
    v_scaled = grid.v / sqrt(grid.cssq)
    tmp = S.rho
    coeffs = []
    for v_comp in v_scaled:
        tmp *= (2 - sqrt(1 + v_comp**2))
        coeffs.append((2 / sqrt(3) * v_comp + sqrt(1 + v_comp**2)) /
                (1 - v_comp / sqrt(3)))

    lvars.append(Eq(prefactor, tmp))
    lvars.append(Eq(coeff1, coeffs[0]))
    lvars.append(Eq(coeff2, coeffs[1]))
    cfs = [coeff1, coeff2]

    for ei, weight in zip(grid.basis, grid.entropic_weights):
        t = prefactor * weight
        for j, comp in enumerate(ei):
            t *= cfs[j]**(comp / (sqrt(3) * sqrt(grid.cssq)))
        out.append(t)

    return EqDef([out], lvars)

def elbm_d3q15_equilibrium(grid, rho=None):
    """
    Form of equilibrium defined in PRL 97, 010201 (2006).
    """
    if rho is None:
        rho = S.rho

    prefactor = Symbol('prefactor')
    coeff1 = Symbol('coeff1')
    coeff2 = Symbol('coeff2')
    coeff3 = Symbol('coeff3')
    vsq = Symbol('vsq')
    vx, vy, vz = grid.v

    lvars = []
    lvars.append(Eq(vsq, grid.v.dot(grid.v)))
    lvars.append(Eq(prefactor, poly_factorize(
        rho * (1 - 3 * vsq / 2 + 9 * vsq**2 / 8 +
        Rational(27, 16) * (-vsq**3 + 2 * (vy**2 + vz**2) *
            (vsq * vx**2 + vy**2 * vz**2) +
            20 * vx**2 * vy**2 * vz**2) +
        81 * vsq**4 / 128 +
        Rational(81, 32) * (vx**8 + vy**8 + vz**8
            - 36 * vx**2 * vy**2 * vz**2 * vsq
            - vx**4 * vy**4
            - vy**4 * vz**4
            - vx**4 * vz**4)))))

    cfs = [coeff1, coeff2, coeff3]

    for i, coeff in enumerate(cfs):
        tmp = (1 + 3 * grid.v[i] + 9 * grid.v[i]**2 / 2 +
            9 * grid.v[i]**3 / 2 + 27 * grid.v[i]**4 / 8)

        # Cyclic permutation.
        x = i
        y = (i + 1) % 3
        z = (i + 2) % 3

        tmp += Rational(27, 8) * (grid.v[x]**5
                - 4 * grid.v[x] * grid.v[y]**2 * grid.v[z]**2)
        tmp += Rational(81, 16) * (grid.v[x]**6
                - 8 * grid.v[x]**2 * grid.v[y]**2 * grid.v[z]**2)
        tmp += Rational(81, 16) * (grid.v[x]**7
                - 10 * grid.v[x]**3 * grid.v[y]**2 * grid.v[z]**2
                + 2 * grid.v[x] * grid.v[y]**2 * grid.v[z]**2 * vsq)
        tmp += Rational(243, 128) * (grid.v[x]**8
                + 16 * grid.v[x]**2 * grid.v[y]**2 * grid.v[z]**2
                * (grid.v[y]**2 + grid.v[z]**2))

        lvars.append(Eq(coeff, poly_factorize(tmp)))

    out = []
    for ei, weight in zip(grid.basis, grid.entropic_weights):
        t = prefactor * weight
        for j, comp in enumerate(ei):
            t *= cfs[j]**comp
        out.append(t)

    return EqDef([out], lvars)

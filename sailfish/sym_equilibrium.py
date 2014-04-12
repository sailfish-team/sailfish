"""Symbolic expressions for various LB equilibria."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import namedtuple
import sympy
from sympy import Rational, Symbol, Eq
from sailfish.sym import S, D3Q15, D3Q19
from sailfish.sym_codegen import poly_factorize

EqDef = namedtuple('EqDef', 'expression local_vars')

def free_energy_equilibrium_fluid(grid, config):
    """Returns the equilibrium for the fluid field in the binary fluid
    free energy model.

    Form of the equilibrium function taken from Phys Rev E 78, 056709."""
    if (grid.dim == 3 and grid.Q != 19) or (grid.dim == 2 and grid.Q != 9):
        raise TypeError('The binary liquid model requires the D2Q9 or D3Q19 grid.')

    pb = Symbol('pb')

    # This is zero for grids with square of sound speed = 1/3.
    lambda_ = S.visc * (1 - grid.cssq * 3)

    out = []
    lvars = [Eq(pb, S.rho / 3 + S.A * (- (S.phi**2) / 2 + Rational(3,4) * S.phi**4))]
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
    out = [sympy.simplify(S.rho - t_sum)] + out
    return EqDef(out, lvars)

def free_energy_equilibrium_order_param(grid, config):
    """Returns the equilibrium for the order parameter field in the binary fluid
    free energy model. See free_energy_equilibrium_fluid for more
    information."""
    mu = Symbol('mu')
    lvars = [Eq(mu, S.A * (-S.phi + S.phi**3) - S.kappa * S.g1d2m0)]
    out = []
    t_sum = 0
    for i, ei in enumerate(grid.basis[1:]):
        t = S.wi[i] * (S.Gamma * mu + ei.dot(grid.v) * S.phi + Rational(3,2) * S.phi * (
                -Rational(1,3) * grid.v.dot(grid.v) + (ei.dot(grid.v))**2))

        t_sum += t
        out.append(t)

    # The first term is chosen so that the order parameter is conserved.
    out = [sympy.simplify(S.phi - t_sum)] + out
    return EqDef(out, lvars)

def shallow_water_equilibrium(grid, config):
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

    return EqDef(out, local_vars=[])

def bgk_equilibrium(grid, config, rho=None, rho_0=None, order=2):
    """Get expressions for the BGK equilibrium distribution.

    :param grid: the grid class to be used
    """
    out = []

    if rho is None:
        rho = S.rho

    if rho_0 is None:
        if config.incompressible:
            rho_0 = S.rho_0
        elif config.minimize_roundoff:
            rho_0 = rho + 1.0
        else:
            rho_0 = rho

    for ei, weight in zip(grid.basis, grid.weights):
        h = (ei.dot(grid.v) / grid.cssq +
             ei.dot(grid.v)**2 / grid.cssq**2 / 2 -
             grid.v.dot(grid.v) / 2 / grid.cssq)
        # Aidun, Clausen; Latice-Boltzmann Method for Complex Flows
        # Note: this does not seem to recover the 1st moment in the unit test!
#        if order > 2:
#            h += (ei.dot(grid.v)**3 / grid.cssq**3 / 2 -
#                  ei.dot(grid.v) * grid.v.dot(grid.v) / grid.cssq**2 / 2)

        out.append(weight * (rho + rho_0 * poly_factorize(h)))

    return EqDef(out, local_vars=[])


def elbm_equilibrium(grid):
    """
    Form of the equilibrium defined in Europhys. Lett., 63 (6) pp. 798-804
    (2003).
    """
    prefactor = Symbol('prefactor')
    coeff1 = Symbol('coeff1')
    coeff2 = Symbol('coeff2')
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

    return EqDef(out, lvars)

def elbm_d3q19_equilibrium(grid, order=8):
    rho = S.rho

    prefactor = Symbol('chi')
    coeff1 = Symbol('zeta_x')
    coeff2 = Symbol('zeta_y')
    coeff3 = Symbol('zeta_z')
    vsq = Symbol('vsq')
    vx, vy, vz = grid.v

    o = [(1 if i <= order else 0) for i in range(0, 9)]

    lvars = []
    lvars.append(Eq(vsq, grid.v.dot(grid.v)))
    lvars.append(Eq(prefactor, poly_factorize(
        rho * (1 -
               o[2] * Rational(3, 2) * vsq +
               o[4] * Rational(9, 8) * vsq**2 -
               o[6] * Rational(27, 16) * (vx**6 + vx**4 * (vy**2 + vz**2) +
                                          (vy**2 + vz**2) * (vy**4 + vz**4) +
                                          vx**2 * (vy**4 + 12 * vy**2 * vz**2 +
                                                   vz**4)) +
               o[8] * Rational(81, 128) * (5 * vx**8 + 5 * vy**8 + 4 * vy**6 *
                                           vz**2 + 2 * vy**4 * vz**4 +
                                           4 * vy**2 * vz**6 +
                                           5 * vz**8 + 4 * vx**6 * (vy**2 + vz**2) +
                                           4 * vx**2 * (vy**2 + vz**2) * (
                                               vy**4 + 17 * vy**2 * vz**2 + vz**4) +
                                           2 * vx**4 * (vy**4 + 36 * vy**2 *
                                                        vz**2 + vz**4))))))

    cfs = [coeff1, coeff2, coeff3]

    for i, coeff in enumerate(cfs):
        tmp = (1 +
               o[1] * 3 * grid.v[i] +
               o[2] * Rational(9, 2) * grid.v[i]**2 +
               o[3] * Rational(9, 2) * grid.v[i]**3 +
               o[4] * Rational(27, 8) * grid.v[i]**4)

        # Cyclic permutation.
        x = i
        y = (i + 1) % 3
        z = (i + 2) % 3

        tmp += o[5] * Rational(27, 8) * (
            grid.v[x]**5 + 2 * grid.v[x] * grid.v[y]**2 * grid.v[z]**2)
        tmp += o[6] * Rational(81, 16) * (
            grid.v[x]**6 + 4 * grid.v[x]**2 * grid.v[y]**2 * grid.v[z]**2)
        tmp += o[7] * Rational(81, 16) * (
            grid.v[x] * (grid.v[x]**6 + 4 * grid.v[x]**2 * grid.v[y]**2 * grid.v[z]**2 -
                         grid.v[y]**2 * grid.v[z]**2 * (grid.v[y]**2 +
                                                        grid.v[z]**2)))
        tmp += o[8] * Rational(243, 128) * (
            grid.v[x]**2 * (grid.v[x]**6 - 8 * grid.v[y]**2 * grid.v[z]**2 * (
                grid.v[y]**2 + grid.v[z]**2)))

        lvars.append(Eq(coeff, poly_factorize(tmp)))

    out = []
    for ei, weight in zip(grid.basis, grid.entropic_weights):
        t = prefactor * weight
        for j, comp in enumerate(ei):
            t *= cfs[j]**comp
        out.append(t)

    return EqDef(out, lvars)


def elbm_d3q15_equilibrium(grid, order=8):
    """
    Form of equilibrium defined in PRL 97, 010201 (2006).
    See also Chikatamarla, PhD Thesis, Eq. (5.7), (5.8).
    """
    rho = S.rho

    prefactor = Symbol('chi')
    coeff1 = Symbol('zeta_x')
    coeff2 = Symbol('zeta_y')
    coeff3 = Symbol('zeta_z')
    vsq = Symbol('vsq')
    vx, vy, vz = grid.v

    o = [(1 if i <= order else 0) for i in range(0, 9)]

    lvars = []
    lvars.append(Eq(vsq, grid.v.dot(grid.v)))
    lvars.append(Eq(prefactor, poly_factorize(
        rho * (1 -
               o[2] * Rational(3, 2) * vsq +
               o[4] * Rational(9, 8) * vsq**2 +
               o[6] * Rational(27, 16) * (-vsq**3 + 2 * (vy**2 + vz**2) *
                                   (vsq * vx**2 + vy**2 * vz**2) +
                                   20 * vx**2 * vy**2 * vz**2) +
               o[8] * (Rational(81, 128) * vsq**4 +
                       Rational(81, 32) * (
                           vx**8 + vy**8 + vz**8
                           - 36 * vx**2 * vy**2 * vz**2 * vsq
                           - vx**4 * vy**4
                           - vy**4 * vz**4
                           - vx**4 * vz**4))))))

    cfs = [coeff1, coeff2, coeff3]

    for i, coeff in enumerate(cfs):
        tmp = (1 +
               o[1] * 3 * grid.v[i] +
               o[2] * Rational(9, 2) * grid.v[i]**2 +
               o[3] * Rational(9, 2) * grid.v[i]**3 +
               o[4] * Rational(27, 8) * grid.v[i]**4)

        # Cyclic permutation.
        x = i
        y = (i + 1) % 3
        z = (i + 2) % 3

        tmp += o[5] * Rational(27, 8) * (
            grid.v[x]**5 - 4 * grid.v[x] * grid.v[y]**2 * grid.v[z]**2)
        tmp += o[6] * Rational(81, 16) * (
            grid.v[x]**6 - 8 * grid.v[x]**2 * grid.v[y]**2 * grid.v[z]**2)
        tmp += o[7] * Rational(81, 16) * (
            grid.v[x]**7 - 10 * grid.v[x]**3 * grid.v[y]**2 * grid.v[z]**2 +
            2 * grid.v[x] * grid.v[y]**2 * grid.v[z]**2 * vsq)
        tmp += o[8] * Rational(243, 128) * (
            grid.v[x]**8 + 16 * grid.v[x]**2 * grid.v[y]**2 * grid.v[z]**2
            * (grid.v[y]**2 + grid.v[z]**2))

        lvars.append(Eq(coeff, poly_factorize(tmp)))

    out = []
    for ei, weight in zip(grid.basis, grid.entropic_weights):
        t = prefactor * weight
        for j, comp in enumerate(ei):
            t *= cfs[j]**comp
        out.append(t)

    return EqDef(out, lvars)

def get_equilibrium(config, equilibria, grids, grid_idx):
    if config.entropic_equilibrium:
        grid = grids[grid_idx]
        if grid is D3Q15:
            return elbm_d3q15_equilibrium(grid)
        elif grid is D3Q19:
            return elbm_d3q19_equilibrium(grid)
        else:
            return elbm_equilibrium(grid)
    else:
	    return equilibria[grid_idx](grids[grid_idx], config)


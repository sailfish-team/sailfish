# coding=utf-8

"""Helper code for symbolic processing and RTCG."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import copy
from collections import namedtuple
import operator
from operator import itemgetter
import math
import re
import numpy
import sympy
from sympy import Matrix, Rational, Symbol, Eq

from sailfish import sym_codegen

TargetDist = namedtuple('TargetDist', 'var idx')

#
# Classes for different grid types.
#
# These classes are not supposed to be instantiated -- they serve as containers for
# information about the grid.
#

class DxQy(object):
    mx = Symbol('mx')
    my = Symbol('my')
    mz = Symbol('mz')

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
        elif model == 'elbm':
            return ((cls.Q == 9 and cls.dim == 2) or
                    (cls.Q == 15 and cls.dim == 3))
        else:
            return True

    @classmethod
    def vec_to_dir(cls, vec):
        """Convert a primary direction vector (n-tuple) into a direction number."""
        return cls.vecidx2dir[cls.vec_idx(vec)]

    @classmethod
    def vec_idx(cls, vec):
        return cls.basis.index(Matrix((vec,)))

    @classmethod
    def dir_to_vec(cls, dir):
        """Convert a direction number into the corresponding n-vector."""
        return cls.basis[cls.dir2vecidx[dir]]

class D2Q9(DxQy):
    dim = 2
    Q = 9

    # Gravitational acceleration for free surface models.
    gravity = Symbol('gravity')

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

        cls.mrt_collision[n2i['pxx']] = 1 / (0.5 + S.visc * Rational(12, 2-c1))
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
                        (-8*vec_rho.dot(vec_rho)*S.rho +
                            18*(vec_mx.dot(vec_mx)*cls.mx**2 + vec_my.dot(vec_my)*cls.my**2)))
            elif name == 'ens':
                # The 4 and -18 below are freely adjustable parameters.
                t = (Rational(1, vec_e.dot(vec_e)) *
                        (4*vec_rho.dot(vec_rho)*S.rho +
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

            t = sym_codegen.poly_factorize(t)
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

        cls.mrt_collision[n2i['pxx']] = 2 / (8 * S.visc + 1)
        cls.mrt_collision[n2i['pww']] = cls.mrt_collision[n2i['pxx']]
        cls.mrt_collision[n2i['pxy']] = 2 / (4 * S.visc + 1)
        cls.mrt_collision[n2i['pyz']] = cls.mrt_collision[n2i['pxy']]
        cls.mrt_collision[n2i['pzx']] = cls.mrt_collision[n2i['pxy']]

        # Form of the equilibrium functions follows that from
        # PhysRevE.63.066702.

        mrt_eq = {
            'm3x': 0,
            'm3y': 0,
            'm3z': 0,
            'pxy': 1/S.rho0 * (cls.mx * cls.my),
            'pyz': 1/S.rho0 * (cls.my * cls.mz),
            'pzx': 1/S.rho0 * (cls.mx * cls.mz),
            'pxx': 1/S.rho0 * (2 * cls.mx**2 - cls.my**2 - cls.mz**2),
            'pww': 1/S.rho0 * (cls.my**2 - cls.mz**2),
            'en': 3*S.rho*(13*cls.cssq - 8)/2 + 13/(2 * S.rho0)*(cls.mx**2 + cls.my**2 + cls.mz**2),
        }

        for i, name in enumerate(cls.mrt_names):
            if cls.mrt_collision[i] == 0:
                cls.mrt_equilibrium.append(0)
            else:
                cls.mrt_equilibrium.append(mrt_eq[name])


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

    entropic_weights = weights

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

        cls.mrt_collision[n2i['pxx']] = 1 / (0.5 + 3*S.visc)
        cls.mrt_collision[n2i['pww']] = cls.mrt_collision[n2i['pxx']]
        cls.mrt_collision[n2i['pxy']] = cls.mrt_collision[n2i['pxx']]
        cls.mrt_collision[n2i['pyz']] = cls.mrt_collision[n2i['pxx']]
        cls.mrt_collision[n2i['pzx']] = cls.mrt_collision[n2i['pxx']]

        # Form of the equilibrium functions follows that from
        # dHumieres, PhilTranA, 2002.
        mrt_eq = {
            'en':  -S.rho + 1/S.rho0 * (cls.mx**2 + cls.my**2 + cls.mz**2),
            'ens': -S.rho,
            'ex':  -Rational(7,3)*cls.mx,
            'ey':  -Rational(7,3)*cls.my,
            'ez':  -Rational(7,3)*cls.mz,
            'pxx': 1/S.rho0 * (2*cls.mx**2 - (cls.my**2 + cls.mz**2)),
            'pww': 1/S.rho0 * (cls.my**2 - cls.mz**2),
            'pxy': 1/S.rho0 * (cls.mx * cls.my),
            'pyz': 1/S.rho0 * (cls.my * cls.mz),
            'pzx': 1/S.rho0 * (cls.mx * cls.mz),
            'mxyz': 0
        }

        for i, name in enumerate(cls.mrt_names):
            if cls.mrt_collision[i] == 0:
                cls.mrt_equilibrium.append(0)
            else:
                cls.mrt_equilibrium.append(mrt_eq[name])

class D3Q19(DxQy):
    dim = 3
    Q = 19

    basis = map(lambda x: Matrix((x, )),
                [(0,0,0),
                (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
                (1,1,0), (-1,1,0), (1,-1,0), (-1,-1,0),
                (0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1),
                (1,0,1), (-1,0,1), (1,0,-1), (-1,0,-1),
                ])

    weights = map(lambda x: Rational(*x),
            [(1,3), (1,18), (1,18), (1,18), (1,18), (1,18), (1,18),
                (1,36), (1,36), (1,36), (1,36), (1,36), (1,36),
                (1,36), (1,36), (1,36), (1,36), (1,36), (1,36)])

    entropic_weights = weights

    mrt_names = ['rho', 'en', 'eps', 'mx', 'ex', 'my', 'ey', 'mz', 'ez',
                 'pxx3', 'pixx3', 'pww', 'piww', 'pxy', 'pyz', 'pzx', 'm3x', 'm3y', 'm3z']

    mrt_collision = [0.0, 1.19, 1.4, 0.0, 1.2, 0.0, 1.2, 0.0, 1.2,
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
            [3*x[0]*x[0] - x.dot(x) for x in cls.basis],
            [(3*x.dot(x) - 5) * (3*x[0]*x[0] - x.dot(x)) for x in cls.basis],
            [x[1]*x[1] - x[2]*x[2] for x in cls.basis],
            [(3*x.dot(x) - 5) * (x[1]*x[1] - x[2]*x[2]) for x in cls.basis],
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

        cls.mrt_collision[n2i['pxx3']] = 1 / (0.5 + 3*S.visc)
        cls.mrt_collision[n2i['pww']] = cls.mrt_collision[n2i['pxx3']]
        cls.mrt_collision[n2i['pxy']] = cls.mrt_collision[n2i['pxx3']]
        cls.mrt_collision[n2i['pyz']] = cls.mrt_collision[n2i['pxx3']]
        cls.mrt_collision[n2i['pzx']] = cls.mrt_collision[n2i['pxx3']]

        # Form of the equilibrium functions follows that from
        # dHumieres, PhilTranA, 2002.
        mrt_eq = {
            'en':  -11 * S.rho + 19/S.rho0 * (cls.mx**2 + cls.my**2 + cls.mz**2),
            'eps': -Rational(475,63)/S.rho0 * (cls.mx**2 + cls.my**2 + cls.mz**2),
            'ex':  -Rational(2,3)*cls.mx,
            'ey':  -Rational(2,3)*cls.my,
            'ez':  -Rational(2,3)*cls.mz,
            'pxx3': 1/(S.rho0) * (2*cls.mx**2 - (cls.my**2 + cls.mz**2)),
            'pww': 1/S.rho0 * (cls.my**2 - cls.mz**2),
            'pxy': 1/S.rho0 * (cls.mx * cls.my),
            'pyz': 1/S.rho0 * (cls.my * cls.mz),
            'pzx': 1/S.rho0 * (cls.mx * cls.mz),
            'm3x': 0,
            'm3y': 0,
            'm3z': 0,
            'pixx3': 0,
            'piww': 0,
        }

        for i, name in enumerate(cls.mrt_names):
            if cls.mrt_collision[i] == 0:
                cls.mrt_equilibrium.append(0)
            else:
                cls.mrt_equilibrium.append(mrt_eq[name])


class D3Q27(DxQy):
    dim = 3
    Q = 27

    basis = map(lambda x: Matrix((x, )),
                [(0,0,0),
                 (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1),
                 (1,1,0), (-1,1,0), (1,-1,0), (-1,-1,0),
                 (0,1,1), (0,-1,1), (0,1,-1), (0,-1,-1),
                 (1,0,1), (-1,0,1), (1,0,-1), (-1,0,-1),
                 (1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                 (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)])

    weights = map(lambda x: Rational(*x),
                  [(8, 27),
                   (2, 27), (2, 27), (2, 27), (2, 27), (2, 27), (2, 27),
                   (1, 54), (1, 54), (1, 54), (1, 54), (1, 54), (1, 54),
                   (1, 54), (1, 54), (1, 54), (1, 54), (1, 54), (1, 54),
                   (1, 216), (1, 216), (1, 216), (1, 216),
                   (1, 216), (1, 216), (1, 216), (1, 216)])


def alpha_series():
    """See Phys Rev Lett 97, 010201 (2006) for the expression."""

    a1 = Symbol('a1')
    a2 = Symbol('a2')
    a3 = Symbol('a3')
    a4 = Symbol('a4')

    alpha = sym_codegen.poly_factorize(2
            - 4 * a2 / a1
            + 16 * a2**2 / a1**2
            - 8 * a3 / a1
            + 80 * a2 * a3 / a1**2
            - 80 * a2**3 / a1**3
            - 16 * a4 / a1)

    return alpha


def free_energy_mrt_matrix(grid):
    tau0 = Symbol('tau0')

    def matrix_constructor(i, j):
        if i == j:
            if type(grid.mrt_collision[i]) is float:
                if grid.mrt_collision[i] == 0.0:
                    return 0.0
                else:
                    return 1.0
            else:
                return 1.0 / tau0
        else:
            return 0

    return grid.mrt_matrix.inv() * Matrix(grid.Q, grid.Q, matrix_constructor) * grid.mrt_matrix


def bb_swap_pairs(grid):
    """Get a set of indices which have to be swapped for a full bounce-back."""
    ret = set()

    for i, j in enumerate(grid.idx_opposite):
        # Nothing to swap with.
        if i == j:
            continue

        ret.add(min(i,j))

    return ret

def slip_bb_swap_pairs(grid, normal_dir):
    ret = set()
    normal_vec = grid.dir_to_vec(normal_dir)

    for i, ei in enumerate(grid.basis):
        sp = ei.dot(normal_vec)
        if sp > 0:
            expected = copy.deepcopy(ei)
            for j, val in enumerate(normal_vec):
                if val != 0:
                    expected[j] *= -1

            for j, ej in enumerate(grid.basis):
                if ej == expected:
                    ret.add((i, j))
                    break
    return ret

def missing_dirs_from_tag(grid, tag_code):
    """Generates a list of missing mass fractions given an orientation tag code.

    Useful to debugging.

    :param grid: grid object
    :param tag_code: encoded map of missing mass fractions; bits set to 1
        indicate directions pointing to fluid nodes
    """

    ret = []

    for i, name in enumerate(grid.idx_name[1:]):
        if (tag_code & 1) == 0:
            ret.append(grid.idx_name[grid.idx_opposite[i + 1]])
        tag_code >>= 1

    return ret

def get_missing_dists(grid, orientation):
    """Returns an iterable of missing distribution indices.

    :param grid: grid object
    :param orientation: orientation code
    """
    # Normal vector points inside the simulation domain.
    normal = grid.dir_to_vec(orientation)
    _, unknown = _get_known_dists(grid, normal)
    return unknown


def fill_missing_dists(grid, distp, missing_dir):
    """
    :param grid: grid object
    :param distp: name of the pointer to the Dist structure
    :param missing_dir: orientation code
    """
    syms = [Symbol('%s->%s' % (distp, x)) for x in grid.idx_name]
    ret = []

    for idx in get_missing_dists(grid, missing_dir):
        ret.append((TargetDist(syms[idx], idx), syms[grid.idx_opposite[idx]]))

    return ret

def ex_rho(grid, distp, incompressible, missing_dir=None):
    """Express density as a function of the distributions.

    :param distp: name of the pointer to the distribution structure
    :param incompressible: if ``True``, an expression for the incompressible
        model will be returned
    :param missing_dir: direction number specified if an expression for
        a node where not all distributions are known is necessary. This
        parameter identifies the normal vector pointing towards the
        fluid (i.e. the distributions in this direction are unknown).

    :rtype: sympy expression for the density
    """
    syms = [Symbol('%s->%s' % (distp, x)) for x in grid.idx_name]
    ret = 0

    if missing_dir is None:
        for sym in syms:
            ret += sym
        return ret

    # This is derived by considering a system of equations for the macroscopic
    # quantities and cancelling out the unknown distributions so as to get an
    # expression for rho using only known quantities.
    if incompressible:
        return S.rho + S.rho0 * grid.dir_to_vec(missing_dir).dot(grid.v)
    else:
        return S.rho / (1 - grid.dir_to_vec(missing_dir).dot(grid.v))

def ex_velocity(grid, distp, comp, config, momentum=False, missing_dir=None,
                par_rho=None):
    """Express velocity as a function of the distributions.

    :param distp: name of the pointer to the distribution structure
    :param comp: velocity component number: 0, 1 or 2 (for 3D lattices)
    :param config: LBConfig object
    :param momentum: if ``True``, an expression for momentum is returned instead
        of for velocity
    :param missing_dir: direction number specified if an expression for
        a node where not all distributions are known is necessary. This
        parameter identifies the normal vector pointing towards the
        fluid (i.e. the distributions in this direction are unknown).
    :param par_rho: name of the variable (a string) containing the externally
        imposed density (e.g. from a boundary condition)
    :param roundoff: if True, the round-off optimization model is used and
        rho should be replaced by (1.0 + rho) in calculations.

    :rtype: sympy expression for the velocity in a given direction
    """
    syms = [Symbol('%s->%s' % (distp, x)) for x in grid.idx_name]
    ret = 0

    if missing_dir is None:
        for i, sym in enumerate(syms):
            ret += grid.basis[i][comp] * sym

        if config.incompressible:
            rho = S.rho0
        elif config.minimize_roundoff:
            rho = S.rho + 1.0
        else:
            rho = S.rho

        if not momentum:
            ret = ret / rho
    else:
        prho = Symbol(par_rho)

        for i, sym in enumerate(syms):
            sp = grid.basis[i].dot(grid.dir_to_vec(missing_dir))
            if sp <= 0:
                ret = 1

        rho = S.rho
        if config.minimize_roundoff:
            rho += 1

        ret = ret * (rho - prho)
        ret *= -grid.dir_to_vec(missing_dir)[comp]
        if not momentum:
            ret = ret / prho

    return ret

def ex_flux(grid, distp, comp_a, comp_b, config):
    syms = [Symbol('%s->%s' % (distp, x)) for x in grid.idx_name]
    ret = 0

    for i, sym in enumerate(syms):
        ret += grid.basis[i][comp_a] * grid.basis[i][comp_b] * sym

    # rho_0 c_s^2 δ_αβ
    if config.minimize_roundoff and comp_a == comp_b:
        ret += grid.cssq  # rho_0 == 1

    return ret

# rho / 3 * \delta_{ab} + rho u_a u_b
def ex_eq_flux(grid, comp_a, comp_b):
    if comp_a != comp_b:
        return S.rho * grid.v[comp_a] * grid.v[comp_b]
    else:
        return S.rho * (grid.v[comp_a]**2 + grid.cssq)

def free_energy_mrt(grid, dest_dist, src_dist):
    src_syms = Matrix([Symbol('%s.%s' % (src_dist, x)) for x in grid.idx_name])
    dst_syms = [Symbol('%s->%s' % (dest_dist, x)) for x in grid.idx_name]
    ret = []

    mtx = free_energy_mrt_matrix(grid)

    for i, rhs in enumerate(mtx * src_syms):
        ret.append((dst_syms[i], rhs))

    return ret

def bgk_to_mrt(grid, bgk_dist, mrt_dist):
    bgk_syms = Matrix([Symbol('%s->%s' % (bgk_dist, x)) for x in grid.idx_name])
    mrt_syms = [Symbol('%s.%s' % (mrt_dist, x)) for x in grid.mrt_names]

    ret = []

    for i, rhs in enumerate(grid.mrt_matrix * bgk_syms):
        ret.append((mrt_syms[i], rhs))

    return ret

def mrt_to_bgk(grid, bgk_dist, mrt_dist):
    bgk_syms = [Symbol('%s->%s' % (bgk_dist, x)) for x in grid.idx_name]
    mrt_syms = Matrix([Symbol('%s.%s' % (mrt_dist, x)) for x in grid.mrt_names])

    ret = []

    for i, rhs in enumerate(grid.mrt_matrix.inv() * mrt_syms):
        ret.append((bgk_syms[i], rhs))
    return ret

def _get_known_dists(grid, normal):
    unknown = []
    known = []

    for i, vec in enumerate(grid.basis):
        if normal.dot(vec) > 0:
            unknown.append(i)
        else:
            known.append(i)

    return known, unknown


def noneq_bb(grid, orientation, eq):
    normal = grid.dir_to_vec(orientation)
    known, unknown = _get_known_dists(grid, normal)
    ret = []

    # Bounce-back of the non-equilibrium parts.
    for i in unknown:
        oi = grid.idx_opposite[i]
        ret.append((Symbol('fi->%s' % grid.idx_name[i]),
                    Symbol('fi->%s' % grid.idx_name[oi]) - eq[oi] + eq[i]))

    for i in range(0, len(ret)):
        t = sym_codegen.poly_factorize(ret[i][1])

        ret[i] = (ret[i][0], t)

    return ret

def zouhe_fixup(grid, orientation):
    normal = grid.dir_to_vec(orientation)
    known, unknown = _get_known_dists(grid, normal)

    unknown_not_normal = set(unknown)
    unknown_not_normal.remove(grid.basis.index(normal))

    ret = []

    # Momentum differences.
    mdiff = [Symbol('nvx'), Symbol('nvy'), Symbol('nvz')]

    if grid.dim == 2:
        basis = [Matrix([1,0]), Matrix([0,1])]
    else:
        basis = [Matrix([1,0,0]), Matrix([0,1,0]), Matrix([0,0,1])]

    # Scale by number of adjustable distributions.
    for i in range(0, grid.dim):
        if basis[i].dot(normal) != 0:
            mdiff[i] = 0
            continue

        md = mdiff[i]

        cnt = 0
        for idir in unknown_not_normal:
            if grid.basis[idir].dot(basis[i]) != 0:
                cnt += 1

        ret.append((md, md / cnt))

    # Adjust distributions to conserve momentum in directions other than
    # the wall normal.
    for i in unknown_not_normal:
        val = 0
        ei = grid.basis[i]
        if ei[0] != 0 and mdiff[0] != 0:
            val += mdiff[0] * ei[0]
        if ei[1] != 0 and mdiff[1] != 0:
            val += mdiff[1] * ei[1]
        if grid.dim == 3 and ei[2] != 0 and mdiff[2] != 0:
            val += mdiff[2] * ei[2]

        if val != 0:
            csym = Symbol('fi->%s' % grid.idx_name[i])
            ret.append((csym, csym + val))

    return ret


def get_prop_dists(grid, dir_, axis=0):
    """Computes a list of base vectors with a specific value of the X component (`dir`)."""
    ret = []

    for i, ei in enumerate(grid.basis):
        if ei[axis] == dir_ and i > 0:
            ret.append(i)

    return ret

def get_interblock_dists(grid, direction, opposite=False):
    """Computes a list of indices of distributions that would be transferred
    to a node pointed to by the vector 'direction'.
    """
    d = Matrix((direction,))

    def process_dists(dists):
        if opposite:
            return [grid.idx_opposite[x] for x in dists]
        else:
            return dists

    ret = []
    for i, ei in enumerate(grid.basis):
        if ei.dot(d) >= d.dot(d):
            ret.append(i)
    return process_dists(ret)

def relaxation_time(viscosity):
    return (6.0 * viscosity + 1.0) / 2.0

def _pressure_tensor(grid):
    press = [Symbol('flux[%d]' % i) for i in range(grid.dim * (grid.dim + 1) / 2)]
    P = sympy.ones(grid.dim)  # P_ab - rho cs^2 \delta_ab
    k = 0
    for i in range(grid.dim):
        for j in range(i, grid.dim):
            P[i, j] = press[k]
            P[j, i] = press[k]
            k += 1
    return P

# e_ia e_ib - cs^2 \delta_ab
def _q_tensor(grid, ei):
    return ei.transpose().multiply(ei) - sympy.eye(grid.dim) * grid.cssq

def grad_approx(grid):
    """Returns expressions for the Grad distributions as defined by Eq. 11 in
    EPL 74 (2), pp. 215-221 (2006).

    :param grid: a grid object
    """
    out = []
    P = _pressure_tensor(grid) - sympy.eye(grid.dim) * S.rho * grid.cssq

    for ei, weight in zip(grid.basis, grid.weights):
        Q = _q_tensor(grid, ei)
        t = sum(P.multiply_elementwise(Q)) / (grid.cssq**2 * 2)
        t += S.rho + S.rho * ei.dot(grid.v) / grid.cssq
        t *= weight
        out.append(t)
    return out

def reglb_flux_tensor(grid):
    """Returns expressions for the non-equilibrium flux tensor contraction
    with the Q tensor, used for the regularized LB model.
    """
    out = []
    P = _pressure_tensor(grid)
    for ei, weight in zip(grid.basis, grid.weights):
        Q = _q_tensor(grid, ei)
        out.append(sum(P.multiply_elementwise(Q)) * weight / (2 * grid.cssq**2))
    return out

#
# Shan-Chen model.
#
def shan_chen_linear(field):
    f = Symbol(field)
    return f

def shan_chen_classic(field):
    rho0 = 1.0
    f = Symbol(field)
    return rho0 * (1.0 - sympy.exp(-f / rho0))

SHAN_CHEN_POTENTIALS = {
    'linear': shan_chen_linear,
    'classic': shan_chen_classic
}

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

    :param vectors: a collection of vectors to orthogonalize

    :rtype: orthogonalized vectors
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

    D1Q3_entropic_weights = {
            -1: Rational(1,6),
            0:  Rational(2,3),
            1:  Rational(1,6)}

    for grid in KNOWN_GRIDS:
        grid.vx = S.vx
        grid.vy = S.vy
        grid.vz = S.vz

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

        grid.dir2vecidx = {}
        grid.vecidx2dir = {}
        dir = 1

        # Weights used by the entropic LB model are calculated automatically
        # from the D1Q3 weights unless specified explicitly in the grid class
        # definition.
        needs_entropic_weights = not hasattr(grid, 'entropic_weights')
        if needs_entropic_weights:
            grid.entropic_weights = []

        for k, ei in enumerate(grid.basis):
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

            # Index primary direction vectors.  For cartesian grids, there
            # are always 2*Q such vectors.
            if ei.dot(ei) == 1:
                grid.dir2vecidx[dir] = k
                grid.vecidx2dir[k] = dir
                dir += 1

            if needs_entropic_weights:
                ent_weight = 1
                for comp in ei:
                    ent_weight *= D1Q3_entropic_weights[int(comp)]
                grid.entropic_weights.append(ent_weight)

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


# A container class for all commonly used sympy symbols.
class S(object):
    aliases = {}

    @classmethod
    def alias(cls, sym_dst, sym_src):
        setattr(cls, sym_dst, sym_src)
        cls.aliases[sym_src] = sym_dst

    @classmethod
    def make_vector(cls, sym_dst, dim, *syms):
        if dim == 3:
            setattr(cls, sym_dst, Matrix(([syms[0], syms[1], syms[2]],)))
        else:
            setattr(cls, sym_dst, Matrix(([syms[0], syms[1]],)))


class SlfSymbol(Symbol):
    def __new__(cls, name, comment=None):
        return Symbol.__new__(cls, name)

    def __init__(self, name, comment=None):
        Symbol.__init__(name)
        self.comment = comment


def _prepare_symbols():
    comp_map = {0: 'x', 1: 'y', 2: 'z'}

    # Grid ID -> relaxation time symbol.
    S.relaxation_times = []
    S.densities = []

    # Up to 9 grids.
    for grid_num in range(0, 9):
        # 0th moment
        name = 'g%sm0' % grid_num
        s = Symbol(name)
        setattr(S, name, s)
        S.densities.append(s)

        # Laplacian.
        name = 'g%sd2m0' % grid_num
        setattr(S, name, Symbol(name))
        # Gradient.
        name = 'g%sd1m0' % grid_num
        for comp in ('x', 'y', 'z'):
            setattr(S, name + comp, Symbol(name + comp))

        # 1st moment
        name = 'g%sm1' % grid_num
        for comp in ('x', 'y', 'z'):
            setattr(S, name + comp, Symbol(name + comp))

        # 2nd moment
        name = 'g%sm2' % grid_num
        for c1 in range(0, 3):
            for c2 in range(c1, 3):
                n = name + comp_map[c1] + comp_map[c2]
                setattr(S, n, Symbol(n))

        # External acceleration.
        name = 'g%sea' % grid_num
        for comp in ('x', 'y', 'z'):
            setattr(S, name + comp, Symbol(name + comp))

        # Relaxation time.
        name = 'tau%d' % grid_num
        t = Symbol(name)
        setattr(S, name, t)
        S.relaxation_times.append(t)


    # Commonly used aliases.
    S.alias('vx', S.g0m1x)
    S.alias('vy', S.g0m1y)
    S.alias('vz', S.g0m1z)
    S.alias('rho', S.g0m0)
    S.alias('phi', S.g1m0)

    S.alias('eax', S.g0eax)
    S.alias('eay', S.g0eay)
    S.alias('eaz', S.g0eaz)

    # For incompressible models, this symbol is replaced with the average
    # density, usually 1.0.  For compressible models, it is the same as
    # the density rho.
    #
    # The incompressible model is described in:
    #   X. He, L.-S. Luo, Lattice Boltzmann model for the incompressible Navier-Stokes
    #   equation, J. Stat. Phys. 88 (1997) 927-944
    S.rho0 = Symbol('rho0')
    S.visc = Symbol('visc')
    S.gravity = Symbol('gravity')

    # Initial velocity. Use these symbols to refer to the initial velocity
    # (without acceleration correction) in relaxation code.
    S.ivx = Symbol('iv0[0]')
    S.ivy = Symbol('iv0[1]')
    S.ivz = Symbol('iv0[2]')

    # Node coordinate in the global coordinate system.
    S.gx = SlfSymbol('gx', 'X node location in the global coordinate system')
    S.gy = SlfSymbol('gy', 'Y node location in the global coordinate system')
    S.gz = SlfSymbol('gz', 'Z node location in the global coordinate system')
    S.time = SlfSymbol('phys_time', 'time in physical units')

KNOWN_GRIDS = (D2Q9, D3Q13, D3Q15, D3Q19, D3Q27)

_prepare_symbols()
_prepare_grids()

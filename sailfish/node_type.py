# # -*- coding: utf-8 -*-
"""Supporting code for different node types."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import namedtuple
import numpy as np
from sympy.core import expr

ScratchSize = namedtuple('ScratchSize', ('dim2', 'dim3'))


# Node type classes.  Use this to define boundary conditions.
class LBNodeType(object):
    # Initialized when module is loaded.
    id = None
    wet_node = False

    # If True, the node does not participate in the simulation.
    excluded = False

    # If True, the node does not require any special processing to calculate
    # macroscopic fluid quantities.  Typical examples of nodes requiring
    # special treatment are edge and corner nodes where not all mass fractions
    # are known.
    standard_macro = False

    # If True, the node needs a basic orientation vector (only primary
    # directions are currently supported).
    needs_orientation = False

    # Number of floating point values that the node needs to keep as additional
    # information in global memory. This can be an int or an instance of
    # ScratchSize.
    scratch_space = 0

    # What is the effective location of the boundary condition along the
    # direction of the normal vector. Positive values indicate offset towards
    # the fluid domain, negative values insidate offset away from the fluid
    # domain.
    location = 0.0

    def __init__(self, **params):
        if 'orientation' in params:
            self.orientation = params['orientation']
            del params['orientation']
        self.params = params

    @classmethod
    def scratch_space_size(cls, dim):
        """Returns the required size (# of floats) of the scratch space.

        :param dim: dimensionality of the simulation (2 or 3)
        """
        if type(cls.scratch_space) is int:
            return cls.scratch_space
        else:
            if dim == 2:
                return cls.scratch_space.dim2
            else:
                return cls.scratch_space.dim3

############################################################################
# Special node types, do not use directly.
############################################################################

class _NTFluid(LBNodeType):
    """Fluid node."""
    wet_node = True
    standard_macro = True
    id = 0


class _NTGhost(LBNodeType):
    """Ghost node."""
    excluded = True


class _NTUnused(LBNodeType):
    """Unused node."""
    excluded = True

############################################################################
# Wall (no-slip) nodes.
############################################################################

class NTHalfBBWall(LBNodeType):
    """Half-way bounce-back (no-slip) node.

    The effective location of the wall is half a lattice spacing away from
    the bulk of the fluid.
    """
    wet_node = True
    standard_macro = True
    needs_orientation = True
    location = -0.5


class NTFullBBWall(LBNodeType):
    """Full-way bounce-back (no-slip) node.

    The effective location of the wall is half a lattice spacing between
    the wall node and the fluid node.
    """
    standard_macro = True
    location = 0.5

    # Only necessary for wetting conditions in binary fluid models.
    needs_orientation = True


class NTWallTMS(LBNodeType):
    """Wall boundary condition for turbulent flows, based on the Tamm-Mott-Smith
    approximation.

    For more info see:
    S.S. Chikatamarla, I.V. Karlin, "Entropic lattice Boltzmann method for
    turbulent flow simulations: Boundary conditions", Physica A (2013),
    doi: 10.1016/j.physa.2012.12.034
    """
    wet_node = True
    needs_orientation = True
    location = -0.5

    # This will cause the standard procedure to compute the instantaneous u and
    # rho as defined in the paper.
    standard_macro = True

    @classmethod
    def update_context(cls, ctx):
        ctx['misc_bc_vars'].extend(('tg_rho', 'tg_v'))

############################################################################
# Density (pressure) nodes.
############################################################################

class NTEquilibriumDensity(LBNodeType):
    """Density boundary condition using the equilibrium distribution."""
    needs_orientation = True
    wet_node = True

    def __init__(self, density, orientation=None):
        self.params = {'density': density}
        self.orientation = orientation


class NTRegularizedDensity(LBNodeType):
    """Density boundary condition using the regularized LB model and
    non-equilibrium bounce-back.

    See Phys. Rev. E 77, 056703 (2008) for more info.
    """
    needs_orientation = True
    wet_node = True

    def __init__(self, density, orientation=None):
        self.params = {'density': density}
        self.orientation = orientation


class NTGuoDensity(LBNodeType):
    """Guo density."""

    def __init__(self, density):
        self.params = {'density': density}


class NTZouHeDensity(LBNodeType):
    """Zou-He density."""
    needs_orientation = True
    wet_node = True

    def __init__(self, density, orientation=None):
        self.params = {'density': density}
        self.orientation = orientation


############################################################################
# Velocity nodes.
############################################################################

class NTEquilibriumVelocity(LBNodeType):
    """Velocity boundary condition using the equilibrium distribution."""
    needs_orientation = True
    wet_node = True

    def __init__(self, velocity, orientation=None):
        self.params = {'velocity': velocity}
        self.orientation = orientation


class NTZouHeVelocity(LBNodeType):
    """Zou-he velocity."""
    needs_orientation = True
    wet_node = True

    def __init__(self, velocity, orientation=None):
        self.params = {'velocity': velocity}
        self.orientation = orientation


class NTRegularizedVelocity(LBNodeType):
    """Velocity boundary condition using the regularized LB model and
    non-equilibrium bounce-back.

    See Phys. Rev. E 77, 056703 (2008) for more info.
    """
    needs_orientation = True
    wet_node = True

    def __init__(self, velocity, orientation=None):
        self.params = {'velocity': velocity}
        self.orientation = orientation

############################################################################
# Outflow (zero-gradient) nodes.
############################################################################

class NTGradFreeflow(LBNodeType):
    """Outflow node using Grad's approximation.

    Note: this node type currently only works with the AB memory layout.
    """
    wet_node = True
    standard_macro = True
    scratch_space = ScratchSize(dim2=3, dim3=6)


class NTDoNothing(LBNodeType):
    """Outflow node without any special processing for undefined distributions.

    The value from the previous time step is retained for all undefined
    distributions (i.e. pointing into the domain at the boundary).

    Using this boundary condition is only necessary with the AA memory layout.
    In the AB memory layout, leaving the outflow nodes defined as NTFluid works
    just fine."""
    wet_node = True
    needs_orientation = True


class NTCopy(LBNodeType):
    """Copies distributions from another node.

    This can be used to implement a crude vanishing gradient
    boundary condition."""
    wet_node = True
    standard_macro = True

    def __init__(self, normal):
        """
        :param normal: direction number corresponding to the normal vector
            pointing outward (i.e. outside of the domain); direction numbers
            can be generated using grid.vec_to_dir.
        """
        self.params = {'normal': normal}


class NTYuOutflow(LBNodeType):
    """Implements the open boundary condition described in:

    Yu, D., Mei, R. and Shyy, W. (2005) 'Improved treatment of
    the open boundary in the method of lattice Boltzmann
    equation', page 5.

    This is an extrapolation based method, using data from next-nearest
    neighbors:

    f_i(x_j) = 2 f_i(x_j - n) - f_i(x_j - 2n)
    """
    wet_node = True
    standard_macro = True

    def __init__(self, normal):
        """
        :param normal: direction number corresponding to the normal vector
            pointing outward (i.e. outside of the domain); direction numbers
            can be generated using grid.vec_to_dir.
        """
        self.params = {'normal': normal}


class NTNeumann(LBNodeType):
    """Implements a Neumann boundary condition.

    This is a nonlocal boundary condition accessing the nearest neighbors.
    Note that this condition requires a layer of ghost nodes to be present
    in the direction pointed to by the normal vector.

    Based on the NBC description in:
    Junk M, Yang Z, Outflow boundary conditions for the lattice
    Boltzmann method, Progress in Computational Fluid Dynamics,
    Vol. 8, Nos. 1–4, 2008

    Implements:
      \partial u / \partial n (t, x_j) = \phi(t, x_j)
    via:
      f_i(t+1, j_0) = f_iopp^c(t, j_o + c_i) +
                      6 feq_i (u(t, x_j1) + 2\phi(t, x_j)) \cdot c_i
    with:
      j_0: ghost node
      x_j: actual boundary node
      x_j1: fluid node at x_j - normal
      c_i: incoming distributions
      iopp: direction opposite to i
      feq_i: f_eq(1, 0)
      f_i^c: distribution after collision (prior to streaming)
    """
    def __init__(self, normal):
        """
        :param normal: direction number corresponding to the normal vector
            pointing outward (i.e. outside of the domain); direction numbers
            can be generated using grid.vec_to_dir.
        """
        self.params = {'normal': normal}

############################################################################
# Other nodes.
############################################################################

class NTSlip(LBNodeType):
    """Full-slip node."""
    standard_macro = True

def __init_node_type_list():
    """Assigns IDs to classes descendant from LBNodeType."""
    ret = []
    import sys
    curr_module = sys.modules[__name__]
    for symbol in dir(curr_module):
        obj = getattr(curr_module, symbol)
        try:
            if obj != LBNodeType and issubclass(obj, LBNodeType):
                ret.append(obj)
                if obj.id is None:
                    obj.id = len(ret)
        except TypeError:
            pass

    return dict([(node_type.id, node_type) for node_type in ret])

def get_wet_node_type_ids():
    return [id for id, nt in _NODE_TYPES.iteritems() if nt.wet_node]

def get_dry_node_type_ids():
    return [id for id, nt in _NODE_TYPES.iteritems() if not nt.wet_node]

def get_orientation_node_type_ids():
    return [id for id, nt in _NODE_TYPES.iteritems() if nt.needs_orientation]

def multifield(values, where=None):
    """Collapses a list of numpy arrays into a structured field that can be
    used for setting boundary condition parameters at multiple nodes at the
    same time.

    :param values: iterable of ndarrays or floats
    :param where: numpy boolean array or indexing expression to select
            nodes from the arrays in 'values'; if you are creating the
            fields in 'values' using mgrid expressions (with hx, hy, ...)
            which cover the whole subdomain, 'where' should be the same
            array you're passing to set_node
    """
    # TODO(michalj): Add support for automatic array extension (broadcasting).
    shape = None
    new_values = []
    for val in values:
        if isinstance(val, np.ndarray):
            assert (shape is None) or (shape == val.shape)
            new_values.append(val.astype(np.float64))
            shape = val.shape
        else:
            new_values.append(None)

    assert shape is not None

    for i, (old, new) in enumerate(zip(values, new_values)):
        if new is None:
            new_values[i] = np.zeros(shape, dtype=np.float64)
            new_values[i][:] = old  # broadcast to the whole array

    if where is not None:
        return np.core.records.fromarrays(new_values)[where]
    else:
        return np.core.records.fromarrays(new_values).flatten()

class DynamicValue(object):
    """A node parameter that is evaluated on the device.

    This is typically used for time-dependent boundary conditions."""

    def __init__(self, *params):
        """:param params: sympy expressions"""
        self.params = params

    def __hash__(self):
        return hash(self.params)

    def __cmp__(self, other):
        return cmp(self.params, other.params)

    def __iter__(self):
        if type(self.params) is tuple or type(self.params) is list:
            return iter(self.params)
        else:
            return iter((self.params, ))

    def __len__(self):
        if type(self.params) is tuple or type(self.params) is list:
            return len(self.params)
        else:
            return 1

    def __getitem__(self, i):
        return self.params[i]

    def has_symbols(self, *args):
        """Returns True if any of the expressions used for this DynamicValue
        depends on at least one of the symbols provided as arguments."""
        for symbol in args:
            for param in self.params:
                if isinstance(param, expr.Expr) and symbol in param:
                    return True
        return False

# Maps node type IDs to their classes.
_NODE_TYPES = __init_node_type_list()

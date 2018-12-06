# # -*- coding: utf-8 -*-
"""Supporting code for different node types."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import hashlib
from collections import namedtuple
import numpy as np
from sympy import Symbol, ImmutableDenseMatrix
from sympy.core import expr
from sailfish.util import is_number

ScratchSize = namedtuple('ScratchSize', ('dim2', 'dim3'))


class LBNodeType(object):
    """Base class for node types."""

    # Initialized when module is loaded.
    id = None

    #: If True, the node undergoes normal relaxation process.
    wet_node = False

    #: If True, the node does not participate in the simulation.
    excluded = False

    #: If True, the node participates in the propagation step only. This is
    #: necessary for some schemes that use propagation in shared memory.
    propagation_only = False

    #: If True, the node does not require any special processing to calculate
    #: macroscopic fluid quantities.  Typical examples of nodes requiring
    #: special treatment are edge and corner nodes where not all mass fractions
    #: are known.
    standard_macro = False

    #: If True, the node needs a basic orientation vector (only primary
    #: directions are currently supported).
    needs_orientation = False

    #: If True, the node supports tagging of active directions (i.e. of links
    #: that connect it to a fluid node.
    link_tags = False

    #: Number of floating point values that the node needs to keep as additional
    #: information in global memory. This can be an int or an instance of
    #: ScratchSize.
    scratch_space = 0

    #: Indicates the effective location of the boundary condition along the
    #: direction of the normal vector. Positive values indicate offset towards
    #: the fluid domain, negative values indicate offset away from the fluid
    #: domain.
    location = 0.0

    #: If True, the node can be marked as unused. Only valid for wet nodes.
    allow_unused = False

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


class _NTPropagationOnly(LBNodeType):
    """Unused node participating in the propagation step.

    These lie beyond the active fluid nodes, and are used to provide sentinel
    values for the propagation step.
    """
    propagation_only = True

############################################################################
# Wall (no-slip) nodes.
############################################################################

class NTHalfBBWall(LBNodeType):
    """Half-way bounce-back (no-slip) node.

    The effective location of the wall is half a lattice spacing away from
    the bulk of the fluid. With such a location of the wall,
    this node type provides 2nd order spatial accuracy.

    The idea of a half-way bounce-back node is similar to the full-way
    bounce-back, but the reflection takes only one time step
    :math:`f_{i}^{pre}(x, t+1) = f_{opp(i)}^{post}(x, t)`.

    For non-stationary flows the half-way bounce-back rule is more accurate
    than the full-way bounce-back. For stationary flows there is no difference
    between them as the one step time lag in the full-way bounce-back rule
    does not impact the amount of momentum and mass transferred between
    the fluid and the boundary.

    The wall normal vector is necessary for the half-way bounce-back rule to
    work as only the unknown distributions (those for which there is no data
    streaming from other nodes) at the boundary nodes are replaced.
    """
    wet_node = True
    standard_macro = True
    needs_orientation = True
    link_tags = True
    location = -0.5
    allow_unused = True


class NTFullBBWall(LBNodeType):
    """Full-way bounce-back (no-slip) node.

    Use this node type if you need walls whose normal vector is unknown
    or is not aligned with the Cartesian axes.

    The effective location of the wall is half a lattice spacing between
    the wall node and the fluid node. With such a location of the wall,
    this node type provides 2nd order spatial accuracy.

    Full-way bounce-back works as follows:

    * at time :math:`t` distributions are propagated from the fluid to the
      bounce-back node
    * at time :math:`t+1` the distributions at the bounce-back node are
      reflected across the node center, and then streamed in the standard way

    This can be summarized as :math:`f_{i}^{pre}(x, t+2) = f_{opp(i)}^{post}(x, t)` since it
    takes two time steps for the reflected distributions to reach back the fluid nodes.
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
    link_tags = True
    location = 0.5
    allow_unused = True

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
    """Zou-He density.

    Uses bounce-back of the off-equilibrium distributions.
    """
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
    """Zou-He velocity.

    Uses bounce-back of the off-equilibrium distributions.
    """
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
    standard_macro = True


class NTCopy(LBNodeType):
    """Copies distributions from another node.

    This can be used to implement a crude vanishing gradient
    boundary condition."""
    wet_node = True
    standard_macro = True
    needs_orientation = True

    
class NTExtendedCopy(LBNodeType):
    """Copies distributions from another node.

    This can be used to implement extended periodic 
    boundary condition."""
    wet_node = True
    standard_macro = True
    needs_orientation = True
    
    def __init__(self, transformation=None, orientation=None):
        assert transformation.shape == (4,4), "Invalid shape of transformation array"
        self.trans_sympy = ImmutableDenseMatrix(transformation)
        self.params = {'transformation': self.trans_sympy}
        self.orientation = orientation
        
        
class NTYuOutflow(LBNodeType):
    """Implements the open boundary condition described in:

    Yu, D., Mei, R. and Shyy, W. (2005) 'Improved treatment of
    the open boundary in the method of lattice Boltzmann
    equation', page 5.

    This is an extrapolation based method, using data from next-nearest
    neighbors:

    .. math:: f_i(x_j) = 2 f_i(x_j - n) - f_i(x_j - 2n)
    """
    wet_node = True
    standard_macro = True
    needs_orientation = True


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
        .. math:: \\frac{\partial u}{\partial n (t, x_j)} = \\varphi(t, x_j)
    via:
        .. math:: \phi(t, x_{j0}) = u(t, x_{j1}) + 2 \\varphi (t, x_j)
        .. math:: f_i(t+1, x_{j0}) = f_{\mathrm{\\bar{i}}}^c(t, x_{j0} + c_i) +
                      6 w_i \phi(t, x_{j0}) \cdot c_i
    with:

     * :math:`x_j`: boundary node
     * :math:`x_{j0}`: ghost node
     * :math:`x_{j1}`: fluid node at :math:`x_{j0}` - 2 * normal
     * :math:`c_i`: incoming distributions
     * :math:`\\bar{i}`: direction opposite to i
     * :math:`f_i^c`: distribution after collision (prior to streaming)
    """
    wet_node = True
    standard_macro = True
    needs_orientation = True


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

def get_wet_node_type_ids(allow_unused=None):
    return [id for id, nt in _NODE_TYPES.items() if nt.wet_node and
            (allow_unused is None or nt.allow_unused == allow_unused)]

def get_dry_node_type_ids():
    return [id for id, nt in _NODE_TYPES.items() if not nt.wet_node]

def get_orientation_node_type_ids():
    return [id for id, nt in _NODE_TYPES.items() if nt.needs_orientation]

def get_link_tag_node_type_ids():
    return [id for id, nt in _NODE_TYPES.items() if nt.link_tags]

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
        self.need_mf=self.need_structured_array(params)
        self.data=None
        if self.need_mf:
            self.data=self._get_structured_array(params)
            
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

    def __str__(self):
        return 'DynamicValue(' + ', '.join(str(x) for x in self.params) + ')'

    def __getitem__(self, i):
        return self.params[i]

    def has_symbols(self, *args):
        """Returns True if any of the expressions used for this DynamicValue
        depends on at least one of the symbols provided as arguments."""
        for symbol in args:
            for param in self.params:
                if isinstance(param, expr.Expr) and symbol in param.free_symbols:
                    return True
        return False

    def get_timeseries(self):
        """Returns a generator iterating over all time series used in params."""
        for param in self.params:
            if isinstance(param, LinearlyInterpolatedTimeSeries):
                yield param
                continue
            elif not isinstance(param, expr.Expr):
                continue
            for arg in param.args:
                if isinstance(arg, LinearlyInterpolatedTimeSeries):
                    yield arg
                    
    def need_structured_array(self, params):
        for param in params:
            if isinstance(param, SpatialArray):
                return True
            elif not isinstance(param, expr.Expr):
                continue
            for arg in param.args:
                if isinstance(arg, SpatialArray):
                    return True
        return False
    
    def _get_structured_array(self, params):
        """Returns a structured numpy array with spatial data"""
        data=()
        where=[]
        for param in params:
            if isinstance(param, SpatialArray): 
                data+=(param._data,)
                if where!=[] and where is not None:
                    assert where==param._where
                else:
                    where=param._where.copy()
                continue
            elif is_number(param):
                data+=(param,)
                continue
            elif not isinstance(param, expr.Expr):
                continue
            for arg in param.args:
                if isinstance(arg, SpatialArray):
                    data+=(arg._data,)
                    if where!=[] and where is not None:
                        assert where==arg._where
                    else:
                        where=arg._where.copy()
                    break
            else:
                data+=(1.0,)         
        return multifield(data, where)
                    


class LinearlyInterpolatedTimeSeries(Symbol):
    """A time-dependent scalar data source based on a discrete time series."""

    def __new__(cls, data, step_size=1.0):
        return Symbol.__new__(cls, 'lits%s_%s' % (hashlib.sha1(np.array(data)).hexdigest(),
                                                  step_size))

    def __init__(self, data, step_size=1.0):
        """A continuous scalar data source from a discrete time series.

        Time series data points are linearly interpolated in order to generate
        values for points not present in the time series. The time series is
        wrapped in order to generate an infinite sequence.

        :param data: iterable of scalar values
        :param step_size: number of LB iterations corresponding to one unit in
            the time series (i.e. time distance between two neighboring points).
        """

        Symbol.__init__('unused')
        if type(data) is list or type(data) is tuple:
            data = np.float64(data)

        # Copy here is necessary so that the caller doesn't accidentally change
        # the underlying array later. Also, we need the array to be C-contiguous
        # (for __hash__ below), which might not be the case if it's a view.
        self._data = data.copy()
        self._step_size = step_size

        # To be set later by the geometry encoder class. This is necessary due
        # to how the printing system in Sympy works (see _ccode below).
        self._offset = None

    def __hash__(self):
        return (hash(hashlib.sha1(str(self._step_size).encode('ascii')).digest()) ^
                hash(hashlib.sha1(self._data).digest()))

    def __str__(self):
        return 'LinearlyInterpolatedTimeSeries([%d items], %f)' % (
            self._data.size, self._step_size)

    def __eq__(self, other):
        if not isinstance(other, LinearlyInterpolatedTimeSeries):
            return False
        return np.all(other._data == self._data) and self._step_size == other._step_size

    def _ccode(self, printer):
        assert self._offset is not None
        return 'timeseries_interpolate(%d, %d, %.18f, iteration_number)' % (
            self._offset, self._data.size, self._step_size)

    def data_hash(self):
        """Returns a hash of the underying data series."""
        return hashlib.sha1(self._data).digest()

class SpatialArray(Symbol):
    """A spatial-dependent scalar data source."""

    def __new__(cls, data,index='x', where=None):
        return Symbol.__new__(cls, 'spatarr%s_%s' % (hashlib.sha1(np.array(data)).hexdigest(),
                                                  index)) 

    def __init__(self, data, index='x', where=None):
        """A continuous scalar data source from a discrete time series.

        :param data: iterable of scalar values
        :param index: index of velocity coordinate given by data.
        """

        Symbol.__init__('unused')
       
        if type(data) is list or type(data) is tuple:
            data = np.float64(data)

        self._data = data.copy()
        self._index = index
        self._where = where.copy()
        self._hash = hash(hashlib.sha1(str(self._index).encode('ascii')).digest()) ^ \
                hash(hashlib.sha1(self._data).digest())
        
    def __hash__(self):
        return self._hash

    def __str__(self):
        return 'SpatialArray([%d items], %s)' % (
            self._data.size, self._index)
                
    def __eq__(self, other):
        if not isinstance(other, SpatialArray):
            return False
        return np.all(other._data == self._data) and self._index == other._index

    def _ccode(self, printer):
        return 'spatial_array_%s' % (self._index)

    def data_hash(self):
        """Returns a hash of the underying data series."""
        return hashlib.sha1(self._data).digest()
    
# Maps node type IDs to their classes.
_NODE_TYPES = __init_node_type_list()

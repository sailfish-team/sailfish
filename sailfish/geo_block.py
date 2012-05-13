"""Intra- and inter-block geometry processing."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict, namedtuple
import inspect
import operator
import numpy as np
from sailfish import sym, util
import sailfish.node_type as nt

def span_area(span):
    area = 1
    for elem in span:
        if type(elem) is int:
            continue
        area *= elem.stop - elem.start
    return area


ConnectionPair = namedtuple('ConnectionPair', 'src dst')

# Used for creating connections between subdomains.  Without PBCs,
# virtual == real.  With PBC, the real subdomain is the actual subdomain
# as defined by the simulation geometry, and the virtual subdomain is a
# copy created due to PBC.
SubdomainPair = namedtuple('SubdomainPair', 'real virtual')


def _get_src_slice(b1, b2, slice_axes):
    """Returns slice lists identifying nodes in b1 from which
    information is sent to b2:

    - slices in b1's dist buffer for distribution data
    - slices in the global coordinate system for distribution data
    - slices in a b1's field buffer from which macroscopic data is to be
      collected
    - slices in a b1's field buffer selecting nodes to which data from the
      target subdomain is to be written

    The returned slices correspond to the axes specified in slice_axes.

    :param b1: source SubdomainSpec
    :parma b2: target SubdomainSpec
    :param slice_axes: list of axis numbers identifying axes spanning
        the area of nodes sending information to the target node
    """
    src_slice = []
    src_slice_global = []
    src_macro_slice = []
    dst_macro_slice = []

    for axis in slice_axes:
        # Effective span along the current axis, two versions: including
        # ghost nodes, and including real nodes only.
        b1_ghost_min = b1.location[axis] - b1.envelope_size
        b1_ghost_max = b1.end_location[axis] - 1 + b1.envelope_size
        b1_real_min = b1.location[axis]
        b1_real_max = b1.end_location[axis] - 1

        # Same for the 2nd subdomain.
        b2_real_min = b2.location[axis]
        b2_real_max = b2.end_location[axis] - 1
        b2_ghost_min = b2.location[axis] - b2.envelope_size
        b2_ghost_max = b2.end_location[axis] - 1 + b2.envelope_size

        # Span in global simulation coordinates.  Data is transferred
        # from the ghost nodes to real nodes only.
        global_span_min = max(b1_ghost_min, b2_real_min)
        global_span_max = min(b1_ghost_max, b2_real_max)

        # No overlap, bail out.
        if global_span_min > b1_ghost_max or global_span_max < b1_ghost_min:
            return None, None, None, None

        # For macroscopic fields, the inverse holds true: data is transferred
        # from real nodes to ghost nodes.
        macro_span_min = max(b2_ghost_min, b1_real_min)
        macro_span_max = min(b2_ghost_max, b1_real_max)

        macro_recv_min = max(b1_ghost_min, b2_real_min)
        macro_recv_max = min(b1_ghost_max, b2_real_max)

        src_slice.append(slice(global_span_min - b1_ghost_min,
            global_span_max - b1_ghost_min + 1))
        src_slice_global.append(slice(global_span_min,
            global_span_max + 1))
        src_macro_slice.append(slice(macro_span_min - b1_ghost_min,
            macro_span_max - b1_ghost_min + 1))
        dst_macro_slice.append(slice(macro_recv_min - b1_ghost_min,
            macro_recv_max - b1_ghost_min + 1))

    return src_slice, src_slice_global, src_macro_slice, dst_macro_slice


def _get_dst_full_slice(b1, b2, src_slice_global, full_map, slice_axes):
    """Identifies nodes that transmit full information.

    Returns a tuple of:
    - offset vector in the plane ortogonal to the connection axis in the local
      coordinate system of the destination subdomain (real nodes only)
    - slice selecting part of the buffer (real nodes only) with nodes
      containing information about all distributions
    - same as the previous one, but the slice is for the transfer buffer

    :param b1: source SubdomainSpec
    :param b2: target SubdomainSpec
    :param src_slice_global: list of slices in the global coordinate system
        identifying nodes from which information is sent to the target
        subdomain
    :param full_map: boolean array selecting the source nodes that have the
        full set of distributions to be transferred to the target subdomain
    :param slice_axes: list of axis numbers identifying axes spanning
        the area of nodes sending information to the target node
    """
    dst_slice = []
    dst_full_buf_slice = []
    dst_low = []
    for i, global_pos in enumerate(src_slice_global):
        b2_start = b2.location[slice_axes[i]]
        dst_low.append(global_pos.start - b2_start)

    full_idxs = np.argwhere(full_map)
    if len(full_idxs) > 0:
        # Lowest and highest coordinate along each axis.
        full_min = np.min(full_idxs, 0)
        full_max = np.max(full_idxs, 0)
        # Loop over axes.
        for i, (lo, hi) in enumerate(zip(full_min, full_max)):
            b1_start = b1.location[slice_axes[i]]
            b2_start = b2.location[slice_axes[i]]
            # Offset in the local real coordinate system of the target
            # subdomain (same as dst_low).
            curr_to_dist = src_slice_global[i].start - b2_start
            dst_slice.append(slice(lo + curr_to_dist, hi + 1 + curr_to_dist))
            dst_full_buf_slice.append(slice(lo, hi+1))

    return dst_low, dst_slice, dst_full_buf_slice


def _get_dst_partial_map(dists, grid, src_slice_global, b1, slice_axes,
        conn_axis):
    """Identifies nodes that only transmit partial information.

    :param dists: indices of distributions that point to the target
        subdomain
    :param grid: grid object defining the connectivity of the lattice
    :param src_slice_global: list of slices in the global coordinate system
        identifying nodes from which information is sent to the target
        subdomain
    :param b1: source SubdomainSpec
    :param slice_axes: list of axis numbers identifying axes spanning
        the area of nodes sending information to the target node
    :param conn_axis: axis along which the two subdomains are connected
    """
    # Location of the b1 block in global coordinates (real nodes only).
    min_loc = np.int32([b1.location[ax] for ax in slice_axes])
    max_loc = np.int32([b1.end_location[ax] for ax in slice_axes])

    # Creates an array where the entry at [x,y] is the global coordinate pair
    # corresponding to the node [x,y] in the transfer buffer.
    src_coords = np.mgrid[src_slice_global]
    # [2,x,y] -> [x,y,2]
    src_coords = np.rollaxis(src_coords, 0, len(src_coords.shape))

    last_axis = len(src_coords.shape) - 1

    # Selects source nodes that have the full set of distributions (`dists`).
    full_map = np.ones(src_coords.shape[:-1], dtype=np.bool)

    # Maps distribution index to a boolean array selecting (in src_coords)
    # nodes for which the distribution identified by the key is defined.
    dist_idx_to_dist_map = {}

    for dist_idx in dists:
        # When we follow the basis vector backwards, do we end up at a
        # real (non-ghost) node in the source subdomain?
        basis_vec = list(grid.basis[dist_idx])
        del basis_vec[conn_axis]
        src_block_node = src_coords - basis_vec
        dist_map = np.logical_and(src_block_node >= min_loc,
                                  src_block_node < max_loc)

        dist_map = np.logical_and.reduce(dist_map, last_axis)
        full_map = np.logical_and(full_map, dist_map)
        dist_idx_to_dist_map[dist_idx] = dist_map

    # Maps distribution index to an array of indices (pairs in 3D, single
    # coordinate in 2D) in the local subdomain coordinate system (real nodes).
    dst_partial_map = {}
    buf_min_loc = [span.start for span in src_slice_global]

    for dist_idx, dist_map in dist_idx_to_dist_map.iteritems():
        partial_nodes = src_coords[np.logical_and(dist_map,
                                   np.logical_not(full_map))]
        if len(partial_nodes) > 0:
            partial_nodes -= buf_min_loc
            dst_partial_map[dist_idx] = partial_nodes

    return dst_partial_map, full_map


class LBConnection(object):
    """Container object for detailed data about a directed connection between two
    blocks (at the level of the LB model)."""

    @classmethod
    def make(cls, b1, b2, face, grid):
        """Tries to create an LBCollection between two sudomains.

        If no connection exists, returns None.  Otherwise, returns
        a new instance of LBConnection describing the connection details.

        :param b1: SubdomainSpec for the source subdomain
        :param b2: SubdomainSpec for the target subdomain
        :param face: face ID along which the connection is to be created
        :param grid: grid object defining the connectivity of the lattice
        """
        conn_axis = b1.face_to_axis(face)
        slice_axes = range(0, b1.dim)
        slice_axes.remove(conn_axis)

        src_slice, src_slice_global, src_macro_slice, dst_macro_slice = \
                _get_src_slice(b1, b2, slice_axes)
        if src_slice is None:
            return None

        normal = b1.face_to_normal(face)
        dists = sym.get_interblock_dists(grid, normal)

        dst_partial_map, full_map = _get_dst_partial_map(dists, grid,
                src_slice_global, b1, slice_axes, conn_axis)
        dst_low, dst_slice, dst_full_buf_slice = _get_dst_full_slice(
                b1, b2, src_slice_global, full_map, slice_axes)

        # No full or partial connection means that the topology of the grid
        # is such that there are not distributions pointing to the 2nd block.
        if not dst_slice and not dst_partial_map:
            return None

        return LBConnection(dists, src_slice, dst_low, dst_slice, dst_full_buf_slice,
                dst_partial_map, src_macro_slice, dst_macro_slice, b1.id)

    def __init__(self, dists, src_slice, dst_low, dst_slice, dst_full_buf_slice,
            dst_partial_map, src_macro_slice, dst_macro_slice, src_id):
        """
        In 3D, the order of all slices always follows the natural ordering: x, y, z

        :param dists: indices of distributions to be transferred
        :param src_slice: slice in the full buffer of the source block,
        :param dst_low: position of the buffer in the non-ghost coordinate system of
            the dest block
        :param dst_slice: slice in the non-ghost buffer of the dest block, to which
            full dists can be written
        :param dst_full_buf_slice: slice in the transfer buffer selecting nodes with
            all dists; by definition: size(dst_full_buf_slice) == size(dst_slice)
        :param dst_partial_map: dict mapping distribution indices to lists of positions
            in the transfer buffer
        :param src_macro_slice: slice in a real scalar buffer (including ghost nodes)
            selecting nodes from which field data is to be transferred to the
            target subdomain
        :param dst_macro_slice: slice in a real scalar buffer (including ghost nodes)
            selecting nodes to which field data is to be written when received
            from the target subdomain
        :param src_id: ID of the source block
        """
        self.dists = dists
        self.src_slice = src_slice
        self.dst_low = dst_low
        self.dst_slice = dst_slice
        self.dst_full_buf_slice = dst_full_buf_slice
        self.dst_partial_map = dst_partial_map
        self.src_macro_slice = src_macro_slice
        self.dst_macro_slice = dst_macro_slice
        self.block_id = src_id

    def __eq__(self, other):
        return ((self.dists == other.dists) and
                (self.src_slice == other.src_slice) and
                (self.dst_low == other.dst_low) and
                (self.dst_slice == other.dst_slice) and
                (self.dst_full_buf_slice == other.dst_full_buf_slice) and
                (self.block_id == other.block_id))

    def __ne__(self, other):
        return not self.__eq__(other)

    @property
    def elements(self):
        """Size of the connection buffer in elements."""
        return len(self.dists) * span_area(self.src_slice)

    @property
    def transfer_shape(self):
        """Logical shape of the transfer buffer."""
        return [len(self.dists)] + map(lambda x: int(x.stop - x.start), reversed(self.src_slice))

    @property
    def partial_nodes(self):
        return sum([len(v) for v in self.dst_partial_map.itervalues()])

    @property
    def full_shape(self):
        """Logical shape of the buffer for nodes with a full set of distributions."""
        return [len(self.dists)] + map(lambda x: int(x.stop - x.start), reversed(self.dst_slice))

    @property
    def macro_transfer_shape(self):
        """Logical shape of the transfer buffer for a set of scalar macroscopic
        fields."""
        return map(lambda x: int(x.stop - x.start),
                reversed(self.src_macro_slice))

class SubdomainSpec(object):
    dim = None

    # Face IDs.
    X_LOW = 0
    X_HIGH = 1
    Y_LOW = 2
    Y_HIGH = 3
    Z_LOW = 4
    Z_HIGH = 5

    def __init__(self, location, size, envelope_size=None, id_=None, *args, **kwargs):
        self.location = location
        self.size = size
        # Actual size of the simulation domain, including the envelope (ghost
        # nodes).  This is set later when the envelope size is known.
        self.actual_size = None
        self.envelope_size = envelope_size
        self._runner = None
        self._id = id_
        self._clear_connections()
        self._clear_connectors()

        self.vis_buffer = None
        self.vis_geo_buffer = None
        self._periodicity = [False] * self.dim

    def __repr__(self):
        return '{0}({1}, {2}, id_={3})'.format(self.__class__.__name__,
                self.location, self.size, self._id)

    @property
    def runner(self):
        return self._runner

    @runner.setter
    def runner(self, x):
        self._runner = x

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, x):
        self._id = x

    @property
    def num_nodes(self):
        return reduce(operator.mul, self.size)

    @property
    def periodic_x(self):
        """X-axis periodicity within this block."""
        return self._periodicity[0]

    @property
    def periodic_y(self):
        """Y-axis periodicity within this block."""
        return self._periodicity[1]

    def update_context(self, ctx):
        ctx['dim'] = self.dim
        ctx['envelope_size'] = self.envelope_size
        # TODO(michalj): Fix this.
        # This requires support for ghost nodes in the periodicity code
        # on the GPU.
        # ctx['periodicity'] = self._periodicity
        ctx['periodicity'] = [False, False, False]
        ctx['periodic_x'] = 0 #int(self._block.periodic_x)
        ctx['periodic_y'] = 0 #int(self._block.periodic_y)
        ctx['periodic_z'] = 0 #periodic_z

    def enable_local_periodicity(self, axis):
        """Makes the block locally periodic along a given axis."""
        assert axis <= self.dim-1
        self._periodicity[axis] = True
        # TODO: As an optimization, we could drop the ghost node layer in this
        # case.

    def _add_connection(self, face, cpair):
        if cpair in self._connections[face]:
            return
        self._connections[face].append(cpair)

    def _clear_connections(self):
        self._connections = defaultdict(list)

    def _clear_connectors(self):
        self._connectors = {}

    def add_connector(self, block_id, connector):
        assert block_id not in self._connectors
        self._connectors[block_id] = connector

    def get_connection(self, face, block_id):
        """Returns a LBConnection object describing the connection to 'block_id'
        via 'face'."""
        try:
            for pair in self._connections[face]:
                if pair.dst.block_id == block_id:
                    return pair
        except KeyError:
            pass

    def get_connections(self, face, block_id):
        ret = []
        for pair in self._connections[face]:
            if pair.dst.block_id == block_id:
                ret.append(pair)
        return ret

    def connecting_blocks(self):
        """Returns a list of pairs: (face, block ID) representing connections
        to different blocks."""
        ids = set([])
        for face, v in self._connections.iteritems():
            for pair in v:
                ids.add((face, pair.dst.block_id))
        return list(ids)

    def has_face_conn(self, face):
        return face in self._connections.keys()

    def set_actual_size(self, envelope_size):
        # TODO: It might be possible to optimize this a little by avoiding
        # having buffers on the sides which are not connected to other blocks.
        self.actual_size = [x + 2 * envelope_size for x in self.size]
        self.envelope_size = envelope_size

    def set_vis_buffers(self, vis_buffer, vis_geo_buffer):
        self.vis_buffer = vis_buffer
        self.vis_geo_buffer = vis_geo_buffer

    @classmethod
    def face_to_dir(cls, face):
        if face in (cls.X_LOW, cls.Y_LOW, cls.Z_LOW):
            return -1
        else:
            return 1

    @classmethod
    def face_to_axis(cls, face):
        """Returns the axis number corresponding to a face constant."""
        if face == cls.X_HIGH or face == cls.X_LOW:
            return 0
        elif face == cls.Y_HIGH or face == cls.Y_LOW:
            return 1
        elif face == cls.Z_HIGH or face == cls.Z_LOW:
            return 2

    def face_to_normal(self, face):
        """Returns the normal vector for a face."""
        comp = self.face_to_dir(face)
        pos  = self.face_to_axis(face)
        direction = [0] * self.dim
        direction[pos] = comp
        return direction

    def opposite_face(self, face):
        opp_map = {
            self.X_HIGH: self.X_LOW,
            self.Y_HIGH: self.Y_LOW,
            self.Z_HIGH: self.Z_LOW
        }
        opp_map.update(dict((v, k) for k, v in opp_map.iteritems()))
        return opp_map[face]

    @classmethod
    def axis_dir_to_face(cls, axis, dir_):
        if axis == 0:
            if dir_ == -1:
                return cls.X_LOW
            elif dir_ == 1:
                return cls.X_HIGH
        elif axis == 1:
            if dir_ == -1:
                return cls.Y_LOW
            elif dir_ == 1:
                return cls.Y_HIGH
        elif axis == 2:
            if dir_ == -1:
                return cls.Z_LOW
            elif dir_ == -1:
                return cls.Z_HIGH

    def connect(self, pair, grid=None):
        """Creates a connection between this block and another block.

        A connection can only be created when the blocks are next to each
        other.

        :returns: True if the connection was successful
        :rtype: bool
        """
        # Convenience helper for tests.
        if type(pair) is not SubdomainPair:
            pair = SubdomainPair(pair, pair)

        assert pair.real.id != self.id

        def connect_x(r1, r2, v1, v2):
            c1 = LBConnection.make(v1, v2, self.X_HIGH, grid)
            c2 = LBConnection.make(v2, v1, self.X_LOW, grid)

            if c1 is None:
                return False

            r1._add_connection(self.X_HIGH, ConnectionPair(c1, c2))
            r2._add_connection(self.X_LOW, ConnectionPair(c2, c1))
            return True

        def connect_y(r1, r2, v1, v2):
            c1 = LBConnection.make(v1, v2, self.Y_HIGH, grid)
            c2 = LBConnection.make(v2, v1, self.Y_LOW, grid)

            if c1 is None:
                return False

            r1._add_connection(self.Y_HIGH, ConnectionPair(c1, c2))
            r2._add_connection(self.Y_LOW, ConnectionPair(c2, c1))
            return True

        def connect_z(r1, r2, v1, v2):
            c1 = LBConnection.make(v1, v2, self.Z_HIGH, grid)
            c2 = LBConnection.make(v2, v1, self.Z_LOW, grid)

            if c1 is None:
                return False

            r1._add_connection(self.Z_HIGH, ConnectionPair(c1, c2))
            r2._add_connection(self.Z_LOW, ConnectionPair(c2, c1))
            return True

        if self.ex == pair.virtual.ox:
            return connect_x(self, pair.real, self, pair.virtual)
        elif pair.virtual.ex == self.ox:
            return connect_x(pair.real, self, pair.virtual, self)
        elif self.ey == pair.virtual.oy:
            return connect_y(self, pair.real, self, pair.virtual)
        elif pair.virtual.ey == self.oy:
            return connect_y(pair.real, self, pair.virtual, self)
        elif self.dim == 3:
            if self.ez == pair.virtual.oz:
                return connect_z(self, pair.real, self, pair.virtual)
            elif pair.virtual.ez == self.oz:
                return connect_z(pair.real, self, pair.virtual, self)

        return False

class SubdomainSpec2D(SubdomainSpec):
    dim = 2

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy = location
        self.nx, self.ny = size
        self.ex = self.ox + self.nx
        self.ey = self.oy + self.ny
        self.end_location = [self.ex, self.ey]  # first node outside the block
        SubdomainSpec.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 2-tuple of slice objects that selects all non-ghost nodes."""

        es = self.envelope_size
        return (slice(es, es + self.ny), slice(es, es + self.nx))


class SubdomainSpec3D(SubdomainSpec):
    dim = 3

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy, self.oz = location
        self.nx, self.ny, self.nz = size
        self.ex = self.ox + self.nx
        self.ey = self.oy + self.ny
        self.ez = self.oz + self.nz
        self.end_location = [self.ex, self.ey, self.ez]  # first node outside the block
        self._periodicity = [False, False, False]
        SubdomainSpec.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 3-tuple of slice objects that selects all non-ghost nodes."""
        es = self.envelope_size
        return (slice(es, es + self.nz), slice(es, es + self.ny), slice(es, es + self.nx))

    @property
    def periodic_z(self):
        """Z-axis periodicity within this block."""
        return self._periodicity[2]


class Subdomain(object):
    """Abstract class for the geometry of a SubdomainSpec."""

    NODE_MISC_MASK = 0
    NODE_MISC_SHIFT = 1
    NODE_TYPE_MASK = 2

    @classmethod
    def add_options(cls, group):
        pass

    def __init__(self, grid_shape, block, grid, *args, **kwargs):
        """
        :param grid: grid object specifying the connectivity of the lattice
        """
        self.block = block
        self.grid_shape = grid_shape
        self.grid = grid
        # The type map allocated by the block runner already includes
        # ghost nodes, and is formatted in a way that makes it suitable
        # for copying to the compute device.  The entries in this array are
        # node type IDs.
        self._type_map = block.runner.make_scalar_field(np.uint32, register=False)
        self._type_vis_map = np.zeros(list(reversed(block.size)),
                dtype=np.uint8)
        self._type_map_encoded = False
        self._param_map = block.runner.make_scalar_field(dtype=np.int_,
                register=False)
        self._params = {}
        self._encoder = None
        self._seen_types = set([0])

    @property
    def config(self):
        return self.block.runner.config

    def boundary_conditions(self, *args):
        raise NotImplementedError('boundary_conditions() not defined in a child'
                'class.')

    def initial_conditions(self, sim, *args):
        raise NotImplementedError('initial_conditions() not defined in a child '
                'class')

    def _verify_params(self, where, node_type):
        """Verifies that the node parameters are set correctly."""

        for name, param in node_type.params.iteritems():
            # Single number.
            if util.is_number(param):
                continue
            # Single vector.
            elif type(param) is tuple:
                for el in param:
                    if not util.is_number(el):
                        raise ValueError("Tuple elements have to be numbers.")
            # Field.  If more than a single number is needed per node, this
            # needs to be a numpy record array.
            elif isinstance(param, np.ndarray):
                assert param.size == np.sum(where)
            # TODO(kasiaj): Add support for sympy expressions here for
            # time-dep. boundary conditions.
            else:
                raise ValueError("Unrecognized node param: %s" % name)

    def set_node(self, where, node_type):
        """Set a boundary condition at selected node(s).

        :param where: index expression selecting nodes to set
        :param node_type: LBNodeType subclass or instance
        """
        assert not self._type_map_encoded
        if inspect.isclass(node_type):
            assert issubclass(node_type, nt.LBNodeType)
            node_type = node_type()
        else:
            assert isinstance(node_type, nt.LBNodeType)

        self._verify_params(where, node_type)
        self._type_map[where] = node_type.id
        key = hash((node_type.id, frozenset(node_type.params.items())))
        assert np.all(self._param_map[where] == 0),\
                "Overriding previously set nodes is not allowed."
        self._param_map[where] = key
        self._params[key] = node_type
        self._seen_types.add(node_type.id)

    def reset(self):
        self._type_map_encoded = False
        mgrid = self._get_mgrid()
        self.boundary_conditions(*mgrid)

        # Cache the unencoded type map for visualization.
        self._type_vis_map[:] = self._type_map[:]

        self._postprocess_nodes()
        self._define_ghosts()

        # TODO: At this point, we should decide which GeoEncoder class to use.
        from sailfish import geo_encoder
        self._encoder = geo_encoder.GeoEncoderConst(self)
        self._encoder.prepare_encode(self._type_map.base,
                self._param_map.base, self._params)

    def init_fields(self, sim):
        mgrid = self._get_mgrid()
        self.initial_conditions(sim, *mgrid)

    def update_context(self, ctx):
        assert self._encoder is not None
        self._encoder.update_context(ctx)

    def encoded_map(self):
        if not self._type_map_encoded:
            self._encoder.encode()
            self._type_map_encoded = True

        return self._type_map.base

    def visualization_map(self):
        return self._type_vis_map


class Subdomain2D(Subdomain):
    dim = 2

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.gy, self.gx = grid_shape
        Subdomain.__init__(self, grid_shape, block, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.block.oy:self.block.oy + self.block.ny,
                                 self.block.ox:self.block.ox + self.block.nx])

    def _define_ghosts(self):
        assert not self._type_map_encoded
        es = self.block.envelope_size
        if not es:
            return
        self._type_map.base[0:es, :] = nt._NTGhost.id
        self._type_map.base[:, 0:es] = nt._NTGhost.id
        self._type_map.base[es + self.block.ny:, :] = nt._NTGhost.id
        self._type_map.base[:, es + self.block.nx:] = nt._NTGhost.id

    def _postprocess_nodes(self):
        dry_types = self._type_map.dtype.type(nt.get_dry_node_type_ids())

        # Find nodes which are walls themselves and are completely surrounded by
        # walls.  These nodes are marked as unused, as they do not contribute to
        # the dynamics of the fluid in any way.
        cnt = np.zeros_like(self._type_map.base).astype(np.uint32)
        for i, vec in enumerate(self.grid.basis):
            a = np.roll(self._type_map.base, int(-vec[0]), axis=1)
            a = np.roll(a, int(-vec[1]), axis=0)
            cnt[util.in_anyd(a, dry_types)] += 1

        self._type_map.base[(cnt == self.grid.Q)] = nt._NTUnused.id


class Subdomain3D(Subdomain):
    dim = 3

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.gz, self.gy, self.gx = grid_shape
        Subdomain.__init__(self, grid_shape, block, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.block.oz:self.block.oz + self.block.nz,
                                 self.block.oy:self.block.oy + self.block.ny,
                                 self.block.ox:self.block.ox + self.block.nx])

    def _define_ghosts(self):
        assert not self._type_map_encoded
        es = self.block.envelope_size
        if not es:
            return
        self._type_map.base[0:es, :, :] = nt._NTGhost.id
        self._type_map.base[:, 0:es, :] = nt._NTGhost.id
        self._type_map.base[:, :, 0:es] = nt._NTGhost.id
        self._type_map.base[es + self.block.nz:, :, :] = nt._NTGhost.id
        self._type_map.base[:, es + self.block.ny:, :] = nt._NTGhost.id
        self._type_map.base[:, :, es + self.block.nx:] = nt._NTGhost.id

    def _postprocess_nodes(self):
        dry_types = self._type_map.dtype.type(nt.get_dry_node_type_ids())

        # Find nodes which are walls themselves and are completely surrounded by
        # walls.  These nodes are marked as unused, as they do not contribute to
        # the dynamics of the fluid in any way.
        cnt = np.zeros_like(self._type_map.base).astype(np.uint32)
        for i, vec in enumerate(self.grid.basis):
            a = np.roll(self._type_map.base, int(-vec[0]), axis=2)
            a = np.roll(a, int(-vec[1]), axis=1)
            a = np.roll(a, int(-vec[2]), axis=0)
            cnt[util.in_anyd(a, dry_types)] += 1

        self._type_map.base[(cnt == self.grid.Q)] = nt._NTUnused.id



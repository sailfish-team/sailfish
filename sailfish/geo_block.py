"""Intra- and inter-block geometry processing."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

from collections import defaultdict, namedtuple
import numpy as np
from sailfish import sym

def bit_len(num):
    """Returns the minimal number of bits necesary to encode `num`."""
    length = 0
    while num:
        num >>= 1
        length += 1
    return max(length, 1)

def span_area(span):
    area = 1
    for elem in span:
        if type(elem) is int:
            continue
        area *= elem.stop - elem.start
    return area

ConnectionPair = namedtuple('ConnectionPair', 'src dst')

# TODO(michalj): Add support for data about complementary distributions and
# additional macroscopic variables (necessary for multifluid models).
class LBConnection(object):
    """Container object for detailed data about a directed connection between two
    blocks (at the level of the LB model)."""

    @classmethod
    def make(cls, b1, b2, face, grid):
        """Tries to create an LBCollection for the face 'face' of block b1 to
        block2.  If no connection exists, returns None.  Otherwise, returns
        a new instance of LBConnection describing the connection details.
        """
        dim = b1.dim
        slice_axes = range(0, dim)
        conn_axis = b1.face_to_axis(face)
        slice_axes.remove(conn_axis)

        src_slice = []
        src_slice_global = []

        for axis in slice_axes:
            b1_min = b1.location[axis] - b1.envelope_size
            b1_max = b1.end_location[axis]-1 + b1.envelope_size

            b2_min = b2.location[axis]
            b2_max = b2.end_location[axis]-1

            # In global simulation coordinates.
            global_span_min = max(b1_min, b2_min)
            global_span_max = min(b1_max, b2_max)

            # No overlap, bail out.
            if global_span_min > b1_max or global_span_max < b1_min:
                return None

            src_slice.append(slice(global_span_min - b1_min,
                global_span_max - b1_min+1))
            src_slice_global.append(slice(global_span_min,
                global_span_max+1))

        normal = b1.face_to_normal(face)
        dists = sym.get_interblock_dists(grid, normal)

        # Create an array where the entry at [x,y] is the global coordinate pair
        # corresponding to the node [x,y] in the transfer buffer.
        src_coords = np.mgrid[src_slice_global]
        # [2,x,y] -> [x,y,2]
        src_coords = np.rollaxis(src_coords, 0, len(src_coords.shape))
        min_loc = []
        max_loc = []

        for axis in slice_axes:
            min_loc.append(b1.location[axis])
            max_loc.append(b1.end_location[axis])

        # Location of the b1 block in global coordinates.
        min_loc = np.uint32(min_loc)
        max_loc = np.uint32(max_loc)
        last_axis = len(src_coords.shape)-1
        full_map = np.ones(src_coords.shape[:-1], dtype=np.bool)

        dst_partial_map = {}
        dist_idx_to_dist_map = {}

        for dist_idx in dists:
            basis_vec = list(grid.basis[dist_idx])
            del basis_vec[conn_axis]

            src_block_node = src_coords - basis_vec
            dist_map = np.logical_and(src_block_node >= min_loc,
                                      src_block_node < max_loc)
            dist_map = np.logical_and.reduce(dist_map, last_axis)
            full_map = np.logical_and(full_map, dist_map)
            dist_idx_to_dist_map[dist_idx] = dist_map

        buf_min_loc = []
        for span in src_slice_global:
            buf_min_loc.append(span.start)

        for dist_idx, dist_map in dist_idx_to_dist_map.iteritems():
            partial_nodes = src_coords[np.logical_and(dist_map,
                                            np.logical_not(full_map))]
            if len(partial_nodes) > 0:
                partial_nodes -= buf_min_loc
                dst_partial_map[dist_idx] = partial_nodes

        # Slice selecting part of the buffer with nodes containing information
        # about all distributions.
        dst_slice = []
        dst_full_buf_slice = []
        dst_low = []
        for i, global_pos in enumerate(src_slice_global):
            b2_start = b2.location[slice_axes[i]]
            dst_low.append(global_pos.start - b2_start)

        full_idxs = np.argwhere(full_map)
        if len(full_idxs) > 0:
            full_min = np.min(full_idxs, 0)
            full_max = np.max(full_idxs, 0)
            for i, (lo, hi) in enumerate(zip(full_min, full_max)):
                b1_start = b1.location[slice_axes[i]]
                b2_start = b2.location[slice_axes[i]]
                curr_to_dist = src_slice_global[i].start - b2_start
                dst_slice.append(slice(lo+curr_to_dist, hi+1+curr_to_dist))
                dst_full_buf_slice.append(slice(lo, hi+1))

        # No full or partial connection means that the topology of the grid
        # is such that there are not distributions pointing to the 2nd block.
        if not dst_slice and not dst_partial_map:
            return None

        return LBConnection(dists, src_slice, dst_low, dst_slice, dst_full_buf_slice,
                dst_partial_map, b1.id)

    def __init__(self, dists, src_slice, dst_low, dst_slice, dst_full_buf_slice,
            dst_partial_map, src_id):
        """
        In 3D, the order of all slices always follows the natural ordering: x, y, z

        dists: indices of distributions to be transferred
        src_slice: slice in the full buffer of the source block,
        dst_low: position of the buffer in the non-ghost coordinate system of
            the dest block
        dst_slice: slice in the non-ghost buffer of the dest block, to which
            full dists can be written
        dst_full_buf_slice: slice in the buffer selecting nodes with all dists;
            by definition: size(dst_full_buf_slice) == size(dst_slice)
        dst_partial_map: dict mapping distribution indices to list of positions
            in the transfer buffer
        """
        self.dists = dists
        self.src_slice = src_slice
        self.dst_low = dst_low
        self.dst_slice = dst_slice
        self.dst_full_buf_slice = dst_full_buf_slice
        self.dst_partial_map = dst_partial_map
        self.block_id = src_id

    @property
    def elements(self):
        """Size of the connection buffer in elements."""
        return len(self.dists) * span_area(self.src_slice)

    @property
    def transfer_shape(self):
        """Logical shape of the transfer buffer."""
        return [len(self.dists)] + map(lambda x: x.stop - x.start, reversed(self.src_slice))

    @property
    def partial_nodes(self):
        return sum([len(v) for v in self.dst_partial_map.itervalues()])

    @property
    def full_shape(self):
        return [len(self.dists)] + map(lambda x: x.stop - x.start, reversed(self.dst_slice))

class LBBlock(object):
    dim = None

    # Face IDs.
    _X_LOW = 0
    _X_HIGH = 1
    _Y_LOW = 2
    _Y_HIGH = 3
    _Z_LOW = 4
    _Z_HIGH = 5

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

    def __str__(self):
        return '{0}({1}, {2})'.format(self.__class__.__name__, self.location, self.size)

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

    def enable_local_periodicity(self, axis):
        """Makes the block locally periodic along a given axis."""
        assert axis <= self.dim-1
        self._periodicity[axis] = True
        # TODO: As an optimization, we could drop the ghost node layer in this
        # case.

    def _add_connection(self, face, cpair):
        if face in self._connections:
            for pair in self._connections[face]:
                # We already have a connection for this face.
                # TODO(michalj): Make this an assertion.
                if cpair.dst.block_id == pair.dst.block_id:
                    return
        self._connections.setdefault(face, []).append(cpair)

    def _clear_connections(self):
        self._connections = {}

    def _clear_connectors(self):
        self._connectors = {}

    def add_connector(self, block_id, connector):
        assert block_id not in self._connectors
        self._connectors[block_id] = connector

    def get_connection(self, face, block_id):
        """Returns a LBConnection object describing the connection to 'block_id'
        via 'face'."""
        for pair in self._connections[face]:
            if pair.dst.block_id == block_id:
                return pair

    def connecting_blocks(self):
        """Returns a list of pairs: (face, block ID) representing connections
        to different blocks."""
        ids = []
        for face, v in self._connections.iteritems():
            for pair in v:
                ids.append((face, pair.dst.block_id))
        return ids

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
        if face in (cls._X_LOW, cls._Y_LOW, cls._Z_LOW):
            return -1
        else:
            return 1

    @classmethod
    def face_to_axis(cls, face):
        """Returns the axis number corresponding to a face constant."""
        if face == cls._X_HIGH or face == cls._X_LOW:
            return 0
        elif face == cls._Y_HIGH or face == cls._Y_LOW:
            return 1
        elif face == cls._Z_HIGH or face == cls._Z_LOW:
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
            self._X_HIGH: self._X_LOW,
            self._Y_HIGH: self._Y_LOW,
            self._Z_HIGH: self._Z_LOW
        }
        opp_map.update(dict((v, k) for k, v in opp_map.iteritems()))
        return opp_map[face]

    @classmethod
    def axis_dir_to_face(cls, axis, dir_):
        if axis == 0:
            if dir_ == -1:
                return cls._X_LOW
            elif dir_ == 1:
                return cls._X_HIGH
        elif axis == 1:
            if dir_ == -1:
                return cls._Y_LOW
            elif dir_ == 1:
                return cls._Y_HIGH
        elif axis == 2:
            if dir_ == -1:
                return cls._Z_LOW
            elif dir_ == -1:
                return cls._Z_HIGH

    def connect(self, block, geo=None, axis=None, grid=None):
        """Creates a connection between this block and another block.

        A connection can only be created when the blocks are next to each
        other.

        :param block: block object to connect to
        :param `geo`: None for a local block connection; for a global
               connection due to lattice periodicity, a LBGeometry object
        :returns: True if the connection was successful
        :rtype: bool
        """
        assert block.id != self.id

        def connect_x():
            c1 = LBConnection.make(self, block, self._X_HIGH, grid)
            c2 = LBConnection.make(block, self, self._X_LOW, grid)

            if c1 is None:
                return False

            self._add_connection(self._X_HIGH, ConnectionPair(c1, c2))
            block._add_connection(self._X_LOW, ConnectionPair(c2, c1))
            return True

        def connect_y():
            c1 = LBConnection.make(self, block, self._Y_HIGH, grid)
            c2 = LBConnection.make(block, self, self._Y_LOW, grid)

            if c1 is None:
                return False

            self._add_connection(self._Y_HIGH, ConnectionPair(c1, c2))
            block._add_connection(self._Y_LOW, ConnectionPair(c2, c1))
            return True

        def connect_z():
            c1 = LBConnection.make(self, block, self._Z_HIGH, grid)
            c2 = LBConnection.make(block, self, self._Z_LOW, grid)

            if c1 is None:
                return False

            self._add_connection(self._Z_HIGH, ConnectionPair(c1, c2))
            block._add_connection(self._Z_LOW, ConnectionPair(c2, c1))
            return True

        if geo is not None:
            assert axis is not None
            if axis == 0:
                if self.ox == 0 and block.ex == geo.gx:
                    return block.connect(self, geo, axis, grid)
                elif block.ox == 0 and self.ex == geo.gx:
                    return connect_x()
            elif axis == 1:
                if self.oy == 0 and block.ey == geo.gy:
                    return block.connect(self, geo, axis, grid)
                elif block.oy == 0 and self.ey == geo.gy:
                    return connect_y()
            elif axis == 2:
                if self.oz == 0 and block.ez == geo.gz:
                    return block.connect(self, geo, axis, grid)
                elif block.oz == 0 and self.ez == geo.gz:
                    return connect_z()

        elif self.ex == block.ox:
            return connect_x()
        elif block.ex == self.ox:
            return block.connect(self, grid=grid)
        elif self.ey == block.oy:
            return connect_y()
        elif block.ey == self.oy:
            return block.connect(self, grid=grid)
        elif self.dim == 3:
            if self.ez == block.oz:
                return connect_z()
            elif block.ez == self.oz:
                return block.connect(self, grid=grid)

        return False

class LBBlock2D(LBBlock):
    dim = 2

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy = location
        self.nx, self.ny = size
        self.ex = self.ox + self.nx
        self.ey = self.oy + self.ny
        self.end_location = [self.ex, self.ey]  # first node outside the block
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 2-tuple of slice objects that selects all non-ghost nodes."""

        es = self.envelope_size
        return (slice(es, es + self.ny), slice(es, es + self.nx))


class LBBlock3D(LBBlock):
    dim = 3

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy, self.oz = location
        self.nx, self.ny, self.nz = size
        self.ex = self.ox + self.nx
        self.ey = self.oy + self.ny
        self.ez = self.oz + self.nz
        self.end_location = [self.ex, self.ey, self.ez]  # first node outside the block
        self._periodicity = [False, False, False]
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 3-tuple of slice objects that selects all non-ghost nodes."""
        es = self.envelope_size
        return (slice(es, es + self.nz), slice(es, es + self.ny), slice(es, es + self.nx))

    @property
    def periodic_z(self):
        """Z-axis periodicity within this block."""
        return self._periodicity[2]



class GeoEncoder(object):
    """Takes information about geometry as specified by the simulation and
    encodes it into buffers suitable for processing on a GPU.

    This is an abstract class.  Its implementations provide a specific encoding
    scheme."""
    def __init__(self, geo_block):
        self._type_id_map = {}
        self.geo_block = geo_block

    def encode(self):
        raise NotImplementedError("encode() should be implemented in a subclass")

    def update_context(self, ctx):
        raise NotImplementedError("update_context() should be implemented in a subclass")

    def _type_id(self, node_type):
        if node_type in self._type_id_map:
            return self._type_id_map[node_type]
        else:
            # Does not end with 0xff to make sure the compiler will not complain
            # that x < <val> always evaluates true.
            return 0xfffffffe


class GeoEncoderConst(GeoEncoder):
    """Encodes information about the type, optional parameters and orientation
    of a node into a single uint32.  Optional parameters are stored in const
    memory."""

    def __init__(self, geo_block):
        GeoEncoder.__init__(self, geo_block)

        self._bits_type = 0
        self._bits_param = 0
        self._type_map = None
        self._param_map = None
        self._geo_params = []
        # TODO: Generalize this.
        self._num_velocities = 0
        self.config = geo_block.block.runner.config

    def prepare_encode(self, type_map, param_map, param_dict):
        """
        Args:
          type_map: uint32 array representing node type information
        """
        uniq_types = np.unique(type_map)

        for i, node_type in enumerate(uniq_types):
            self._type_id_map[node_type] = i

        self._bits_type = bit_len(uniq_types.size)
        self._type_map = type_map
        self._param_map = param_map

        # Group parameters by type.
        type_dict = defaultdict(list)
        for param_hash, (node_type, val) in param_dict.iteritems():
            type_dict[node_type].append((param_hash, val))

        max_len = 0
        for node_type, values in type_dict.iteritems():
            l = len(values)
            if node_type == GeoBlock.NODE_VELOCITY:
                self._num_velocities = l
            max_len = max(max_len, l)
        self._bits_param = bit_len(max_len)

        # TODO(michalj): Generalize this to other node types.
        for param_hash, val in type_dict[GeoBlock.NODE_VELOCITY]:
            self._geo_params.extend(val)
        for param_hash, val in type_dict[GeoBlock.NODE_PRESSURE]:
            self._geo_params.append(val)

        self._type_dict = type_dict

    def encode(self):
        assert self._type_map is not None

        # TODO: optimize this using numpy's built-in routines
        param = np.zeros_like(self._type_map)
        for node_type, values in self._type_dict.iteritems():
            for i, (hash_value, _) in enumerate(values):
                param[self._param_map == hash_value] = i

        orientation = np.zeros_like(self._type_map)
        cnt = np.zeros_like(self._type_map)

        for i, vec in enumerate(self.geo_block.grid.basis):
            l = len(list(vec)) - 1
            shifted_map = self._type_map
            for j, shift in enumerate(vec):
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            cnt[(shifted_map == GeoBlock.NODE_WALL)] += 1
            # FIXME: we're currently only processing the primary directions
            # here
            if vec.dot(vec) == 1:
                idx = np.logical_and(self._type_map != GeoBlock.NODE_FLUID,
                        shifted_map == GeoBlock.NODE_FLUID)
                orientation[idx] = self.geo_block.grid.vec_to_dir(list(vec))

            # Mark any nodes completely surrounded by walls as unused.
            self._type_map[(cnt == self.geo_block.grid.Q)] = GeoBlock.NODE_UNUSED

        # Remap type IDs.
        max_type_code = max(self._type_id_map.keys())
        type_choice_map = np.zeros(max_type_code+1, dtype=np.uint32)
        for orig_code, new_code in self._type_id_map.iteritems():
            type_choice_map[orig_code] = new_code

        self._type_map[:] = self._encode_node(orientation, param,
                np.choose(np.int32(self._type_map), type_choice_map))

        # Drop the reference to the map array.
        self._type_map = None

    def update_context(self, ctx):
        ctx.update({
            'geo_fluid': self._type_id(GeoBlock.NODE_FLUID),
            'geo_wall': self._type_id(GeoBlock.NODE_WALL),
            'geo_slip': self._type_id(GeoBlock.NODE_SLIP),
            'geo_unused': self._type_id(GeoBlock.NODE_UNUSED),
            'geo_velocity': self._type_id(GeoBlock.NODE_VELOCITY),
            'geo_pressure': self._type_id(GeoBlock.NODE_PRESSURE),
            'geo_boundary': self._type_id(GeoBlock.NODE_BOUNDARY),
            'geo_ghost': self._type_id(GeoBlock.NODE_GHOST),
            'geo_misc_shift': self._bits_type,
            'geo_type_mask': (1 << self._bits_type) - 1,
            'geo_param_shift': self._bits_param,
            'geo_obj_shift': 0,
            'geo_dir_other': 0,
            'geo_num_velocities': self._num_velocities,
            'geo_params': self._geo_params
        })

    def _encode_node(self, orientation, param, node_type):
        """Encodes information for a single node into a uint32.

        The node code consists of the following bit fields:
          orientation | param_index | node_type
        """
        misc_data = (orientation << self._bits_param) | param
        return node_type | (misc_data << self._bits_type)


# TODO: Implement this class.
class GeoEncoderBuffer(GeoEncoder):
    pass

# TODO: Implement this class.
class GeoEncoderMap(GeoEncoder):
    pass


class GeoBlock(object):
    """Abstract class for the geometry of a LBBlock."""

    # TODO: Deprecate these in favor of BC classes.
    NODE_FLUID = 0
    NODE_WALL = 1
    NODE_VELOCITY = 2
    NODE_PRESSURE = 3
    NODE_BOUNDARY = 4
    NODE_GHOST = 5
    NODE_UNUSED = 6
    NODE_SLIP = 7
    NODE_MISC_MASK = 0
    NODE_MISC_SHIFT = 1
    NODE_TYPE_MASK = 2

    @classmethod
    def add_options(cls, group):
        pass

    def __init__(self, grid_shape, block, grid, *args, **kwargs):
        """
        Args:
          grid: grid object specifying the connectivity of the lattice
        """
        self.block = block
        self.grid_shape = grid_shape
        self.grid = grid
        # The type map allocated by the block runner already includes
        # ghost nodes, and is formatted in a way that makes it suitable
        # for copying to the compute device.
        self._type_map = block.runner.make_scalar_field(np.uint32)
        self._type_vis_map = np.zeros(list(reversed(block.size)),
                dtype=np.uint8)
        self._type_map_encoded = False
        self._param_map = block.runner.make_scalar_field(np.uint32)
        self._params = {}
        self._encoder = None

    @property
    def config(self):
        return self.block.runner.config

    def _define_nodes(self, *args):
        raise NotImplementedError('_define_nodes() not defined in a child'
                'class.')

    def set_geo(self, where, type_, params=None):
        assert not self._type_map_encoded

        # TODO: if type_ is a class, we should just store its ID; if it's
        # an object, the ID should be dynamically assigned
        self._type_map[where] = type_
        key = (type_, params)
        self._param_map[where] = hash(key)
        self._params[hash(key)] = key

    def reset(self):
        self._type_map_encoded = False
        mgrid = self._get_mgrid()
        self._define_nodes(*mgrid)

        # Cache the unencoded type map for visualization.
        self._type_vis_map[:] = self._type_map[:]

        self._postprocess_nodes()
        self._define_ghosts()

        # TODO: At this point, we should decide which GeoEncoder class to use.
        self._encoder = GeoEncoderConst(self)
        self._encoder.prepare_encode(self._type_map.base, self._param_map,
                self._params)

    def init_fields(self, sim):
        mgrid = self._get_mgrid()
        self._init_fields(sim, *mgrid)

    def update_context(self, ctx):
        assert self._encoder is not None

        self._encoder.update_context(ctx)

        # FIXME(michalj)
        ctx.update({
                'bc_wall': 'fullbb',
                'bc_velocity': 'equilibrium',
                'bc_wall_': BCWall,
                'bc_velocity_': BCWall,
                'bc_pressure_': BCWall,
                })

    def encoded_map(self):
        if not self._type_map_encoded:
            self._encoder.encode()
            self._type_map_encoded = True

        return self._type_map.base

    def visualization_map(self):
        return self._type_vis_map


class GeoBlock2D(GeoBlock):
    dim = 2

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.gy, self.gx = grid_shape
        GeoBlock.__init__(self, grid_shape, block, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.block.oy:self.block.oy + self.block.ny,
                                 self.block.ox:self.block.ox + self.block.nx])

    def _define_ghosts(self):
        assert not self._type_map_encoded
        es = self.block.envelope_size
        if not es:
            return
        self._type_map.base[0:es, :] = self.NODE_GHOST
        self._type_map.base[:, 0:es] = self.NODE_GHOST
        self._type_map.base[es + self.block.ny:, :] = self.NODE_GHOST
        self._type_map.base[:, es + self.block.nx:] = self.NODE_GHOST

    def _postprocess_nodes(self):
        # Find nodes which are walls themselves and are completely surrounded by
        # walls.  These nodes are marked as unused, as they do not contribute to
        # the dynamics of the fluid in any way.
        cnt = np.zeros_like(self._type_map.base).astype(np.uint32)
        for i, vec in enumerate(self.grid.basis):
            a = np.roll(self._type_map.base, int(-vec[0]), axis=1)
            a = np.roll(a, int(-vec[1]), axis=0)
            cnt[(a == self.NODE_WALL)] += 1

        self._type_map.base[(cnt == self.grid.Q)] = self.NODE_UNUSED


class GeoBlock3D(GeoBlock):
    dim = 3

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.gz, self.gy, self.gx = grid_shape
        GeoBlock.__init__(self, grid_shape, block, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.block.oz:self.block.oz + self.block.nz,
                                 self.block.oy:self.block.oy + self.block.ny,
                                 self.block.ox:self.block.ox + self.block.nx])

    def _define_ghosts(self):
        assert not self._type_map_encoded
        es = self.block.envelope_size
        if not es:
            return
        self._type_map.base[0:es, :, :] = self.NODE_GHOST
        self._type_map.base[:, 0:es, :] = self.NODE_GHOST
        self._type_map.base[:, :, 0:es] = self.NODE_GHOST
        self._type_map.base[es + self.block.nz:, :, :] = self.NODE_GHOST
        self._type_map.base[:, es + self.block.ny:, :] = self.NODE_GHOST
        self._type_map.base[:, :, es + self.block.nx:] = self.NODE_GHOST

    def _postprocess_nodes(self):
        # Find nodes which are walls themselves and are completely surrounded by
        # walls.  These nodes are marked as unused, as they do not contribute to
        # the dynamics of the fluid in any way.
        cnt = np.zeros_like(self._type_map.base).astype(np.uint32)
        for i, vec in enumerate(self.grid.basis):
            a = np.roll(self._type_map.base, int(-vec[0]), axis=2)
            a = np.roll(a, int(-vec[1]), axis=1)
            a = np.roll(a, int(-vec[2]), axis=0)
            cnt[(a == self.NODE_WALL)] += 1

        self._type_map.base[(cnt == self.grid.Q)] = self.NODE_UNUSED


# TODO: Finish this.
#
# Boundary conditions.
#
class LBBC(object):
    parametrized = False
    wet_nodes = False
    location = 0.0

class BCWall(LBBC):
    pass

class BCHalfBBWall(BCWall):
    wet_nodes = True
    pass

class BCFullBBWall(BCWall):
    pass

class BCSlip(LBBC):
    pass

class BCVelocity(LBBC):
    parametrized = True

class BCPressure(LBBC):
    paremetrized = True



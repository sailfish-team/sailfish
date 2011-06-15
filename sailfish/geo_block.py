__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

from collections import defaultdict
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
        # Corner nodes are 0-elem spans and would multiply by 1 here.
        if elem.start != elem.stop:
            area *= elem.stop - elem.start
    return area

def full_selector_to_face_selector(sel):
    ret = []
    for elem in sel:
        if type(elem) is int:
            continue
        ret.append(elem)
    return ret

# XXX fix this for 3D
def is_corner_span(span):
    for coord in span:
        if type(coord) is slice:
            if coord.start == coord.stop:
                if coord.start == 0:
                    return True, -1
                else:
                    return True, 1
    return False, None


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
        self._id = None
        self._clear_connections()
        self._clear_connectors()

        self.vis_buffer = None
        self.vis_geo_buffer = None
        self.id = id_
        self._periodicity = [False] * self.dim

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

    def connect(self, block):
        """Creates a connection between this block and another block.

        A connection can only be created when the blocks are next to each
        other.

        :param block: block object to connect to
        :returns: True if the connection was successful
        :rtype: bool
        """
        raise NotImplementedError('Method should be defined by subclass.')

    def _add_connection(self, axis, span, block_id):
        if axis in self._connections:
            if (span, block_id) in self._connections[axis]:
                return
        self._connections.setdefault(axis, []).append((span, block_id))

    def _clear_connections(self):
        self._connections = {}

    def _clear_connectors(self):
        self._connectors = {}

    def add_connector(self, block_id, connector):
        assert block_id not in self._connectors
        self._connectors[block_id] = connector

    def get_connection_selector(self, face, block_id):
        """Returns a list of slices/ints which can be used to select
        nodes that propagate data to block 'block_id' via face 'face'"""
        for sel, bid in self._connections[face]:
            if bid == block_id:
                return sel
        return None

    def connecting_blocks(self):
        """Returns a list of pairs: (block IDs, axis) representing connections
        to different blocks."""
        ids = []
        for axis, v in self._connections.iteritems():
            for span, bid in v:
                ids.append((axis, bid))

        return ids

    def set_actual_size(self, envelope_size):
        # TODO: It might be possible to optimize this a little by avoiding
        # having buffers on the sides which are not connected to other blocks.
        self.actual_size = [x + 2 * envelope_size for x in self.size]
        self.envelope_size = envelope_size

    def set_vis_buffers(self, vis_buffer, vis_geo_buffer):
        self.vis_buffer = vis_buffer
        self.vis_geo_buffer = vis_geo_buffer

    def face_to_dir(self, face):
        if face in (self._X_LOW, self._Y_LOW, self._Z_LOW):
            return -1
        else:
            return 1

    def face_to_axis(self, face):
        if face == self._X_HIGH or face == self._X_LOW:
            return 0
        elif face == self._Y_HIGH or face == self._Y_LOW:
            return 1
        elif face == self._Z_HIGH or face == self._Z_LOW:
            return 2

    def opposite_face(self, face):
        opp_map = {
            self._X_HIGH: self._X_LOW,
            self._Y_HIGH: self._Y_LOW,
            self._Z_HIGH: self._Z_LOW
        }
        opp_map.update(dict((v, k) for k, v in opp_map.iteritems()))
        return opp_map[face]

    # XXX, fix this for 3D
    def _direction_from_span_face(self, face, span):
        comp = self.face_to_dir(face)
        pos  = self.face_to_axis(face)
        direction = [0] * self.dim
        direction[pos] = comp

        corner, corner_dir = is_corner_span(span)
        if corner:
            direction[1 - pos] = corner_dir
        return direction

    def connection_dists(self, grid, face, span, opposite=False):
        return sym.get_interblock_dists(grid,
                self._direction_from_span_face(face, span), opposite)

    def connection_buf_size(self, grid, face, block_id=None, span=None):
        if block_id is not None:
            assert span is None
            span = self.get_connection_selector(face, block_id)
        else:
            assert span is not None

        buf_size = len(self.connection_dists(grid, face, span))
        buf_size *= span_area(span)
        return buf_size


class LBBlock2D(LBBlock):
    dim = 2

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy = location
        self.nx, self.ny = size
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 2-tuple of slice objects that selects all non-ghost nodes."""

        es = self.envelope_size
        return (slice(es, es + self.ny), slice(es, es + self.nx))

    def connect(self, block, geo=None, axis=None):
        """Tries to connect the current block to another block, and returns True
        if successful.

        :param `geo`: None for a local block connection; for a global
               connection due to lattice periodicity, a LBGeometry object
        """
        hx = self.nx + self.ox
        hy = self.ny + self.oy

        tg_hx = block.nx + block.ox
        tg_hy = block.ny + block.oy

        assert block.id != self.id

        def get_span(tg_od, sf_od, tg_hd, sf_hd, tg_nd, sf_nd):
            """Calculates the connection slices for both blocks.

            The arguments names follow the scheme:
            tg; target (block)
            sf: self
            hd: max position (hx, hy)
            nd: min position (nx, ny)
            """
            if tg_od < sf_od:
                span_min_tg = sf_od - tg_od
                span_min = 0
            else:
                span_min_tg = 0
                span_min = tg_od - sf_od

            assert span_min >= 0
            assert span_min_tg >= 0

            if sf_hd > tg_hd:
                span_max = tg_hd - sf_od
                span_max_tg = tg_nd
            else:
                span_max = sf_nd
                span_max_tg = sf_hd - tg_od

            return span_min, span_max, span_min_tg, span_max_tg

        def connect_x():
            """Connects the blocks along the X axis.  The wall at the highest X
            coordinate of `self` is connected to the wall at the lowest X
            coordinate of `block`."""
            span_min, span_max, span_min_tg, span_max_tg = get_span(
                block.oy, self.oy, tg_hy, hy, block.ny, self.ny)

            if span_max < span_min or span_max_tg < span_min_tg:
                return False

            span = slice(span_min, span_max)
            span_tg = slice(span_min_tg, span_max_tg)

            self._add_connection(self._X_HIGH, (self.nx-1, span), block.id)
            block._add_connection(self._X_LOW, (0, span_tg), self.id)
            return True

        def connect_y():
            """Connects the blocks along the Y axis.  The wall at the highest Y
            coordinate of `self` is connected to the wall at the lowest Y
            coordinate of `block`."""
            span_min, span_max, span_min_tg, span_max_tg = get_span(
                    block.ox, self.ox, tg_hx, hx, block.nx, self.nx)

            if span_max < span_min or span_max_tg < span_min_tg:
                return False

            span = slice(span_min, span_max)
            span_tg = slice(span_min_tg, span_max_tg)

            self._add_connection(self._Y_HIGH, (span, self.ny-1), block.id)
            block._add_connection(self._Y_LOW, (span_tg, 0), self.id)
            return True

        # Check if a global connection across the simulation domain is
        # requested.
        if geo is not None:
            assert axis is not None

            if axis == 0:
                if self.ox == 0 and tg_hx == geo.gx:
                    return block.connect(self, geo, axis)
                elif block.ox == 0 and hx == geo.gx:
                    return connect_x()
            elif axis == 1:
                if self.oy == 0 and tg_hy == geo.gy:
                    return block.connect(self, geo, axis)
                elif block.oy == 0 and hy == geo.gy:
                    return connect_y()

        elif hx == block.ox:
            return connect_x()
        elif tg_hx == self.ox:
            return block.connect(self)
        elif hy == block.oy:
            return connect_y()
        elif tg_hy == self.oy:
            return block.connect(self)

        return False



class LBBlock3D(LBBlock):
    dim = 3

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy, self.oz = location
        self.nx, self.ny, self.nz = size
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
        self._num_params = 0
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
            self._num_params += l
            if node_type == GeoBlock.NODE_VELOCITY:
                self._num_velocities = l
            max_len = max(max_len, l)
        self._bits_param = bit_len(max_len)

        # TODO(michalj): Genealize this to other node types.
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
                np.choose(self._type_map, type_choice_map))

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
            'num_params': self._num_params,
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
        self._type_map.base[es+self.block.ny:, :] = self.NODE_GHOST
        self._type_map.base[:, es+self.block.nx:] = self.NODE_GHOST

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
        # TODO: actually define ghost nodes here
        raise NotImplementedError('_define_ghosts')

    def _postprocess_nodes(self):
        raise NotImplementedError("_postprocess_nodes()")

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



__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import numpy as np

def bit_len(num):
    """Returns the minimal number of bits necesary to encode `num`."""
    length = 0
    while num:
        num >>= 1
        length += 1
    return max(length, 1)


# TODO: envelope size should be calculated automatically

class LBBlock(object):
    dim = None

    _X_LOW = 0
    _X_HIGH = 1
    _Y_LOW = 2
    _Y_HIGH = 3
    _Z_LOW = 4
    _Z_HIGH = 5

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.location = location
        self.size = size
        self.envelope_size = envelope_size
        # Actual size of the simulation domain, including the envelope (ghost
        # nodes)
        # XXX: calculate that.
        self.actual_size = size
        self._runner = None
        self._id = None
        self._clear_connections()
        self._clear_connectors()

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
        self._connections.setdefault(axis, []).append((span, block_id))

    def _clear_connections(self):
        self._connections = {}

    def _clear_connectors(self):
        self._connectors = {}

    def add_connector(self, block_id, connector):
        self._connectors[block_id] = connector

    def _get_connection_span(self, axis, block_id):
        """Method used for testing."""
        for span, bid in self._connections[axis]:
            if bid == block_id:
                return span
        return None

    def connecting_blocks(self):
        """Returns a list of pairs: (block IDs, axis) representing connections
        to different blocks."""
        ids = []
        for axis, v in self._connections.iteritems():
            for span, bid in v:
                ids.append((axis, bid))

        return ids

    def connection_buf_size(self, axis, block_id):
        raise NotImplementedError('Method should be defined by subclass.')

    def update_context(self, ctx):
        raise NotImplementedError('Method should be defined by subclass.')

class LBBlock2D(LBBlock):
    dim = 2

    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy = location
        self.nx, self.ny = size
        self._periodicity = [False, False]
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 2-tuple of slice objects that selects all non-ghost nodes."""
        return slice(0, None), slice(0, None)

    @property
    def periodic_x(self):
        """X-axis periodicity within this block."""
        return self._periodicity[0]

    @property
    def periodic_y(self):
        """Y-axis periodicity within this block."""
        return self._periodicity[1]

    def enable_local_periodicity(self, axis):
        """Makes the block locally periodic along a given axis."""
        assert axis <= self.dim-1
        self._periodicity[axis] = True

    def connect(self, block, geo=None):
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
            if self.ox == 0 and tg_hx == geo.gx:
                return block.connect(self, geo)
            elif block.ox == 0 and hx == geo.gx:
                return connect_x()
            if self.oy == 0 and tg_hy == geo.gy:
                return block.connect(self, geo)
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

    def connection_buf_size(self, axis, block_id):
        # XXX: This is broken.
        for span, b_id in self._connections[axis]:
            if b_id != block_id:
                continue

            size = 1
            for coord in span:
                if type(coord) is slice:
                    size *= max(1, coord.stop - coord.start)

            # FIXME: This should include ghost nodes.
            return size

        return 0


    def update_context(self, ctx):
        ctx['dim'] = self.dim


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
        return slice(0, None), slice(0, None), slice(0, None)

    def update_context(self, ctx):
        ctx['dim'] = self.dim


class GeoEncoder(object):
    """Takes information about geometry as specified by the simulation and
    encodes it into buffers suitable for processing on a GPU.

    This is an abstract class.  Its implementations provide a specific encoding
    scheme."""
    def __init__(self):
        self._type_id_map = {}

    def encode(self):
        raise NotImplementedError("encode() should be implemented in a subclass")

    def update_context(self, ctx):
        raise NotImplementedError("update_context() should be implemented in a subclass")

    def _type_id(self, node_type):
        if node_type in self._type_id_map:
            return self._type_id_map[node_type]
        else:
            return 0xffffffff


class GeoEncoderConst(GeoEncoder):
    """Encodes information about the type, optional parameters and orientation
    of a node into a single uint32.  Optional parameters are stored in const
    memory."""

    def __init__(self):
        GeoEncoder.__init__(self)

        self._bits_type = 0
        self._bits_orientation = 0
        self._bits_param = 0
        self._type_map = None

    def prepare_encode(self, type_map):
        """
        Args:
          type_map: uint32 array representing node type information
        """
        uniq_types = np.unique(type_map)

        for i, node_type in enumerate(uniq_types):
            self._type_id_map[node_type] = i

        self._bits_type = bit_len(uniq_types.size)
        self._type_map = type_map

    def encode(self):
        assert self._type_map is not None

        # TODO: process params and orientation here.
        param = np.zeros_like(self._type_map)
        orientation = np.zeros_like(self._type_map)

        self._type_map[:] = self._encode_node(orientation, param, self._type_map)

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
            'geo_misc_shift': self._bits_type,
            'geo_type_mask': (1 << self._bits_type) - 1,
            'geo_param_shift': self._bits_param,
            'geo_obj_shift': 0,
            'geo_dir_other': 0,
            'geo_num_velocities': 0,
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
    # TODO: Note: a ghost node should be able to carry information about
    # the normal node type.

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
        self._type_map_encoded = False
        self._type_map_view = self._type_map.view()[block._nonghost_slice]
        self._encoder = None

    def define_nodes(self, *args):
        raise NotImplementedError('define_nodes() not defined in a child'
                'class.')

    def set_geo(self, where, type_, params=None):
        assert not self._type_map_encoded

        # TODO: if type_ is a class, we should just store its ID; if it's
        # an object, the ID should be dynamically assigned
        self._type_map_view[where] = type_

    def reset(self):
        self._type_map_encoded = False
        mgrid = self._get_mgrid()
        self._define_nodes(*mgrid)
        self._define_ghosts()
        self._postprocess_nodes()

        # TODO: At this point, we should decide which GeoEncoder class to use.
        self._encoder = GeoEncoderConst()
        self._encoder.prepare_encode(self._type_map)

    def init_fields(self, sim):
        mgrid = self._get_mgrid()
        self._init_fields(sim, *mgrid)

    def update_context(self, ctx):
        assert self._encoder is not None

        self._encoder.update_context(ctx)

        ctx.update({
                'bc_wall_': BCWall,
                'bc_velocity_': BCWall,
                'bc_pressure_': BCWall,
                })

    def encoded_map(self):
        if not self._type_map_encoded:
            self._encoder.encode()

        return self._type_map

class GeoBlock2D(GeoBlock):
    dim = 2

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.gy, self.gx = grid_shape
        GeoBlock.__init__(self, grid_shape, block, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.block.ox:self.block.oy + self.block.ny,
                                 self.block.oy:self.block.ox + self.block.nx])

    def _define_ghosts(self):
        assert not self._type_map_encoded
        # TODO: actually define ghost nodes here

    def _postprocess_nodes(self):
        # Find nodes which are walls themselves and are completely surrounded by
        # walls.  These nodes are marked as unused, as they do not contribute to
        # the dynamics of the fluid in any way.
        cnt = np.zeros_like(self._type_map).astype(np.uint32)
        for i, vec in enumerate(self.grid.basis):
            a = np.roll(self._type_map, int(-vec[0]), axis=1)
            a = np.roll(a, int(-vec[1]), axis=0)
            cnt[(a == self.NODE_WALL)] += 1

        self._type_map[(cnt == self.grid.Q)] = self.NODE_UNUSED


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



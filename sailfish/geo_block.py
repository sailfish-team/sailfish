__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import numpy as np

# TODO: envelope size should be calculated automatically

class LBBlock(object):
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
        # nodes).
        self.actual_size = size
        self._runner = None
        self._id = None
        self._clear_connections()

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

    def _get_connection_span(self, axis, block_id):
        """Method used for testing."""
        for span, bid in self._connections[axis]:
            if bid == block_id:
                return span
        return None


class LBBlock2D(LBBlock):
    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy = location
        self.nx, self.ny = size
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 2-tuple of slice objects that selects all non-ghost nodes."""
        return slice(0, None), slice(0, None)

    def connect(self, block):
        hx = self.nx + self.ox
        hy = self.ny + self.oy

        tg_hx = block.nx + block.ox
        tg_hy = block.ny + block.oy

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

        if hx == block.ox:
            span_min, span_max, span_min_tg, span_max_tg = get_span(
                    block.oy, self.oy, tg_hy, hy, block.ny, self.ny)

            if span_max < span_min or span_max_tg < span_min_tg:
                return False

            span = slice(span_min, span_max)
            span_tg = slice(span_min_tg, span_max_tg)

            self._add_connection(self._X_HIGH, (self.nx-1, span), block.id)
            block._add_connection(self._X_LOW, (0, span_tg), self.id)
            return True
        elif tg_hx == self.ox:
            return block.connect(self)
        elif hy == block.oy:
            span_min, span_max, span_min_tg, span_max_tg = get_span(
                    block.ox, self.ox, tg_hx, hx, block.nx, self.nx)

            if span_max < span_min or span_max_tg < span_min_tg:
                return False

            span = slice(span_min, span_max)
            span_tg = slice(span_min_tg, span_max_tg)

            self._add_connection(self._Y_HIGH, (span, self.ny-1), block.id)
            block._add_connection(self._Y_LOW, (span_tg, 0), self.id)
            return True
        elif tg_hy == self.oy:
            return block.connect(self)

        return False


class LBBlock3D(LBBlock):
    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy, self.oz = location
        self.nx, self.ny, self.nz = size
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 3-tuple of slice objects that selects all non-ghost nodes."""
        return slice(0, None), slice(0, None), slice(0, None)

    # TODO: implement the connect method


class GeoBlock(object):
    """Abstract class for the geometry of a LBBlock."""

    NODE_GHOST = 0
    # TODO: what happens if the ghost is actually a boundary node?

    @classmethod
    def add_options(cls, group):
        pass

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.block = block
        self.grid_shape = grid_shape
        # The type map allocated by the block runner already includes
        # ghost nodes, and is formatted in a way that makes it suitable
        # for copying to the compute device.
        self._type_map = block.runner.make_int_field()
        self._type_map_encoded = False
        self._type_map_view = self._type_map.view()[block._nonghost_slice]

    def define_nodes(self, *args):
        raise NotImplementedError('define_nodes() not defined in a child'
                'class.')

    def set_geo(self, where, type_, params=None):
        assert not self._type_map_encoded
        self._type_map_view[where] = type_

    def reset(self):
        self._type_map_encoded = False
        mgrid = self._get_mgrid()
        self.define_nodes(*mgrid)
        self._define_ghosts()


        lb_group.add_option('--bc_wall', dest='bc_wall', help='boundary condition implementation to use for wall nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_WALL in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='fullbb')
        lb_group.add_option('--bc_slip', dest='bc_slip', help='boundary condition implementation to use for slip nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_SLIP in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='slipbb')
        lb_group.add_option('--bc_velocity', dest='bc_velocity', help='boundary condition implementation to use for velocity nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_VELOCITY in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='equilibrium')
        lb_group.add_option('--bc_pressure', dest='bc_pressure', help='boundary condition implementation to use for pressure nodes', type='choice',
                choices=[x.name for x in geo.SUPPORTED_BCS if
                    geo.LBMGeo.NODE_PRESSURE in x.supported_types and
                    x.supports_dim(self.geo_class.dim)], default='equilibrium')

class GeoBlock2D(GeoBlock):
    dim = 2

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.gx, self.gy = grid_shape
        GeoBlock.__init__(self, grid_shape, block, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.block.ox:self.block.oy + self.block.ny,
                                 self.block.oy:self.block.ox + self.block.nx])

    def _define_ghosts(self):
        assert not self._type_map_encoded
        # TODO: actually define ghost nodes here


class GeoBlock3D(GeoBlock):
    dim = 3

    def __init__(self, grid_shape, block, *args, **kwargs):
        self.gx, self.gy, self.gz = grid_shape
        GeoBlock.__init__(self, grid_shape, block, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.block.oz:self.block.oz + self.block.nz,
                                 self.block.oy:self.block.oy + self.block.ny,
                                 self.block.ox:self.block.ox + self.block.nx])

    def _define_ghosts(self):
        assert not self._type_map_encoded
        # TODO: actually define ghost nodes here


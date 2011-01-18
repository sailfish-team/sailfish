__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

import numpy as np

# TODO: envelope size should be calculated automatically

class LBBlock(object):
    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.location = location
        self.size = size
        self.envelope_size = envelope_size
        # Actual size of the simulation domain, including the envelope (ghost
        # nodes).
        self.actual_size = size
        self._runner = None
        self._id = None

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

class LBBlock2D(LBBlock):
    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy = location
        self.nx, self.ny = size
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 2-tuple of slice objects that selects all non-ghost nodes."""
        return slice(0, None), slice(0, None)


class LBBlock3D(LBBlock):
    def __init__(self, location, size, envelope_size=None, *args, **kwargs):
        self.ox, self.oy, self.oz = location
        self.nx, self.ny, self.nz = size
        LBBlock.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 3-tuple of slice objects that selects all non-ghost nodes."""
        return slice(0, None), slice(0, None), slice(0, None)


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


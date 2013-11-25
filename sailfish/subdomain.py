"""Intra- and inter-subdomain geometry processing."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict, namedtuple
import ctypes
import inspect
import operator
import multiprocessing as mp
import numpy as np
from scipy.ndimage import filters

from sailfish import util
from sailfish import sym
import sailfish.node_type as nt
from sailfish.subdomain_connection import LBConnection

ConnectionPair = namedtuple('ConnectionPair', 'src dst')

# Used for creating connections between subdomains.  Without PBCs,
# virtual == real.  With PBC, the real subdomain is the actual subdomain
# as defined by the simulation geometry, and the virtual subdomain is a
# copy created due to PBC.
SubdomainPair = namedtuple('SubdomainPair', 'real virtual')


class SubdomainSpec(object):
    """A lightweight class describing the location of a subdomain and its
    connections to other subdomains in the simulation.

    This class does not contain any references to the actual GPU or host data
    structures necessary to run the simulation for this subdomain.
    """
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

        if envelope_size is not None:
            self.set_actual_size(envelope_size)
        else:
            # Actual size of the simulation domain, including the envelope (ghost
            # nodes).  This is set later when the envelope size is known.
            self.actual_size = None
            self.envelope_size = None
        self._runner = None
        self._id = id_
        self._clear_connections()
        self._clear_connectors()

        self.geo_queue = None
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
    def num_actual_nodes(self):
        return reduce(operator.mul, self.actual_size)

    @property
    def periodic_x(self):
        """X-axis periodicity within this subdomain."""
        return self._periodicity[0]

    @property
    def periodic_y(self):
        """Y-axis periodicity within this subdomain."""
        return self._periodicity[1]

    @property
    def periodic(self):
        return any(self._periodicity)

    def update_context(self, ctx):
        ctx['dim'] = self.dim
        # The flux tensor is a symmetric matrix.
        ctx['flux_components'] = self.dim * (self.dim + 1) / 2
        ctx['envelope_size'] = self.envelope_size
        # TODO(michalj): Fix this.
        # This requires support for ghost nodes in the periodicity code
        # on the GPU.
        ctx['periodicity'] = [False, False, False]
        ctx['periodic_x'] = 0 #int(self._block.periodic_x)
        ctx['periodic_y'] = 0 #int(self._block.periodic_y)
        ctx['periodic_z'] = 0 #periodic_z

    def enable_local_periodicity(self, axis):
        """Makes the subdomain locally periodic along a given axis."""
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

    def add_connector(self, subdomain_id, connector):
        assert subdomain_id not in self._connectors
        self._connectors[subdomain_id] = connector

    def get_connection(self, face, subdomain_id):
        """Returns a LBConnection object describing the connection to 'subdomain_id'
        via 'face'."""
        try:
            for pair in self._connections[face]:
                if pair.dst.block_id == subdomain_id:
                    return pair
        except KeyError:
            pass

    def get_connections(self, face, subdomain_id):
        ret = []
        for pair in self._connections[face]:
            if pair.dst.block_id == subdomain_id:
                ret.append(pair)
        return ret

    def connecting_subdomains(self):
        """Returns a list of pairs: (face, subdomain ID) representing connections
        to different subdomains."""
        ids = set([])
        for face, v in self._connections.iteritems():
            for pair in v:
                ids.add((face, pair.dst.block_id))
        return list(ids)

    def has_face_conn(self, face):
        return face in self._connections.keys()

    def set_actual_size(self, envelope_size):
        # TODO: It might be possible to optimize this a little by avoiding
        # having buffers on the sides which are not connected to other subdomains.
        self.actual_size = [x + 2 * envelope_size for x in self.size]
        self.envelope_size = envelope_size

    def init_visualization(self):
        size = reduce(operator.mul, self.size)
        vis_lock = mp.Lock()
        self.vis_buffer = mp.Array(ctypes.c_float, size, lock=vis_lock)
        self.vis_geo_buffer = mp.Array(ctypes.c_uint8, size, lock=vis_lock)
        self.geo_queue = mp.Queue(4096)
        return self.geo_queue

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
        """Creates a connection between this subdomain and another subdomain.

        A connection can only be created when the subdomains are next to each
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
        self.end_location = [self.ex, self.ey]  # first node outside the subdomain
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
        self.end_location = [self.ex, self.ey, self.ez]  # first node outside the subdomain
        self._periodicity = [False, False, False]
        SubdomainSpec.__init__(self, location, size, envelope_size, *args, **kwargs)

    @property
    def _nonghost_slice(self):
        """Returns a 3-tuple of slice objects that selects all non-ghost nodes."""
        es = self.envelope_size
        return (slice(es, es + self.nz), slice(es, es + self.ny), slice(es, es + self.nx))

    @property
    def periodic_z(self):
        """Z-axis periodicity within this subdomain."""
        return self._periodicity[2]


class Subdomain(object):
    """Holds all field and geometry information specific to the subdomain
    described by the corresponding SubdomainSpec. Objects of this class do
    not directly know about the details of memory management on the compute
    device."""

    NODE_MISC_MASK = 0
    NODE_MISC_SHIFT = 1
    NODE_TYPE_MASK = 2

    @classmethod
    def add_options(cls, group):
        pass

    def __init__(self, grid_shape, spec, grid, *args, **kwargs):
        """
        :param grid_shape: size of the lattice for the simulation;
            X dimension is the last element in the tuple
        :param spec: SubdomainSpec for this subdomain
        :param grid: grid object specifying the connectivity of the lattice
        """
        self.spec = spec
        self.grid_shape = grid_shape
        self.grid = grid
        # The type map allocated by the subdomain runner already includes
        # ghost nodes, and is formatted in a way that makes it suitable
        # for copying to the compute device. The entries in this array are
        # node type IDs.
        self._type_vis_map = np.zeros(self.lat_shape, dtype=np.uint8)
        self._type_map_encoded = False
        self._params = {}
        self._encoder = None
        self._seen_types = set([0])
        self._needs_orientation = False
        self.active_node_mask = None

    def allocate(self):
        runner = self.spec.runner
        if self.spec.runner.config.node_addressing == 'indirect':
            self.load_active_node_map()
            self.spec.runner.config.logger.info('Fill ratio is: %0.2f%%' %
                    (self.active_nodes / float(self.spec.num_actual_nodes) * 100))
        self._type_map_ghost, self._sparse_type_map = runner.make_scalar_field(np.uint32, register=False, nonghost_view=False)
        self._type_map = self._type_map_ghost[self.spec._nonghost_slice]
        self._type_map_base = runner.field_base(self._type_map_ghost)
        self._param_map, self._sparse_param_map = runner.make_scalar_field(dtype=np.int_, register=False)
        self._param_map_base = runner.field_base(self._param_map)
        self._orientation, self._sparse_orientation_map = runner.make_scalar_field(np.uint32, register=False)
        self._orientation_base = runner.field_base(self._orientation)

    @property
    def config(self):
        return self.spec.runner.config

    @property
    def lat_shape(self):
        return list(reversed(self.spec.size))

    @property
    def full_lat_shape(self):
        return list(reversed(self.spec.actual_size))

    def boundary_conditions(self, *args):
        raise NotImplementedError('boundary_conditions() not defined in a child'
                ' class.')

    def initial_conditions(self, sim, *args):
        raise NotImplementedError('initial_conditions() not defined in a child '
                'class')

    def load_active_node_map(self):
        """Populates active_node_mask with a dense boolean array filling the area
        described by the corresponding SubdomainSpec. Nodes marked True indicate
        active nodes participating in the simulation."""
        # By default, consider all nodes to be active.
        self.active_node_mask = np.ones(self.full_lat_shape, dtype=np.bool)
        self.spec.runner.config.logger.warning(
            'Using indirect addressing with all nodes active. Consider '
            '--node_addressing=direct for better performance.')

    @property
    def active_nodes(self):
        if self.active_node_mask is not None:
            return long(np.sum(self.active_node_mask))
        else:
            return reduce(operator.mul, self.lat_shape)

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
            # needs to be a numpy record array.  Use node_util.multifield()
            # to create this array easily.
            elif isinstance(param, np.ndarray):
                assert param.size == np.sum(where), ("Your array needs to "
                        "have exactly as many nodes as there are True values "
                        "in the 'where' array.  Use node_util.multifield() to "
                        "generate the array in an easy way.")
            elif isinstance(param, nt.DynamicValue):
                if param.has_symbols(sym.S.time):
                    self.config.time_dependence = True
                if param.has_symbols(sym.S.gx, sym.S.gy, sym.S.gz):
                    self.config.space_dependence = True
                continue
            else:
                raise ValueError("Unrecognized node param: {0} (type {1})".
                        format(name, type(param)))

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
        self._type_map_base[where] = node_type.id
        key = hash((node_type.id, frozenset(node_type.params.items())))
        assert np.all(self._param_map_base[where] == 0),\
                "Overriding previously set nodes is not allowed."
        self._param_map_base[where] = key
        self._params[key] = node_type
        self._seen_types.add(node_type.id)

        if hasattr(node_type, 'orientation') and node_type.orientation is not None:
            self._orientation_base[where] = node_type.orientation
        elif node_type.needs_orientation:
            self._needs_orientation = True

    def update_node(self, where, node_type):
        """Updates a boundary condition at selected node(s).

        Use this method only to update nodes in a _running_ simulation.
        See set_node for a description of params.
        """
        if inspect.isclass(node_type):
            assert issubclass(node_type, nt.LBNodeType)
            node_type = node_type()
        else:
            assert isinstance(node_type, nt.LBNodeType)

        if not self._type_map_encoded:
            raise ValueError('Simulation not started. Use set_node instead.')

        key = hash((node_type.id, frozenset(node_type.params.items())))
        if key not in self._params:
            if node_type.id == 0:
                key = 0
            else:
                raise ValueError('Setting nodes with new parameters is not '
                                 'supported.')

        if node_type.needs_orientation and (not hasattr(node_type, 'orientation')
                                            or node_type.orientation is None):
            raise ValueError('Node orientation not specified.')

        self._type_vis_map[where] = node_type.id
        self._type_map[where] = self._encoder._subdomain_encode_node(
            getattr(node_type, 'orientation', 0),
            node_type.id, key)

    def tag_directions(self):
        """Creates direction tags for nodes that support it.

        Direction tags are a way of summarizing which distributions at a node
        will be undefined after streaming.

        :rvalue: True if there are any nodes supporting tagging, False otherwise
        """
        # For directions which are not periodic, keep the ghost nodes to avoid
        # detecting some missing directions.
        ngs = list(self.spec._nonghost_slice)
        for i, periodic in enumerate(reversed(self.spec._periodicity)):
            if not periodic:
                ngs[i] = slice(None)

        # Limit dry and wet types to these that are actually used in the simulation.
        uniq_types = set(np.unique(self._type_map.base))
        dry_types = list(set(nt.get_dry_node_type_ids()) & uniq_types)
        wet_types = list(set(nt.get_wet_node_type_ids()) & uniq_types)
        orient_types = list(set(nt.get_link_tag_node_type_ids()) & uniq_types)

        if not orient_types:
            return False

        # Convert to a numpy array.
        dry_types = self._type_map.dtype.type(dry_types)
        wet_types = self._type_map.dtype.type(wet_types)
        orient_types = self._type_map.dtype.type(orient_types)
        # Only do direction tagging for nodes that do not have
        # orientation/direction already.
        orient_map = (
            util.in_anyd_fast(self._type_map_base[ngs], orient_types) &
            (self._orientation_base[ngs] == 0))
        l = self.grid.dim - 1
        # Skip the stationary vector.
        for i, vec in enumerate(self.grid.basis[1:]):
            shifted_map = self._type_map_base[ngs]
            for j, shift in enumerate(vec):
                if shift == 0:
                    continue
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            # If the given distribution points to a fluid node, tag it as
            # active.
            idx = orient_map & util.in_anyd_fast(shifted_map, wet_types)
            self._orientation_base[ngs][idx] |= (1 << i)

        return True

    def detect_orientation(self, use_tags):
        # Limit dry and wet types to these that are actually used in the simulation.
        uniq_types = set(np.unique(self._type_map.base))
        dry_types = list(set(nt.get_dry_node_type_ids()) & uniq_types)
        orient_types = list((set(nt.get_orientation_node_type_ids()) -
                             set(nt.get_link_tag_node_type_ids() if use_tags
                                 else [])) & uniq_types)

        if not orient_types:
            return

        # Convert to a numpy array.
        dry_types = self._type_map.dtype.type(dry_types)
        orient_types = self._type_map.dtype.type(orient_types)
        orient_map = util.in_anyd_fast(self._type_map_base, orient_types)
        l = self.grid.dim - 1
        for vec in self.grid.basis:
            # Orientaion only handles the primary directions. More complex
            # setups need link tagging.
            if vec.dot(vec) != 1:
                continue
            shifted_map = self._type_map_base
            for j, shift in enumerate(vec):
                if shift == 0:
                    continue
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            # Only set orientation where it's not already defined (=0).
            idx = orient_map & (shifted_map == 0) & (self._orientation_base == 0)
            self._orientation_base[idx] = self.grid.vec_to_dir(list(vec))

    def reset(self):
        self.config.logger.debug('Setting subdomain geometry...')
        self._type_map_encoded = False

        # Use a coordinate map covering ghost nodes as well. This is
        # necessary so that orientation detection works correctly
        # in case the ghost nodes would correspond to wet nodes from
        # another domain.
        # TODO: When setting nodes on ghosts, do not actually save the
        # node parameters as they will never be used.
        self.boundary_conditions(*self._get_mgrid_base(self.config))
        self.config.logger.debug('... boundary conditions done.')

        have_link_tags = False
        # Defines ghost nodes only where no explicit boundary conditions
        # have been set and where the node does belong to another subdomain.
        # In the last case, the node represents fluid and needs to stay this
        # way for link tagging to work correctly.
        self._define_ghosts(unset_only=True)

        if self._needs_orientation:
            # We do not reset the orientation array here as it is possible to
            # have orientation defined for some nodes and use autodetection for
            # others.
            if self.config.use_link_tags:
                have_link_tags = self.tag_directions()
            self.detect_orientation(self.config.use_link_tags)
            self.config.logger.debug('... orientation done.')

        # Detects unused and propagation-only nodes. Note that this has to take
        # place before ghost nodes are set, as otherwise wall nodes at subdomain
        # boundaries could be marked as unused, i.e.:
        #   G W W W -> G U U W instead of G W U W
        self._postprocess_nodes()
        self.config.logger.debug('... postprocessing done.')
        self._define_ghosts()
        self.config.logger.debug('... ghosts done.')

        # Cache the unencoded type map for visualization.
        self._type_vis_map[:] = self._type_map[:]

        # TODO: At this point, we should decide which GeoEncoder class to use.
        from sailfish import geo_encoder
        self._encoder = geo_encoder.GeoEncoderConst(self)
        self._encoder.prepare_encode(self._type_map_base, self._param_map_base,
                                     self._params, self._orientation_base,
                                     have_link_tags)

        self.config.logger.debug('... encoder done.')

    def get_fo_distributions(self, fo):
        """Computes an array indicating which distributions correspond to
        momentum trasferred from the object to the fluid.

        :param fo: ForceObject description the object to analyze
        :rvalue: dict: distribution index -> N-tuple of arrays indicating
            coordinates of solid object nodes transferring momentum to
            the fluid in the direction corresponding to the distribution
        """
        # Find all non-fluid nodes within the specified bounding box.
        mgrid = self._get_mgrid()
        cond = (self._type_vis_map != nt._NTFluid.id)
        for mx, x0, x1 in zip(mgrid, fo.start, fo.end):
            cond &= (mx >= x0) & (mx <= x1)

        self.config.logger.debug('%s: num solid nodes: %d' %
                                 (fo, np.sum(cond)))

        ret = {}
        l = self.grid.dim - 1
        # Skip the stationary vector.
        for i, vec in enumerate(self.grid.basis[1:]):
            shifted_map = self._type_vis_map.copy()

            for j, shift in enumerate(vec):
                if shift == 0:
                    continue
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            # The current distribution is pointing to a fluid node.
            t = np.where(cond & (shifted_map == nt._NTFluid.id))
            # Only add entries for distributions that actually contribute
            # momentum.
            if t[0].size > 0:
                ret[i + 1] = tuple([x + self.spec.envelope_size for x in t])

        return ret

    @property
    def scratch_space_size(self):
        """Node scratch space size expressed in number of floating point values."""
        return self._encoder.scratch_space_size if self._encoder is not None else 0

    def init_fields(self, sim):
        mgrid = self._get_mgrid()
        self.initial_conditions(sim, *mgrid)

    def update_context(self, ctx):
        assert self._encoder is not None
        self._encoder.update_context(ctx)
        ctx['x_local_device_to_global_offset'] = self.spec.ox - self.spec.envelope_size
        ctx['y_local_device_to_global_offset'] = self.spec.oy - self.spec.envelope_size
        if self.dim == 3:
            ctx['z_local_device_to_global_offset'] = self.spec.oz - self.spec.envelope_size

        ctx['misc_bc_vars'] = []
        for nt_id in self._seen_types:
            node_type = nt._NODE_TYPES[nt_id]
            if hasattr(node_type, 'update_context'):
                node_type.update_context(ctx)

    def encoded_map(self, indirect_address=None):
        if not self._type_map_encoded:
            self._encoder.encode(self._orientation_base)
            self._type_map_encoded = True

        if indirect_address is not None:
            self._sparse_type_map[indirect_address[
                self.active_node_mask]] = self._type_map_ghost[self.active_node_mask]
            return self._sparse_type_map

        return self._type_map_base

    def visualization_map(self):
        """Returns an unencoded type map for visualization/
        postprocessing purposes."""
        return self._type_vis_map

    def fluid_map(self):
        """Returns a boolean array indicating which nodes are "wet" (
        represent fluid and have valid macroscopic fields)."""
        fm = self.visualization_map()
        uniq_types = set(np.unique(fm))
        wet_types = list(set(nt.get_wet_node_type_ids()) & uniq_types)
        wet_types = self._type_map.dtype.type(wet_types)
        return util.in_anyd_fast(fm, wet_types)

    def _fluid_map_base(self):
        assert not self._type_map_encoded

        uniq_types = set(np.unique(self._type_map_base))
        wet_types = list(set(nt.get_wet_node_type_ids()) & uniq_types)
        wet_types = self._type_map.dtype.type(wet_types)
        return util.in_anyd_fast(self._type_map_base, wet_types)


class Subdomain2D(Subdomain):
    dim = 2

    def __init__(self, grid_shape, spec, *args, **kwargs):
        self.gy, self.gx = grid_shape
        Subdomain.__init__(self, grid_shape, spec, *args, **kwargs)

    def _get_mgrid(self):
        """Returns a sequence (in natural order) of indexing arrays for the
        non-ghost slice."""
        return reversed(np.mgrid[self.spec.oy:self.spec.oy + self.spec.ny,
                                 self.spec.ox:self.spec.ox + self.spec.nx])

    def _get_mgrid_base(self, config):
        """Returns a sequence (in natural order) of indexing arrays for the
        base field (including ghosts)."""
        es = self.spec.envelope_size
        ox = self.spec.ox - es
        oy = self.spec.oy - es
        hx, hy = reversed(np.mgrid[oy:oy + self.spec.ny + 2 * es,
                                   ox:ox + self.spec.nx + 2 * es])
        if config.periodic_x:
            hx[hx < 0] += self.gx
            hx[hx >= self.gx] -= self.gx
        if config.periodic_y:
            hy[hy < 0] += self.gy
            hy[hy >= self.gy] -= self.gy
        return hx, hy

    def _define_ghosts(self, unset_only=False):
        assert not self._type_map_encoded
        es = self.spec.envelope_size
        if not es:
            return

        def _slice(slc, face):
            if face in (self.spec.X_LOW, self.spec.X_HIGH):
                return slc[0], slice(None)
            else:
                return slice(None), slc[0]

        def _set(x, face):
            if unset_only:
                # Nodes are fluid.
                tg_map = (x == 0)
                # Nodes do not communicate data to neighboring subdomains.
                for cpair in self.spec._connections.get(face, []):
                    tg_map[_slice(cpair.src.src_slice, face)] = False

                x[tg_map] = nt._NTGhost.id
            else:
                x[:] = nt._NTGhost.id

        _set(self._type_map_base[0:es, :], self.spec.Y_LOW)
        _set(self._type_map_base[:, 0:es], self.spec.X_LOW)
        _set(self._type_map_base[es + self.spec.ny:, :], self.spec.Y_HIGH)
        _set(self._type_map_base[:, es + self.spec.nx:], self.spec.X_HIGH)

    def _postprocess_nodes(self):
        fluid_map = self._fluid_map_base().astype(np.uint8)
        neighbors = np.zeros((3, 3), dtype=np.uint8)
        neighbors[1,1] = 1
        for ei in self.grid.basis:
            neighbors[1 + ei[1], 1 + ei[0]] = 1

        # Any node not connected to at least one wet node is marked unused.
        where = (filters.convolve(fluid_map, neighbors, mode='wrap') == 0)
        self._type_map_base[where] = nt._NTUnused.id

        # If an unused node touches a wet node, mark it as propagation only.
        used_map = (self._type_map_base != nt._NTUnused.id).astype(np.uint8)
        where = (filters.convolve(used_map, neighbors, mode='wrap') > 0)
        self._type_map_base[where & (self._type_map_base == nt._NTUnused.id)] = nt._NTPropagationOnly.id

class Subdomain3D(Subdomain):
    dim = 3

    def __init__(self, grid_shape, spec, *args, **kwargs):
        self.gz, self.gy, self.gx = grid_shape
        Subdomain.__init__(self, grid_shape, spec, *args, **kwargs)

    def _get_mgrid(self):
        return reversed(np.mgrid[self.spec.oz:self.spec.oz + self.spec.nz,
                                 self.spec.oy:self.spec.oy + self.spec.ny,
                                 self.spec.ox:self.spec.ox + self.spec.nx])

    def _get_mgrid_base(self, config):
        """Returns a sequence (in natural order) of indexing arrays for the
        base field (including ghosts)."""
        es = self.spec.envelope_size
        ox = self.spec.ox - es
        oy = self.spec.oy - es
        oz = self.spec.oz - es
        hx, hy, hz = reversed(np.mgrid[oz:oz + self.spec.nz + 2 * es,
                                       oy:oy + self.spec.ny + 2 * es,
                                       ox:ox + self.spec.nx + 2 * es])
        if config.periodic_x:
            hx[hx < 0] += self.gx
            hx[hx >= self.gx] -= self.gx
        if config.periodic_y:
            hy[hy < 0] += self.gy
            hy[hy >= self.gy] -= self.gy
        if config.periodic_z:
            hz[hz < 0] += self.gz
            hz[hz >= self.gz] -= self.gz
        return hx, hy, hz

    def _define_ghosts(self, unset_only=False):
        assert not self._type_map_encoded
        es = self.spec.envelope_size
        if not es:
            return

        def _slice(slc, face):
            if face in (self.spec.X_LOW, self.spec.X_HIGH):
                return slc[1], slc[0], slice(None)
            elif face in (self.spec.Y_LOW, self.spec.Y_HIGH):
                return slc[1], slice(None), slc[0]
            else:
                return slice(None), slc[1], slc[0]

        def _set(x, face):
            if unset_only:
                # Nodes are fluid.
                tg_map = (x == 0)
                # Nodes do not communicate data to neighboring subdomains.
                for cpair in self.spec._connections.get(face, []):
                    tg_map[_slice(cpair.src.src_slice, face)] = False

                x[tg_map] = nt._NTGhost.id
            else:
                x[:] = nt._NTGhost.id

        _set(self._type_map_base[0:es, :, :], self.spec.Z_LOW)
        _set(self._type_map_base[:, 0:es, :], self.spec.Y_LOW)
        _set(self._type_map_base[:, :, 0:es], self.spec.X_LOW)
        _set(self._type_map_base[es + self.spec.nz:, :, :], self.spec.Z_HIGH)
        _set(self._type_map_base[:, es + self.spec.ny:, :], self.spec.Y_HIGH)
        _set(self._type_map_base[:, :, es + self.spec.nx:], self.spec.X_HIGH)

    def _postprocess_nodes(self):
        fluid_map = self._fluid_map_base().astype(np.uint8)
        neighbors = np.zeros((3, 3, 3), dtype=np.uint8)
        neighbors[1,1,1] = 1
        for ei in self.grid.basis:
            neighbors[1 + ei[2], 1 + ei[1], 1 + ei[0]] = 1

        # Any node not connected to at least one wet node is marked unused.
        where = (filters.convolve(fluid_map, neighbors, mode='wrap') == 0)
        self._type_map_base[where] = nt._NTUnused.id

        # If an unused node touches a wet node, mark it as propagation only.
        used_map = (self._type_map_base != nt._NTUnused.id).astype(np.uint8)
        where = (filters.convolve(used_map, neighbors, mode='wrap') > 0)
        self._type_map_base[where & (self._type_map_base == nt._NTUnused.id)] = nt._NTPropagationOnly.id

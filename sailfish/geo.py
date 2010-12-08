import os
import sys
import numpy

from sailfish import sym

# Abstract class implementation, from Peter's Norvig site.
def abstract():
    import inspect
    caller = inspect.getouterframes(inspect.currentframe())[1][3]
    raise NotImplementedError('%s must be implemented in subclass' % caller)

def bitLen(int_type):
    length = 0
    while int_type:
        int_type >>= 1
        length += 1
    return length

class LBMGeo(object):
    """Abstract class for the LBM geometry."""

    #: Dimensionality, needs to be overridden in subclasses.
    dim = 0

    #: Fluid node.
    NODE_FLUID = 0
    #: No-slip boundary condition node.
    NODE_WALL = 1
    #: A node completely surrounded by wall nodes.
    NODE_UNUSED = 2
    #: Velocity boundary condition node.
    NODE_VELOCITY = 3
    #: Pressure boundary condition node.
    NODE_PRESSURE = 4
    #: Boundary nodes of FSI objects.
    NODE_BOUNDARY = 5

    NODE_TYPES = [NODE_FLUID, NODE_WALL, NODE_UNUSED, NODE_VELOCITY,
            NODE_PRESSURE, NODE_BOUNDARY]

    NODE_TYPE_MASK = 0xffffffff
    NODE_MISC_SHIFT = 0
    NODE_MISC_MASK = 0
    NODE_DIR_OTHER = 0

    #: Use :meth:`init_fields` for initial conditions if True, and :meth:`init_dist` otherwise.
    ic_fields = True

    @classmethod
    def _encode_node(cls, misc, type):
        """Encode a node entry for the map of nodes.

        :param orientation: node orientation
        :param type: node type

        :rtype: node code
        """
        return type | (misc << cls.NODE_MISC_SHIFT)

    @classmethod
    def _decode_node_type(cls, code):
        return (code & cls.NODE_TYPE_MASK)

    @classmethod
    def _decode_node_misc(cls, code):
        return ((code & cls.NODE_MISC_MASK) >> cls.NODE_MISC_SHIFT)

    def _encode_orientation_and_param(self, orientation, param):
        return ((orientation << self._param_shift) | param)

    def _decode_orientation_and_param(self, code):
        return (code >> self._param_shift,
                code & ((1 << (self._param_shift+1)) - 1))

    # TODO(mjanusz): This should return a named tuple.
    @classmethod
    def _decode_node(cls, code):
        """Decode an entry from the map of nodes.

        :param code: node code from the map

        :rtype: tuple of misc. data, type
        """
        return cls._decode_node_misc(code), cls._decode_node_type(code)

    def __init__(self, shape, options, float, backend, sim):
        self.sim = sim
        self.shape = shape
        self.options = options
        self.backend = backend
        self.map = sim.make_int_field()
        self.gpu_map = backend.alloc_buf(like=self.map)
        self.float = float
        self.lambda_equilibrium = sym.lambdify_equilibrium(sim)

        self._force_nodes = {}
        self.reset()

    @property
    def dx(self):
        """Lattice spacing in simulation units."""
        return 1.0/min(self.shape)

    @property
    def has_pressure_nodes(self):
        return self._num_pressures > 0

    @property
    def has_velocity_nodes(self):
        return self._num_velocities > 0

    @property
    def fsi_objects(self):
        return self._fsi_objs

    def count_active_nodes(self):
        """Get the number of active nodes in the simulation domain."""
        return numpy.sum(self._decode_node_type(self.map) != self.NODE_UNUSED)

    def define_nodes(self):
        """Define the types of all nodes in the simulation domain.

        Subclasses need to override this method to specify the geometry to be used
        in the simulation.

        Use :meth:`set_geo` and :meth:`fill_geo` to set the type of nodes.  By default,
        all nodes are set as fluid nodes.
        """
        pass

    def init_dist(self, dist):
        """Initialize the particle distributions in the whole simulation domain.

        Subclasses can override this method to provide initial conditions for
        the simulation at the particle distribution level.
        """
        raise NotImplementedError("'init_dist' has not been defined in the geometry class")

    def init_fields(self):
        """Initialize the macroscopic fields in the whole simulation domain.

        Subclasses can override this method to provide initial conditions for
        the simulation at the macroscopic fields level.  The field values will
        later be used to initialize the distributions to their equilibrium
        values.
        """
        pass

    def get_reynolds(self, viscosity):
        """Get the Reynolds number for this geometry."""
        abstract

    def get_defines(self):
        abstract

    def _clear_state(self):
        self._params = None
        self.map[:] = 0
        self._velocity_map = numpy.ma.array(numpy.zeros(shape=([self.dim] + list(self.map.shape)), dtype=self.float),
                mask=True)
        self._pressure_map = numpy.ma.array(numpy.zeros(shape=self.map.shape, dtype=self.float), mask=True)
        self._num_velocities = 0
        self._num_pressures = 0
        self._fsi_objs = []

    def reset(self):
        """Perform a full reset of the geometry."""
        self._clear_state()
        self.define_nodes()
        self._prep_params()
        self._postprocess_nodes()
        self._update_map()

    def get_defines(self):
        return {'geo_fluid': self.NODE_FLUID,
                'geo_wall': self.NODE_WALL,
                'geo_unused': self.NODE_UNUSED,
                'geo_velocity': self.NODE_VELOCITY,
                'geo_pressure': self.NODE_PRESSURE,
                'geo_boundary': self.NODE_BOUNDARY,
                'geo_misc_mask': self.NODE_MISC_MASK,
                'geo_misc_shift': self.NODE_MISC_SHIFT,
                'geo_type_mask': self.NODE_TYPE_MASK,
                'geo_param_shift': self._param_shift,
                'geo_obj_shift': bitLen(len(self.fsi_objects)),
                'geo_dir_other': self.NODE_DIR_OTHER,
                'geo_num_velocities': self._num_velocities,
                }

    def _get_map(self, location):
        """Get a node map entry.

        :param location: a tuple specifying the node location
        """
        return self.map[tuple(reversed(location))]

    def _set_map(self, location, value):
        """Set a node map entry.

        :param location: a tuple specifying the node location
        :param value: the node code as returned by _encode_node()
        """
        self.map[tuple(reversed(location))] = value

    def _update_map(self):
        """Copy the node map to the compute unit."""
        self.backend.to_buf(self.gpu_map)

    def set_geo(self, location, type_, val=None, update=False):
        """Set the type of a grid node.

        :param location: location of the node
        :param type_: type of the node, one of the NODE_* constants
        :param val: optional argument for the node, e.g. the value of velocity or pressure
        :param update: whether to automatically update the geometry for the
            simulation by copying it to the compute unit
        """

        if type(location) is tuple or type(location) is list:
            self._set_map(location, numpy.int32(type_))
            rloc = tuple(reversed(location))
        else:
            self.map[location] = numpy.int32(type_)
            rloc = location

        if val is not None:
            if type_ == self.NODE_VELOCITY:
                if len(val) == self.dim:
                    if type(rloc) is tuple:
                        self._velocity_map[[slice(None)] + list(rloc)] = val
                    else:
                        self._velocity_map[:,rloc] = self.sim.float(val).reshape([self.dim, 1])
                else:
                    raise ValueError('Invalid velocity specified.')
            elif type_ == self.NODE_PRESSURE:
                self._pressure_map[rloc] = val

        if update:
            self._postprocess_nodes(nodes=[location])
            self._update_map()

    def set_geo_from_bool_array(self, array, update=False):
        """Set the geometry for the whole simulation domain using a numpy bool
        array.

        The order of the axes in the array should be [z,],y,x.  The locations
        corresponding to the elements of the array with the value ``True`` will
        be marked as wall (no-slip) nodes, and the ones corresponding to the
        value ``False`` will not be changed.

        :param array: a numpy bool array representing the geometry
        :param update: whether to automatically update the geometry for the
            simulation by copying it to the compute unit
        """
        self.map[array] = numpy.int32(self.NODE_WALL)

        if update:
            self._postprocess_nodes()
            self._update_map()

    def mask_array_by_fluid(self, array):
        """Mask an array so that only fluid nodes are active.

        :param array: a numpy array of the same dimensionality as the simulation domain.
          This will usually be an array containing the macroscopic variables (velocity, density).
        """
        try:
            if get_bc(self.options.bc_wall).wet_nodes:
                return array
            mask = (self._decode_node_type(self.map) == self.NODE_WALL)
            return numpy.ma.array(array, mask=mask)
        except AttributeError:
            return array

    def _prep_array_fill(self, out, location, target, shift=0):
        loc = list(reversed(location))

        if target is None:
            tg = [slice(None)] * len(location)

            # In order for numpy's array broadcasting to work correctly, the dimensions
            # indexed with the ':' operator must be to the right of any dimensions with
            # a specific numerical index, i.e. t[:,:,:] = b[0,0,:] works, but
            # t[:,:,:] = b[0,:,0] does not.
            #
            # To work around this, we swap the axes of the dist array to get the correct
            # form for broadcasting.
            start = None
            for i in reversed(range(0, len(loc))):
                if start is None and loc[i] != slice(None):
                    start = i
                elif start is not None and loc[i] == slice(None):
                    t = loc[start]
                    loc[start] = loc[i]
                    loc[i] = t

                    # shift = +1 if the first dimension is for different velocity directions.
                    out = numpy.swapaxes(out, i+shift, start+shift)
                    start -= 1
        else:
            tg = list(reversed(target))

        return out, loc, tg

    def fill_dist(self, location, dist, target=None):
        """Fill (a part of) the simulation domain with distributions from a specific node(s).

        :param location: location of the node, a n-tuple.  The location can also be a row,
            in which case the coordinate spanning the row should be set to slice(None).
        :param dist: the distribution array
        :param target: if not None, a n-tuple representing the area to which the data from
            the specified node is to be propagated
        """
        out, loc, tg = self._prep_array_fill(dist, location, target, shift=1)

        for i in range(0, len(self.sim.grid.basis)):
            addr = tuple([i] + loc)
            dest = tuple([i] + tg)
            out[dest] = out[addr]

    def fill_geo(self, location, target=None):
        """Fill (a part of) the simulation domain with boundary conditions from a specific node(s).

        See :meth:`fill_dist` for a description of the parameters."""

        out, loc, tg = self._prep_array_fill(self.map, location, target)
        loc = tuple(loc)
        out[tg] = out[loc]

        out, loc, tg = self._prep_array_fill(self._pressure_map, location, target)
        loc = tuple(loc)
        out[tg] = out[loc]

        out, loc, tg = self._prep_array_fill(self._velocity_map, location, target, shift=1)
        for i in range(0, self.dim):
            addr = tuple([i] + loc)
            dest = tuple([i] + tg)
            out[dest] = out[addr]

    def velocity_to_dist(self, location, velocity, dist, rho=1.0):
        """Set the distributions for a node so that the fluid there has a specific velocity.

        This function is used to set the initial conditions for the simulation.

        :param location: location of the node, one of the following:
            1) a n-tuple specifyfing the location
            2) a numpy bool matrix
            3) a slice object
        :param velocity: velocity to set, a n-tuple of floats or numpy arrays
        :param dist: the distribution array
        :param rho: the density to use for the node, a float or a numpy array
        """

        if type(location) is tuple or type(location) is list:
            loc = tuple(reversed(location))
        else:
            loc = location

        for i, lambda_eq in enumerate(self.lambda_equilibrium):
            dist[i][loc] = lambda_eq(rho, *velocity)

        for i, v_component in enumerate(velocity):
            self.sim.velocity[i][loc] = v_component

        self.sim.rho[loc] = rho

    def set_field(self, name, location, value):
        if type(location) is tuple or type(location) is list:
            loc = tuple(reversed(location))
        else:
            loc = location

        fld = getattr(self.sim, name)
        fld[loc] = value

    def _postprocess_nodes(self, nodes=None):
        """Detect types of wall nodes and mark them appropriately.

        :param nodes: optional iterable of locations to postprocess
            If this parameter is used, no orientation detection will be
            performed for the new nodes.
        """
        abstract

    @property
    def params(self):
        return self._params

    def _prep_params(self):
        self._param_map = numpy.zeros(self.map.shape, dtype=numpy.uint32)

        ret = []

        if self.dim == 3:
            v1, i1 = numpy.unique1d(self._velocity_map[0,:,:,:], return_inverse=True)
            v2, i2 = numpy.unique1d(self._velocity_map[1,:,:,:], return_inverse=True)
            v3, i3 = numpy.unique1d(self._velocity_map[2,:,:,:], return_inverse=True)
        else:
            v1, i1 = numpy.unique1d(self._velocity_map[0,:,:], return_inverse=True)
            v2, i2 = numpy.unique1d(self._velocity_map[1,:,:], return_inverse=True)

        # Calculate a single linear index. i1, i2, i3 are arrays of indices for the
        # matrices v1, v2, v3.
        i1m = numpy.max(i1) + 1
        i2m = numpy.max(i2) + 1
        idx = i1 + i1m*i2

        if self.dim == 3:
            idx += i1m*i3*i2m

        vfin, ifin = numpy.unique1d(idx, return_inverse=True)
        ifin = ifin.reshape(self._velocity_map.shape[1:])
        i = 0

        for j, v in enumerate(vfin):
            at = v / i1m
            a1 = v % i1m
            a2 = at % i2m
            a3 = at / i2m

            if (v1[a1] is numpy.ma.masked or v2[a2] is numpy.ma.masked or
                    (self.dim == 3 and v3[a3] is numpy.ma.masked)):
                continue

            midx = (ifin == j)
            ret.extend((v1[a1], v2[a2]))

            if self.dim == 3:
                ret.append(v3[a3])

            self._param_map[midx] = i
            i += 1

        self._num_velocities = i

        pressure, ifin = numpy.unique1d(self._pressure_map, return_inverse=True)
        ifin = ifin.reshape(self._pressure_map.shape)
        i = 0

        for j, v in enumerate(pressure):
            if v is numpy.ma.masked:
                continue

            midx = (ifin == j)
            ret.append(v)
            self._param_map[midx] = i
            i += 1

        self._num_pressures = i
        self._param_shift = bitLen(max(self._num_velocities, self._num_pressures))
        self._params = ret

    # FIXME: This method implicitly assumes that the object can be enclosed in a box
    # which does not intersect any other objects or boundaries.
    def add_force_object(self, obj_id, location, size):
        """Scan the box defined by location and size for nodes and links that cross
        the fluid-solid interface. This function should be called from :meth:`define_nodes`.

        :param obj_id: object ID (any hashable)
        :param location: n-tuple specifying the top corner of the box to scan
        :param size: n-tuple specifying the dimensions of the box to scan
        """
        self._force_nodes[obj_id] = []

        if self.dim == 2:
            mask = numpy.ones_like(self.map)
            mask[location[1]:location[1]+size[1],location[0]:location[0]+size[0]] = 0

            for i, vec in enumerate(self.sim.grid.basis):
                a = numpy.roll(self.map, int(-vec[0]), axis=1)
                a = numpy.roll(a, int(-vec[1]), axis=0)

                b = numpy.logical_and((self.map == self.NODE_FLUID), (a == self.NODE_WALL))
                b = numpy.ma.masked_array(b, mask)

                c = numpy.roll(b, int(vec[0]), axis=1)
                c = numpy.roll(c, int(vec[1]), axis=0)

                self._force_nodes[obj_id].append((numpy.nonzero(b), numpy.nonzero(c)))
        else:
            mask = numpy.ones_like(self.map)
            mask[location[2]:location[2]+size[2],location[1]:location[1]+size[1],location[0]:location[0]+size[0]] = 0

            for i, vec in enumerate(self.sim.grid.basis):
                a = numpy.roll(self.map, int(-vec[0]), axis=2)
                a = numpy.roll(a, int(-vec[1]), axis=1)
                a = numpy.roll(a, int(-vec[2]), axis=0)

                b = numpy.logical_and((self.map == self.NODE_FLUID), (a == self.NODE_WALL))
                b = numpy.ma.masked_array(b, mask)

                c = numpy.roll(b, int(vec[0]), axis=2)
                c = numpy.roll(c, int(vec[1]), axis=1)
                c = numpy.roll(c, int(vec[2]), axis=0)

                # For each diretion, save a map of source (fluid, b array) nodes and
                # target (solid, c array) nodes.
                self._force_nodes[obj_id].append((numpy.nonzero(b), numpy.nonzero(c)))

    def force(self, obj_id, dist):
        """Calculate the force the fluid exerts on a solid object.

        The force is calculated for a time t - \Delta t / 2, given distributions
        at time t.

        To illustrate how the force is calculated, consider the following simplified
        case of momemntum transfer across the boundary (|)::

          t = 0    <-y-- S --x->  |  <-a-- F -b-->
          t = 1    <-a'- S --z->  |  <-c-- F -x'->

        Primes denote quantities after relaxation.  The amount of momentum transferred
        from the fluid node (F) to the solid node (S) is equal to a' - x'.

        :param obj_id: object ID
        :param dist: the distribution array for the current time step

        :rtype: force exterted on the selected object (a n-vector)
        """
        force = numpy.float64([0.0] * self.dim)

        for dir, (fluid_map, solid_map) in enumerate(self._force_nodes[obj_id]):
            force += numpy.float64(list(self.sim.grid.basis[dir])) * (
                    numpy.sum(dist[dir][solid_map]) +
                    numpy.sum(dist[self.sim.grid.idx_opposite[dir]][fluid_map]))
        return force

    def add_fsi_object(self, obj):
        self._fsi_objs.append(obj)


class LBMGeo2D(LBMGeo):
    """Base class for 2D geometries."""

    dim = 2

    NODE_TYPE_MASK = 0x07
    NODE_MISC_SHIFT = 3
    NODE_MISC_MASK = 0xfffffff8

    def __init__(self, shape, *args, **kwargs):
        self.lat_nx, self.lat_ny = shape
        LBMGeo.__init__(self, shape, *args, **kwargs)

    def _get_map(self, location):
        return self.map[location[1], location[0]]

    def _set_map(self, location, value):
        self.map[location[1], location[0]] = value

    def _postprocess_nodes(self, nodes=None):
        lat_nx, lat_ny = self.shape

        if nodes is None:
            # Detect unused nodes.
            cnt = numpy.zeros_like(self.map).astype(numpy.int32)
            orientation = numpy.empty(shape=self.map.shape, dtype=numpy.int32)
            orientation[:] = self.NODE_DIR_OTHER

            for i, vec in enumerate(self.sim.grid.basis):
                a = numpy.roll(self.map, int(-vec[0]), axis=1)
                a = numpy.roll(a, int(-vec[1]), axis=0)

                cnt[(a == self.NODE_WALL)] += 1

                # This will not work correctly on domain boundaries, if they are not
                # periodic.
                if vec.dot(vec) == 1:
                    orientation[
                            numpy.logical_and(self.map != self.NODE_FLUID,
                                a == self.NODE_FLUID)] = self.sim.grid.vec_to_dir(list(vec))

            self.map[(cnt == self.sim.grid.Q)] = self.NODE_UNUSED

            # Postprocess the whole domain here.
            self.map[:] = self._encode_node(
                    self._encode_orientation_and_param(orientation, self._param_map),
                    self.map)

        else:
            nodes_ = nodes

            for loc in nodes_:
                cnode_type = self._get_map(loc)

                if cnode_type != self.NODE_FLUID:
                    self._set_map(loc, self._encode_node(
                            self._encode_orientation_and_param(self.NODE_DIR_OTHER,
                                    self._param_map[tuple(reversed(loc))]),
                        cnode_type))

class LBMGeo3D(LBMGeo):
    """Base class for 3D geometries."""

    dim = 3

    NODE_TYPE_MASK = 0x07
    NODE_MISC_SHIFT = 3
    NODE_MISC_MASK = 0xfffffff8

    def __init__(self, shape, *args, **kwargs):
        self.lat_nx, self.lat_ny, self.lat_nz = shape
        LBMGeo.__init__(self, shape, *args, **kwargs)

    def _get_map(self, location):
        return self.map[location[2], location[1], location[0]]

    def _set_map(self, location, value):
        self.map[location[2], location[1], location[0]] = value

    def _postprocess_nodes(self, nodes=None):
        lat_nx, lat_ny, lat_nz = self.shape

        if nodes is None:
            # Detect unused nodes.
            cnt = numpy.zeros_like(self.map).astype(numpy.int32)
            orientation = numpy.empty(shape=self.map.shape, dtype=numpy.int32)
            orientation[:] = self.NODE_DIR_OTHER

            for i, vec in enumerate(self.sim.grid.basis):
                a = numpy.roll(self.map, int(-vec[0]), axis=2)
                a = numpy.roll(a, int(-vec[1]), axis=1)
                a = numpy.roll(a, int(-vec[2]), axis=0)

                cnt[(a == self.NODE_WALL)] += 1

                # FIXME: Only process the primary 6 directions for now.
                # This will not work correctly on domain boundaries, if they are not
                # periodic.
                if vec.dot(vec) == 1:
                    orientation[
                            numpy.logical_and(self.map != self.NODE_FLUID,
                                a == self.NODE_FLUID)] = self.sim.grid.vec_to_dir(list(vec))

            self.map[(cnt == self.sim.grid.Q)] = self.NODE_UNUSED

            # Postprocess the whole domain here.
            self.map[:] = self._encode_node(
                    self._encode_orientation_and_param(orientation, self._param_map),
                    self.map)
        else:
            nodes_ = nodes

            for loc in nodes_:
                cnode_type = self._get_map(loc)

                if cnode_type != self.NODE_FLUID:
                    self._set_map(loc, self._encode_node(
                            self._encode_orientation_and_param(self.NODE_DIR_OTHER,
                                    self._param_map[tuple(reversed(loc))]),
                        cnode_type))

#
# Boundary conditions
#
class LBMBC(object):
    """Generic boundary condition class."""
    def __init__(self, name, supported_types=set(LBMGeo.NODE_TYPES), dims=set([2,3]), location=0.0, wet_nodes=False):
        """
        :param name: a string representing the boundary condition
        :param location: location of the boundary; if 0.0, the boundary is exactly at the node; otherwise,
            the boundary is located at 'location' * normal vector (pointing into the fluid domain)
            away from the node
        :param wet_nodes: if ``True``, the boundary condition nodes represent fluid particles
            and undergo standard collisions
        """
        self.name = name
        self.location = location
        self.wet_nodes = wet_nodes
        self.supported_types = supported_types
        self.supported_dims = dims

    def supports_dim(self, dim):
        return dim in self.supported_dims

def get_bc(type_):
    return BCS_MAP[type_]

SUPPORTED_BCS = [LBMBC('fullbb', location=0.5, supported_types=set([LBMGeo.NODE_WALL, LBMGeo.NODE_VELOCITY])),
                 LBMBC('halfbb', location=-0.5, wet_nodes=True, supported_types=set([LBMGeo.NODE_WALL])),
                 LBMBC('equilibrium', supported_types=set([LBMGeo.NODE_VELOCITY, LBMGeo.NODE_PRESSURE])),
                 LBMBC('zouhe', wet_nodes=True, supported_types=set([LBMGeo.NODE_WALL, LBMGeo.NODE_VELOCITY,
                     LBMGeo.NODE_PRESSURE]))
                 ]

BCS_MAP = dict((x.name, x) for x in SUPPORTED_BCS)


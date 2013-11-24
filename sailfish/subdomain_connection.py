"""
Code for managing the data connection between two subdomains.

The sketches below illustrate sample subdomain configurations (in 2D) and the
the meaning of different variables used in the Sailfish data structures.

Case A:
==============================================================================

global coordinate system:
       0         1         2         3        4         5
 src_slice_global               |---------|


subdomain full (w/ ghosts) coordinate systems:
(the left-most ghost layer for the 1st subdomain is omitted)

subdomain 1: loc: 1, size: 2

          src_slice AB --------------+
          src_slice AA ----+         |
       src_macro_slice     |         |
                           |         |
                      |----+----|    |
                                |----+----|
                 0         1         2
            +-----------------------------+
            |    F         F    |    G    |
            | SE S SW | SE S SW | .. . SW |
            +-------------------+         +
            |    G         G         G    |
            | SE S SW | .. S SW | .. . SW |
            +-----------------------------+
                                |
full coordinate system:    0    |    1         2         3
                      +---------+-----------------------------+
                    0 |    G         G         G         G    |
                      | .. . SW | .. S SW | SE S SW | SE S SW |
                      +         +-----------------------------+
                    1 |    G    |    F         F         F    |
                      | .. . SW | SE S SW | SE S SW | SE S SW |
                      +---------+-----------------------------+
                                     0         1         2        (non-ghost)
                                     |
                      |----+----|    +-- dst_low
                           |
                           |
        dst_macro_slice ---+

subdomain 2: loc: 3, size: 3
data transfer: 1->2 (SW only):
 - AB: (1, 3) -> (1, 1)
 - AA: (0, 2) -> (0, 0)

dst_low: 3 (global) - 3 (location of 2nd subdomain) = 0
dst_slice: None
dst_partial_map: {SW -> 0}
transfer buffer: [ x x SW ]

Case B:
==============================================================================

global coordinate system:
       0         1         2         3        4         5
 src_slice_global:
           |--------------------------------------------------|

subdomain full (w/ ghosts) coordinate systems:

subdomain 1: loc: 1, size: 3
          src_slice_AB ----------------------------------+
          src_slice AA --------------+                   |
       src_macro_slice               |                   |
                                     |                   |
                      |--------------+--------------|    |
            |--------------------------------------------+----|
                 0         1         2         3         4
            +-------------------------------------------------+
            |    G    |    F         F         F    |    G    |
            | SE . .. | SE S SW | SE S SW | SE S SW | .. . SW |
            +         +-----------------------------+         |
            |    G         G         G         G         G    |
            | SE . .. | SE S .. | SE S SW | .. S SW | .. . SW |
            +-------------------------------------------------+
                           0          1        2

        dst_macro_slice -------------+
                                     |
                      |--------------+--------------|
       0         1         2         3         4         5         6
  +---------------------------------------------------------------------+
  |    G         G         G         G         G         G         G    |
  | .. . SW | .. S SW | SE S SW | SE S SW | SE S SW | SE S .. | SE . .. |
  +         +-------------------------------------------------+         +
  |    G    |    F         F         F         F         F    |    G    |
  | .. . SW | SE S SW | SE S SW | SE S SW | SE S SW | SE S SW | SE . .. |
  +---------------------------------------------------------------------+
                 0         1         2         3         4
                 |              |----+----|
                 |                   |
      dst_low ---+      dst_slice ---+

subdomain 2: loc: 0, size: 5

transfer buffer (AA): | SE S SW | SE S SW | SE S SW |
transfer buffer (AB):
                 0         1         2         3         4
            | SE x xx | SE S xx | SE S SW | xx S SW | xx x SW |
                                |----+----|
                                     |
               dst_full_buf_slice ---+

dst_partial_map:
 SE: 0, 1
 S: 1, 3
 SW: 3, 4
"""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import numpy as np
from sailfish import sym

class MacroConnectionBuffer(object):
    """Contains buffers needed to transfer a single macroscopic field
    between two subdomains."""

    def __init__(self, face, cpair, coll_buf, coll_idx, recv_buf, dist_idx,
            field):
        """
        :param face: face ID
        :param cpair: ConnectionPair, a tuple of two LBConnection objects
        :param coll_buf: GPUBuffer for information collected on the source
            subdomain to be sent to the destination subdomain
        :param coll_idx: GPUBuffer with an array of indices indicating sparse
            nodes from which data is to be collected; this is only used for
            connections via the X_LOW and X_HIGH faces
        :param recv_buf: GPUBuffer for information received from the remote
            subdomain
        :param dist_idx: GPUBuffer with an array of indices indicating
            the position of nodes to which data is to be distributed; this is
            only used for connections via the X_LOW and X_HIGH faces
        :param field: ScalarField that is being transferred using this buffer
        """
        self.face = face
        self.cpair = cpair
        self.coll_buf = coll_buf
        self.coll_idx = coll_idx
        self.recv_buf = recv_buf
        self.dist_idx = dist_idx
        self.field = field

class ConnectionBuffer(object):
    """Contains buffers needed to transfer distributions between two
    subdomains."""

    # TODO: Use a single buffer for dist_full_buf and dist_local_full_buf.
    def __init__(self, face, cpair, coll_buf, coll_idx, recv_buf,
            dist_partial_buf, dist_partial_idx, dist_partial_sel,
            dist_full_buf, dist_full_idx, grid_id=0,
            coll_idx_opposite=None,
            dist_full_idx_opposite=None,
            local_coll_buf=None,
            local_recv_buf=None):
        """
        :param face: face ID
        :param cpair: ConnectionPair, a tuple of two LBConnection objects
        :param coll_buf: GPUBuffer for information collected on the source
            subdomain to be sent to the destination subdomain
        :param coll_idx: GPUBuffer with an array of indices indicating sparse
            nodes from which data is to be collected; this is only used for
            connections via the X_LOW and X_HIGH faces
        :param recv_buf: page-locked numpy buffer with information
            received from the remote subdomain
        :param dist_partial_buf: GPUBuffer, for partial distributions
        :param dist_partial_idx: GPUBuffer, for indices of the partial distributions
        :param dist_partial_sel: selector in recv_buf to get the partial
            distributions
        :param dist_full_buf: GPUBuffer for information to be distributed; this
            is used for nodes where a complete set of distributions is available
            (e.g. not corner or edge nodes)
        :param dist_full_idx: GPUBuffer with an array of indices indicating
            the position of nodes to which data is to be distributed; this is
            only used for connections via the X_LOW and X_HIGH faces
        :param grid_id: grid ID
        :param coll_idx_opposite: like coll_idx, buf for the fully local step of
            the AA access pattern
        :param dist_full_idx_opposite: like dist_full_idx, but for the fully
            local step of the AA access pattern
        :param local_coll_buf: like coll_buf, but for the fully local step of
            the AA access pattern
        :param local_recv_buf: like recv_buf, but for the fully local step of
            the AA access pattern
        """
        self.face = face
        self.cpair = cpair
        self.coll_buf = coll_buf
        self.coll_idx = coll_idx
        self.recv_buf = recv_buf
        self.dist_partial_buf = dist_partial_buf
        self.dist_partial_idx = dist_partial_idx
        self.dist_partial_sel = dist_partial_sel
        self.dist_full_buf = dist_full_buf
        self.dist_full_idx = dist_full_idx
        self.grid_id = grid_id
        self.coll_idx_opposite = coll_idx_opposite
        self.dist_full_idx_opposite = dist_full_idx_opposite
        self.local_coll_buf = local_coll_buf
        self.local_recv_buf = local_recv_buf

    def distribute(self, backend, stream):
        # Serialize partial distributions into a contiguous buffer.
        if self.dist_partial_sel is not None:
            self.dist_partial_buf.host[:] = self.recv_buf[self.dist_partial_sel]
            backend.to_buf_async(self.dist_partial_buf.gpu, stream)

        if self.cpair.dst.dst_slice:
            slc = [slice(0, self.recv_buf.shape[0])] + list(
                    reversed(self.cpair.dst.dst_full_buf_slice))
            self.dist_full_buf.host[:] = self.recv_buf[slc]
            backend.to_buf_async(self.dist_full_buf.gpu, stream)

    def distribute_unpropagated(self, backend, stream):
        backend.to_buf_async(self.local_recv_buf.gpu, stream)


def span_area(span):
    area = 1
    for elem in span:
        if type(elem) is int:
            continue
        area *= elem.stop - elem.start
    return area


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
        basis_vec = [int(x) for x in grid.basis[dist_idx]]
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
    subdomains (at the level of the LB model)."""

    @classmethod
    def make(cls, b1, b2, face, grid):
        """Tries to create an LBCollection between two subdomains.

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
        # is such that there are no distributions pointing to the 2nd subdomain.
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
        :param dst_low: position of the transfer buffer in the non-ghost coordinate
            system of the dest block
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

    def __str__(self):
        return 'LBConnection(dists={0}, src_slice={1}, dst_low={2}, dst_slice={3}, dst_full_buf_slice={4}, dst_partial_map={5}, src_macro_slice={6}, dst_macro_slice={7}, src_id={8})'.format(
                self.dists, self.src_slice, self.dst_low, self.dst_slice, self.dst_full_buf_slice,
                self.dst_partial_map, self.src_macro_slice, self.dst_macro_slice, self.block_id)

    @property
    def elements(self):
        """Size of the connection buffer in elements."""
        return len(self.dists) * span_area(self.src_slice)

    @property
    def transfer_shape(self):
        """Logical shape of the transfer buffer."""
        return [len(self.dists)] + map(lambda x: int(x.stop - x.start), reversed(self.src_slice))

    @property
    def local_transfer_shape(self):
        """Logical shape of the transfer buffer for the fully local step in the AA access pattern."""
        return [len(self.dists)] + map(lambda x: int(x.stop - x.start), reversed(self.src_macro_slice))

    @property
    def partial_nodes(self):
        return sum([len(v) for v in self.dst_partial_map.itervalues()])

    @property
    def full_shape(self):
        """Logical shape of the buffer for nodes with a full set of distributions."""
        return [len(self.dists)] + map(lambda x: int(x.stop - x.start), reversed(self.dst_slice))

    @property
    def full_local_shape(self):
        """Logical shape of the buffer for nodes with a full set of distributions
        for the fully local step in the AA access pattern."""
        return [len(self.dists)] + map(lambda x: int(x.stop - x.start),
            reversed(self.dst_macro_slice))

    @property
    def macro_transfer_shape(self):
        """Logical shape of the transfer buffer for a set of scalar macroscopic
        fields."""
        return map(lambda x: int(x.stop - x.start),
                reversed(self.src_macro_slice))



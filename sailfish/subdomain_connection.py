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


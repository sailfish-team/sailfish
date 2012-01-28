"""Classes to specify global LB simulation geometry and its partitions."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'


from sailfish.geo_block import SubdomainSpec2D, SubdomainSpec3D


class LBGeometry(object):
    """Describes the high-level geometry of a LB simulation."""

    def __init__(self, config):
        self.config = config
        self.gx = config.lat_nx
        self.gy = config.lat_ny
        self.gsize = [self.gx, self.gy]

class LBGeometry2D(LBGeometry):
    """Describes the high-level 2D geometry of a LB simulation."""

    @classmethod
    def add_options(cls, group):
        group.add_argument('--lat_nx', help='lattice width', type=int,
                default=0)
        group.add_argument('--lat_ny', help='lattice height', type=int,
                default=0)
        group.add_argument('--periodic_x', dest='periodic_x',
                help='make the lattice periodic in the X direction',
                action='store_true', default=False)
        group.add_argument('--periodic_y', dest='periodic_y',
                help='make the lattice periodic in the Y direction',
                action='store_true', default=False)

    def blocks(self):
        """Returns a 1-element list containing a single 2D block
        covering the whole domain."""
        return [SubdomainSpec2D((0, 0), (self.config.lat_nx,
                                   self.config.lat_ny))]

class LBGeometry3D(LBGeometry):
    """Describes the high-level 3D geometry of a LB simulation."""

    @classmethod
    def add_options(cls, group):
        LBGeometry2D.add_options(group)
        group.add_argument('--lat_nz', help='lattice depth', type=int,
                default=0)
        group.add_argument('--periodic_z', dest='periodic_z',
                help='make the lattice periodic in the Z direction',
                action='store_true', default=False)

    def __init__(self, config):
        LBGeometry.__init__(self, config)
        self.gz = config.lat_nz
        self.gsize = [self.gx, self.gy, self.gz]

    def blocks(self):
        """Returns a 1-element list containing a single 3D block
        covering the whole domain."""
        return [SubdomainSpec3D((0, 0, 0),
                          (self.config.lat_nx, self.config.lat_ny,
                           self.config.lat_nz))]



"""Supporting code for different node types."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'


# Node type classes.  Use this to define boundary conditions.
class LBNodeType(object):
    # Initialized when module is loaded.
    id = None
    wet_node = False
    location = 0.0

    def __init__(self, **params):
        self.params = params

class _NTGhost(LBNodeType):
    """Ghost node."""
    pass

class _NTUnused(LBNodeType):
    """Unused node."""

class NTHalfBBWall(LBNodeType):
    """Half-way bounce-back (no-slip) node."""
    wet_node = True

class NTFullBBWall(LBNodeType):
    """Full-way bounce-back (no-slip) node."""

class NTSlip(LBNodeType):
    """Full-slip node."""

class NTEquilibriumVelocity(LBNodeType):
    """Velocity boundary condition using the equilibrium distribution."""

    def __init__(self, velocity):
        self.params = {'velocity': velocity}

class NTEquilibriumDensity(LBNodeType):
    """Density boundary condition using the equilibrium distribution."""

class NTZouHeVelocity(LBNodeType):
    """Zou-he velocity."""

class NTZouHeDensity(LBNodeType):
    """Zou-He density."""

class NTGuoDensity(LBNodeType):
    """Guo density."""


def __init_node_type_list():
    """Assigns IDs to classes descendant from LBNodeType."""
    ret = []
    import sys
    curr_module = sys.modules[__name__]
    for symbol in dir(curr_module):
        obj = getattr(curr_module, symbol)
        try:
            if obj != LBNodeType and issubclass(obj, LBNodeType):
                ret.append(obj)
                obj.id = len(ret)
        except TypeError:
            pass

    return dict([(node_type.id, node_type) for node_type in ret])

def get_wet_node_type_ids():
    return [id for id, nt in _NODE_TYPES.iteritems() if nt.wet_node]

def get_dry_node_type_ids():
    return [id for id, nt in _NODE_TYPES.iteritems() if not nt.wet_node]


# Maps node type IDs to their classes.
_NODE_TYPES = __init_node_type_list()

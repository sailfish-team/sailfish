"""Miscellaneous utility functions."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import namedtuple
import sys

from sailfish import sym

TimingInfo = namedtuple('TimingInfo', 'comp bulk bnd coll data recv send wait total block_id')

def get_grid_from_config(config):
    for x in sym.KNOWN_GRIDS:
        if x.__name__ == config.grid:
            return x

    return None

def span_to_direction(span):
    for coord in span:
        if type(coord) is int:
            if coord == 0:
                return -1
            else:
                return 1
    return 0


def get_backends():
    for backend in ['cuda', 'opencl']:
        try:
            module = 'sailfish.backend_{0}'.format(backend)
            __import__('sailfish', fromlist=['backend_{0}'.format(backend)])
            yield sys.modules[module].backend
        except ImportError:
            pass

def get_visualization_engines():
    for engine in ['2d']:
        try:
            module = 'sailfish.vis_{0}'.format(engine)
            __import__('sailfish', fromlist=['vis_{0}'.format(engine)])
            yield sys.modules[module].engine
        except ImportError:
            pass

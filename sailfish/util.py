"""Miscellaneous utility functions."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict, namedtuple
import random
import socket
import sys

import numpy as np

from sailfish import config
from sailfish import sym

TimingInfo = namedtuple('TimingInfo', 'comp bulk bnd coll net_wait recv send total subdomain_id')


class GridError(Exception):
    pass


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


def get_backends(backends=['cuda', 'opencl']):
    for backend in backends:
        try:
            module = 'sailfish.backend_{0}'.format(backend)
            __import__('sailfish', fromlist=['backend_{0}'.format(backend)])
            yield sys.modules[module].backend
        except ImportError:
            pass

def get_visualization_engines():
    for engine in ['2d', 'mpl']:
        try:
            module = 'sailfish.vis_{0}'.format(engine)
            __import__('sailfish', fromlist=['vis_{0}'.format(engine)])
            yield sys.modules[module].engine
        except ImportError:
            pass

def gpufile_to_clusterspec(gpufile, iface=''):
    """Builds a Sailfish cluster definition based on a PBS GPUFILE."""

    nodes = defaultdict(set)
    f = open(gpufile, 'r')
    for line in f:
        line = line.strip()
        host, _, gpu = line.partition('-gpu')
        try:
            gpu = int(gpu)
        except ValueError:
            continue
        nodes[host].add(gpu)
    f.close()

    port = random.randint(8000, 16000)
    if not iface:
        iface = None

    cluster = []
    for node, gpus in nodes.iteritems():
        try:
            ipaddr = socket.gethostbyname(node)
        except socket.error:
            ipaddr = node
        cluster.append(config.MachineSpec('socket=%s:%s' % (ipaddr, port),
            node, gpus=list(gpus), iface=iface))

    class Cluster(object):
        def __init__(self, nodes):
            self.nodes = nodes

    return Cluster(cluster)

def reverse_pairs(iterable, subitems=1):
    it = iter(iterable)
    while it:
        x = []
        for i in range(0, subitems):
            x.append(it.next())

        try:
            y = []
            for i in range(0, subitems):
                y.append(it.next())

            for i in y:
                yield i
        except StopIteration:
            pass

        for i in x:
            yield i

def in_anyd(arr1, arr2):
    """Wrapper around np.in1d which returns an array with the same shape as arr1"""
    return np.in1d(arr1, arr2).reshape(arr1.shape)


def in_anyd_fast(arr1, values):
    """Faster version of in_anyd.

    :param arr1: array to check
    :param values: an iterable of values to look for in arr1
    """
    if len(values) == 0:
        return np.zeros(arr1.shape, dtype=np.bool)

    ret = arr1 == values[0]
    for v in values[1:]:
        ret = np.logical_or(ret, arr1 == v)
    return ret


def is_number(param):
    return type(param) is float or type(param) is int or isinstance(param, np.number)

def logpoints(i,Min=1.,Max=.1,n=10):
    """Return numbers spaced evenly on a log scale, similar to
    np.logspace, the difference if that it gets one number,
    and it take values not exponents as Min,Max

    :param i: get i-th number
    :param Min,Max: range
    :param n: number of samples
    """

    from math import exp,log
    if i<=0:
        return Min
    if i >=(n-1):
        return Max
    
    return exp(log(Min)+i*(log(Max)-log(Min))/(n-1))

def linpoints(i,Min=1.,Max=.1,n=10):
    """Return numbers spaced evenly on a linear scale, similar to
    np.logspace, the difference if that it gets one number.

    :param i: get i-th number
    :param Min,Max: range
    :param n: number of samples
    """

    if i<=0:
        return Min
    if i >=(n-1):
        return Max
    
    return Min+i*(Max-Min)/(n-1)

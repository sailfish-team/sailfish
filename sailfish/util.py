"""Miscellaneous utility functions."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict, namedtuple
import logging
import random
import socket
import sys

import numpy as np
from math import exp, log, ceil

from sailfish import config
from sailfish import sym

TimingInfo = namedtuple('TimingInfo',
                        'comp bulk bnd coll net_wait recv send total total_sq subdomain_id')


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
    for engine in ['2d', 'mpl', '3dcutplane']:
        try:
            module = 'sailfish.vis_{0}'.format(engine)
            __import__('sailfish', fromlist=['vis_{0}'.format(engine)])
            yield sys.modules[module].engine
        except ImportError:
            pass


def _cluster_from_nodes_dict(nodes, iface):
    port = random.randint(8000, 16000)
    if not iface:
        iface = None

    cluster = []
    for node, gpus in sorted(nodes.iteritems()):
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


def lsf_vars_to_clusterspec(vars, iface=''):
    """Builds a Saiflish cluster definition based on LSF environment variables."""

    gpus_per_core = float(vars['FDUST_GPU_PER_CORE'])

    def host_cpu(x):
        while x:
            host = x.pop()
            cpus = int(x.pop())
            yield host, cpus

    hosts = list(reversed(vars['LSB_MCPU_HOSTS'].split()))
    nodes = defaultdict(set)
    for host, cpus in host_cpu(hosts):
        for gpu in range(int(cpus * gpus_per_core)):
            nodes[host].add(gpu)

    return _cluster_from_nodes_dict(nodes, iface)


def gpufile_to_clusterspec(gpufile, iface=''):
    """Builds a Sailfish cluster definition based on a PBS GPUFILE."""

    nodes = defaultdict(set)
    f = open(gpufile, 'r')
    for line in f:
        line = line.strip()
        host, _, gpu = line.rpartition('-gpu')
        try:
            gpu = int(gpu)
        except ValueError:
            continue
        nodes[host].add(gpu)
    f.close()

    return _cluster_from_nodes_dict(nodes, iface)


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


def logpoints(i, min_=1., max_=.1, n=10):
    """Returns i-th number from a set spaced evenly on a log scale,
    similar to np.logspace, the difference is that it gets one number,
    and it take values (not exponents) as min_, max_.

    :param i: get i-th number
    :param min_, max_: range
    :param n: number of samples
    """
    if i <= 0:
        return min_
    if i >= (n - 1):
        return max_

    return exp(log(min_) + i * (log(max_) - log(min_)) / (n - 1))


def linpoints(i, min_=1., max_=.1, n=10):
    """Returns i-th number from a set spaced evenly on a log scale,
    similar to np.logspace, the difference is that it gets one number,
    and it take values (not exponents) as min_, max_.

    :param i: get i-th number
    :param min_, max_: range
    :param n: number of samples
    """
    if i <= 0:
        return min_
    if i >= (n - 1):
        return max_

    return min_ + i * (max_ - min_) / (n - 1)

def setup_logger(config):
    logger = logging.getLogger('saifish')
    formatter = logging.Formatter("[%(relativeCreated)6d %(levelname)5s %(processName)s] %(message)s")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    if config.verbose:
        stream_handler.setLevel(logging.DEBUG)
    elif config.quiet:
        stream_handler.setLevel(logging.WARNING)
    elif config.silent:
        stream_handler.setLevel(logging.ERROR)
    else:
        stream_handler.setLevel(logging.INFO)

    logger.addHandler(stream_handler)

    if config.log:
        handler = logging.FileHandler(config.log)
        handler.setFormatter(formatter)
        handler.setLevel(config.loglevel)
        logger.addHandler(handler)

    logger.setLevel(logging.DEBUG)
    return logger


def kinetic_energy(velocity):
    """Computes the mean kinetic energy of the fluid."""
    return np.sum(np.square(velocity)) / (2.0 * velocity[0].size)

def vorticity(velocity, dx=1.0):
    """Computes the vorticity array from a 3D velocity vector array."""
    dz_ux, dy_ux, dx_ux = np.gradient(velocity[0], dx, dx, dx)
    del dx_ux
    dz_uy, dy_uy, dx_uy = np.gradient(velocity[1], dx, dx, dx)
    del dy_uy
    dz_uz, dy_uz, dx_uz = np.gradient(velocity[2], dx, dx, dx)
    del dz_uz
    return np.array((dy_uz - dz_uy, dz_ux - dx_uz, dx_uy - dy_ux))

def enstrophy(velocity, dx):
    """Computes the enstrophy (mean square vorticity)."""
    return np.sum(np.square(vorticity(velocity, dx))) / (2.0 * velocity[0].size)

def skewness_factor(ux, n):
    """Computes the longitudinal skewness factor.

    :param ux: x component of velocity
    :param n: order of the skewness factor
    """
    _, _, dx_ux = np.gradient(ux)
    return np.mean(np.power(dx_ux, n)) * np.mean(np.square(dx_ux))**(-n/2.0) * (-1.0)**n

def structure_function(ux, r, n):
    return np.mean(np.power(ux - np.roll(ux, r, 2), n))

def energy_spectrum(velocity, buckets=None, density=False):
    """Calculates the energy spectrum E(k).

    :param velocity: velocity field
    :param buckets: if not None, an iterable of wavenumber buckets; if n values
        are provided here, the energy spectrum will contain n-1 values
    :param density: if True, an energy density spectrum in k-space will be
        calculated; if False the energy will simply be integrated
    :rvalue: numpy array with the energy spectrum
    """
    vx = velocity[0]
    vy = velocity[1]
    vz = velocity[2]

    Vx = np.fft.fftshift(np.fft.fftn(vx))
    Vy = np.fft.fftshift(np.fft.fftn(vy))
    Vz = np.fft.fftshift(np.fft.fftn(vz))

    z, y, x = vx.shape
    # Scaling factor.  Numpy's definition of the FFT does not include
    # any normalization.  For a symmetric FT/inverse FT weighting use
    # sqrt(x * y * z).
    scale = x * y * z

    Vx /= scale
    Vy /= scale
    Vz /= scale

    kz, ky, kx = np.mgrid[-z/2:z/2, -y/2:y/2, -x/2:x/2]
    kz += 1
    ky += 1
    kx += 1

    energy = np.abs(Vx)**2 + np.abs(Vy)**2 + np.abs(Vz)**2
    k = np.sqrt(kx**2 + ky**2 + kz**2)
    kmax = int(ceil(x / 2))

    if buckets is None:
        buckets = np.linspace(0, kmax, kmax + 1)

    spectrum = np.zeros(len(buckets))
    for i, (low, high) in enumerate(zip(buckets, buckets[1:])):
        spectrum[i] = np.sum(energy[(k >= low) & (k < high)])
        if density:
            spectrum[i] /= (high**3 - low**3)

    return spectrum

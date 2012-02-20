#!/usr/bin/env python

import numpy as np

from examples.ldc_3d import LDCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry3D
from sailfish.geo_block import SubdomainSpec3D

import util

class BenchmarkXConnGeometry(LBGeometry3D):
    def blocks(self, n=None):
        w = self.gx / 2
        blocks = [SubdomainSpec3D((0, 0, 0), (w, self.gy, self.gz)),
                  SubdomainSpec3D((w, 0, 0), (w, self.gy, self.gz))]
        return blocks


class BenchmarkYConnGeometry(LBGeometry3D):
    def blocks(self, n=None):
        h = self.gy / 2
        blocks = [SubdomainSpec3D((0, 0, 0), (self.gx, h, self.gz)),
                  SubdomainSpec3D((0, h, 0), (self.gx, h, self.gz))]
        return blocks


class BenchmarkZConnGeometry(LBGeometry3D):
    def blocks(self, n=None):
        d = self.gz / 2
        blocks = [SubdomainSpec3D((0, 0, 0), (self.gx, self.gy, d)),
                  SubdomainSpec3D((0, 0, d), (self.gx, self.gy, d))]
        return blocks

def run_benchmark(boundary_split=True, suffix=''):
    settings = {
            'mode': 'benchmark',
            'max_iters': 1000,
            'every': 500,
            'blocks': 2,
            'quiet': True,
            'grid': 'D3Q19',
            'gpus': [0, 1],
            'bulk_boundary_split': boundary_split,
        }

    fmt = ['%d', '%d', '%d', '%d', '%.6e', '%.6e', '%.6e', '%.6e']

    mem_size = 2**32 - 2**29        # assume 4G - 256M of memory
    node_cost = (2 * 19 + 4) * 4    # node cost in bytes

    def test_sizes(sizes, geo_cls):
        for w, h, d in sizes:
            print 'Testing {0} x {1} x {2}...'.format(w, h, d)
            settings.update({'lat_nx': int(w), 'lat_ny': int(h), 'lat_nz': int(d)})
            ctrl = LBSimulationController(LDCSim, geo_cls, settings)
            timing_infos, blocks = ctrl.run()
            summary.append(util.summarize(timing_infos, blocks))
            timings.append(timing_infos)

    summary = []
    timings = []

    # 'h' x 'd' is the connection surface
    #####################################
    w = np.uint32(range(2, 18, 2)) * 64 - 4
    h = np.uint32([128] * len(w))
    d = mem_size / node_cost / w / h
    sizes = [(x, y, z) for x, y, z in zip(w, h, d)]

    test_sizes(sizes, BenchmarkXConnGeometry)
    np.savetxt('3d_2blocks_x_v1{0}.dat'.format(suffix), summary, fmt)
    util.save_timing('3d_2blocks_x_v1{0}.timing'.format(suffix), timings)

    # Same as above, but the 'z' dimension is varied.  Should give the same
    # results as when the 'y' dimension is varied.
    d = np.uint32([128] * len(w))
    h = mem_size / node_cost / w / d
    sizes = [(x, y, z) for x, y, z in zip(w, h, d)]
    summary = []
    timings = []

    test_sizes(sizes, BenchmarkXConnGeometry)
    np.savetxt('3d_2blocks_x_v2{0}.dat'.format(suffix), summary, fmt)
    util.save_timing('3d_2blocks_x_v2{0}.timing'.format(suffix), timings)

    # 'w' x 'd' is the connection surface
    #####################################
    d = np.uint32([128] * len(w))
    h = mem_size / node_cost / w / d
    sizes = [(x, y, z) for x, y, z in zip(w, h, d)]
    summary = []
    timings = []

    test_sizes(sizes, BenchmarkYConnGeometry)
    np.savetxt('3d_2blocks_y{0}.dat'.format(suffix), summary, fmt)
    util.save_timing('3d_2blocks_y{0}.timing'.format(suffix), timings)

    # 'w' x 'h' is the connection surface
    #####################################
    h = np.uint32([128] * len(w))
    d = mem_size / node_cost / w / h
    sizes = [(x, y, z) for x, y, z in zip(w, h, d)]
    summary = []
    timings = []

    test_sizes(sizes, BenchmarkZConnGeometry)
    np.savetxt('3d_2blocks_z{0}.dat'.format(suffix), summary, fmt)
    util.save_timing('3d_2blocks_z{0}.timing'.format(suffix), timings)


if __name__ == '__main__':
    run_benchmark()
    run_benchmark(False, 'nosplit')

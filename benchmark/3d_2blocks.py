#!/usr/bin/env python

import numpy as np

from examples.lbm_ldc_multi_3d import LDCSim
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
            'max_iters': 300,
            'every': 100,
            'blocks': 2,
            'quiet': True,
            'grid': 'D3Q19',
            'gpus': [0,1],
            'bulk_boundary_split': boundary_split,
        }

    fmt = ['%d', '%d', '%d', '%d', '%.6e', '%.6e', '%.6e', '%.6e']

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
    sizes = [(1276, 128, 124), (1148, 128, 138), (1020, 128, 155), (892, 128, 177),
            (762, 256, 104), (636, 256, 124), (508, 256, 156), (380, 256, 207),
            (252, 256, 314), (124, 512, 320)]
    test_sizes(sizes, BenchmarkXConnGeometry)
    np.savetxt('3d_2blocks_x{0}.dat'.format(suffix), summary, fmt)
    util.save_timing('3d_2blocks_x{0}.timing'.format(suffix), timings)

if __name__ == '__main__':
    run_benchmark()
    run_benchmark(False, 'nosplit')

#!/usr/bin/env python
import numpy as np

from examples.lbm_ldc_multi import LDCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import SubdomainSpec2D

import util

class BenchmarkXConnGeometry(LBGeometry2D):
    def blocks(self, n=None):
        w = self.gx / 2
        blocks = [SubdomainSpec2D((0, 0), (w, self.gy)),
                  SubdomainSpec2D((w, 0), (w, self.gy))]
        return blocks


class BenchmarkYConnGeometry(LBGeometry2D):
    def blocks(self, n=None):
        h = self.gy / 2
        blocks = [SubdomainSpec2D((0, 0), (self.gx, h)),
                  SubdomainSpec2D((0, h), (self.gx, h))]
        return blocks


def run_benchmark(boundary_split=True, suffix=''):
    settings = {
            'mode': 'benchmark',
            'max_iters': 1000,
            'every': 500,
            'blocks': 2,
            'quiet': True,
            'gpus': [0,1],
            'bulk_boundary_split': boundary_split,
        }

    mem_size = 2**32 - 2**28        # assume 4G - 256M memory size
    node_cost = (2 * 9 + 4) * 4     # node cost in bytes
    fmt = ['%d', '%d', '%d', '%.6e', '%.6e', '%.6e', '%.6e']

    def test_sizes(sizes, geo_cls):
        for w, h in sizes:
            print 'Testing {0} x {1}...'.format(w, h)
            settings.update({'lat_nx': int(w), 'lat_ny': int(h)})
            ctrl = LBSimulationController(LDCSim, geo_cls, settings)
            timing_infos, blocks = ctrl.run()
            summary.append(util.summarize(timing_infos, blocks))
            timings.append(timing_infos)

    summary = []
    timings = []
    h = np.logspace(6, 15, 10, base=2).astype(np.uint32)
    w = mem_size / node_cost / h
    sizes = [(x, y) for x, y in zip(w, h)]
    test_sizes(sizes, BenchmarkXConnGeometry)
    np.savetxt('2d_2blocks_x{0}.dat'.format(suffix), summary, fmt)
    util.save_timing('2d_2blocks_x{0}.timing'.format(suffix), timings)

    print '---'

    summary = []
    timings = []
    w = np.logspace(10, 17, 8, base=2).astype(np.uint32) - 4
    h = mem_size / node_cost / w
    sizes = [(x, y) for x, y in zip(w, h)]
    test_sizes(sizes, BenchmarkYConnGeometry)
    np.savetxt('2d_2blocks_y{0}.dat'.format(suffix), summary, fmt)
    util.save_timing('2d_2blocks_y{0}.timing'.format(suffix), timings)


if __name__ == '__main__':
    run_benchmark()
    run_benchmark(False, 'nosplit')

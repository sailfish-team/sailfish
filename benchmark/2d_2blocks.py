#!/usr/bin/env python
import numpy as np

from examples.lbm_ldc_multi import LDCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D

import util

class BenchmarkXConnGeometry(LBGeometry2D):
    def blocks(self, n=None):
        w = self.gx / 2
        blocks = [LBBlock2D((0, 0), (w, self.gy)),
                LBBlock2D((w, 0), (w, self.gy))]
        return blocks


class BenchmarkYConnGeometry(LBGeometry2D):
    def blocks(self, n=None):
        h = self.gy / 2
        blocks = [LBBlock2D((0, 0), (self.gx, h)),
                LBBlock2D((0, h), (self.gx, h))]
        return blocks


def run_benchmark():
    settings = {
            'mode': 'benchmark',
            'max_iters': 1000,
            'every': 500,
            'blocks': 2,
            'quiet': True,
            'gpus': [0,1],
        }

    mem_size = 2**32 - 2**28  # assume 2G - 128M memory size
    node_cost = (2 * 9 + 4) * 4  # node cost in bytes
    fmt = ['%d', '%d', '%d', '%.6e', '%.6e', '%.6e', '%.6e']

    summary = []
    sizes = np.logspace(6, 15, 10, base=2).astype(np.uint32)
    for h in sizes:
        w = mem_size / node_cost / h
        print 'Testing {0} x {1}...'.format(w, h)
        settings.update({'lat_nx': int(w), 'lat_ny': int(h)})
        ctrl = LBSimulationController(LDCSim, BenchmarkXConnGeometry,
                settings)
        timing_infos, blocks = ctrl.run()
        summary.append(util.summarize(timing_infos, blocks))

    np.savetxt('2d_2blocks_x.dat', summary, fmt)

    print '---'

    summary = []
    sizes = np.logspace(9, 17, 9, base=2).astype(np.uint32) - 4
    for w in sizes:
        h = mem_size / node_cost / w
        print 'Testing {0} x {1}...'.format(w, h)
        settings.update({'lat_nx': int(w), 'lat_ny': int(h)})
        ctrl = LBSimulationController(LDCSim, BenchmarkYConnGeometry,
                settings)
        timing_infos, blocks = ctrl.run()
        summary.append(util.summarize(timing_infos, blocks))

    np.savetxt('2d_2blocks_y.dat', summary, fmt)


if __name__ == '__main__':
    run_benchmark()

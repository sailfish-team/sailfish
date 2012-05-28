#!/usr/bin/env python
"""Weak scaling with 2D binary phase fluid."""

import numpy as np

from examples.binary_fluid.sc_separation_2d import SeparationSCSim, SeparationDomain
from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry2D
from benchmark.scaling import util

class TestDomain(SeparationDomain):
    def boundary_conditions(self, hx, hy):
        self.set_node(np.logical_or(
            np.logical_or(hx == 0, hy == 0),
            np.logical_or(hx == self.gx - 1, hy == self.gy - 1)),
            self.NODE_WALL)


def run_benchmark(num_blocks):
    # Lattice size is optimized for Tesla C2050.
    settings = {
        'max_iters': 1000,
        'every': 500,
        'quiet': True,
        'block_size': 128,
        'subdomains': num_blocks,
        'conn_axis': 'y',
        'mode': 'benchmark',
        'periodic_x': False,
        'periodic_y': False,
        'lat_nx': 3710,
        'lat_ny': 3990 * num_blocks,
        }

    SeparationSCSim.subdomain = TestDomain
    ctrl = LBSimulationController(SeparationSCSim, EqualSubdomainsGeometry2D, settings)
    timing_infos, min_timings, max_timings, subdomains = ctrl.run()
    util.save_result('weak_2d_binary', num_blocks, timing_infos, min_timings,
            max_timings, subdomains)


if __name__ == '__main__':
    args = util.process_cmdline()
    run_benchmark(args.num_blocks)

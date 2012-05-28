#!/usr/bin/env python
"""Weak scaling with 2D single phase fluid."""

from examples.lbc_2d import LDCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry2D


def run_benchmark(num_blocks):
    # Lattice size is optimized for Tesla C2050.
    settings = {
        'max_iters': 1000,
        'every': 500,
        'quiet': True,
        'block_size': 128,
        'subdomains': num_blocks,
        'conn_axis': 'y',
        'lat_nx': 5118,
        'lat_ny': 5800 * num_blocks,
        }

    ctrl = LBSimulationController(LDCSim, EqualSubdomainsGeometry2D, settings)
    timing_infos, subdomains = ctrl.run()
    util.save_result('weak_2d_single', num_blocks, timing_infos, subdomains)

if __name__ == '__main__':
    args = util.process_cmdline()
    run_benchmark()

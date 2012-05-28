#!/usr/bin/env python
import argparse
import sys

from examples.lbc_2d import LDCSim
from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry2D


def run_benchmark(num_blocks):
    global num_blocks
    settings = {
        'mode': 'benchmark',
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
    timing_infos, blocks = ctrl.run()

    f = open('weak_2d_single_%s' % num_blocks, 'w')
    f.write(str(timing_infos))
    f.close()

    mlups_total = 0
    mlups_comp = 0

    for ti in timing_infos:
        block = blocks[ti.block_id]
        mlups_total += block.num_nodes / ti.total * 1e-6
        mlups_comp  += block.num_nodes / ti.comp * 1e-6

    f = open('weak_2d_single_mlups_%s' % num_blocks, 'w')
    f.write('%s %s\n' % (mlups_total, mlups_comp))
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=1)
    args, remaining = parser.parse_known_args()
    del sys.argv[:1]
    sys.argv.extend(remaining)
    run_benchmark()

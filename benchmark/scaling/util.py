import argparse
import sys


def save_result(filename_base, num_blocks, timing_infos, min_timings,
        max_timings, subdomains):
    f = open('%s_%d' % (filename_base, num_blocks), 'w')
    f.write(str(timing_infos))
    f.close()

    f = open('%s_min_%d' % (filename_base, num_blocks), 'w')
    f.write(str(min_timings))
    f.close()

    f = open('%s_max_%d' % (filename_base, num_blocks), 'w')
    f.write(str(max_timings))
    f.close()

    mlups_total = 0
    mlups_comp = 0

    for ti in timing_infos:
        block = subdomains[ti.block_id]
        mlups_total += block.num_nodes / ti.total * 1e-6
        mlups_comp  += block.num_nodes / ti.comp * 1e-6

    f = open('%s_mlups_%d' % (filename_base, num_blocks), 'w')
    f.write('%.2f %.2f\n' % (mlups_total, mlups_comp))
    f.close()


def process_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_blocks', type=int, default=1)
    args, remaining = parser.parse_known_args()
    del sys.argv[1:]
    sys.argv.extend(remaining)
    return args

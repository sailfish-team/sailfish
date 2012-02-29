import argparse
import sys

import numpy as np

from utils.merge_subdomains import merge_subdomains

def parse_cmd_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--block_size', metavar='N', type=int, default=64,
            help='CUDA block size')
    args, remaining = parser.parse_known_args()
    # Remove processed arguments, but keep everything else in sys.argv.
    del sys.argv[1:]
    sys.argv.extend(remaining)
    return args


def verify_fields(ref, output, digits, max_iters):
    merged = merge_subdomains(output, digits, max_iters, save=False)
    for f in ref.files:
        np.testing.assert_array_almost_equal(merged[f], ref[f])


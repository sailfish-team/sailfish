#!/usr/bin/python
"""
A utility to merge subdomain outputs into a single output file.

Usage:
    ./merge_subdomains.py [--all] file.0.00001.npz
where:
    file.0.00001.npz is any output file for any block from the
    output series to be merged.  If --all is specified, all
    iterations are processed.
"""
from __future__ import print_function
import argparse
import glob
try:
    import cPickle as pickle
except ImportError:
    import pickle

import sys

import numpy as np

from sailfish import io


def get_bounding_box(subdomains):
    dim = subdomains[0].dim

    gx = max((s.end_location[0] for s in subdomains))
    gy = max((s.end_location[1] for s in subdomains))
    if dim == 3:
        gz = max((s.end_location[2] for s in subdomains))
        return gz, gy, gx
    else:
        return gy, gx


def merge_subdomains(base, digits, it, save=True):
    fn_subdomains = io.subdomains_filename(base)
    with open(fn_subdomains, 'rb') as f:
        subdomains = pickle.load(f)
    bb = get_bounding_box(subdomains)

    data = np.load(io.filename(base, digits, subdomains[0].id, it))
    dtype = data['v'].dtype
    dim = data['v'].shape[0]

    out = {}
    for field in data.files:
        if len(data[field].shape) == dim:
            shape = bb
        else:
            shape = list(data[field].shape)
            shape[-dim:] = bb

        out[field] = np.zeros(shape, dtype=dtype)
        out[field][:] = np.nan
        # np.ma.masked_all(shape, dtype=dtype)

    for s in subdomains:
        fn = io.filename(base, digits, s.id, it)
        data = np.load(fn)
        for field in data.files:
            selector = [slice(None)] * (len(data[field].shape) - dim)
            selector.extend([slice(i0, i1) for i0, i1 in reversed(list(zip(s.location, s.end_location)))])
            selector = tuple(selector)
            out[field][selector] = data[field]

    if save:
        np.savez(io.merged_filename(base, digits, it), **out)
    return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    args, remaining = parser.parse_known_args()

    if remaining:
        sample = remaining[0]
    else:
        sys.exit(0)

    base, sub_id, it, _ = sample.rsplit('.', 3)
    digits = len(it)

    if args.all:
        for fn in glob.glob('.'.join([base, sub_id, ('[0-9]' * digits), 'npz'])):
            _, _, it, _ = fn.rsplit('.', 3)
            print('Processing {0}'.format(it))
            merge_subdomains(base, digits, int(it))
    else:
        merge_subdomains(base, digits, int(it))

#!/usr/bin/env pyhton
"""
Preprocesses a geometry .npy file.

Usage:
    ./process_geometry.py <in> <out> <axes> <exp>

Expects in.config and in.npy to be present.
Arguments:
  out: base filename for outputs
  axes: a string such as 'xyz' indicating the target axis order (original order
        is 'xyz')
  exp: a sequence of 6 digits (two per axis), indicating the expected
       number of outlets/inlets, e.g. 012000; order is the same as the original
       order of axes, prior to any reshuffling
"""

import gzip
import json
import os
import sys
import numpy as np
from scipy import ndimage

if len(sys.argv) < 5:
    print 'Usage: ./process_geometry.py <in> <out> <axes> <exp>'
    sys.exit(0)

fname_in, fname_out, axes, expected_io = sys.argv[1:5]

config = json.load(open(fname_in + '.config', 'r'))
if os.path.exists(fname_in + '.npy.gz'):
    geo = np.load(gzip.GzipFile(fname_in + '.npy.gz', 'r'))
else:
    geo = np.load(open(fname_in + '.npy'))
fluid = np.logical_not(geo)

def make_slice(axis, pos):
    ret = []
    for i in range(0, 3):
        if 2 - axis == i:
            ret.append(pos)
        else:
            ret.append(slice(None))
    return ret

# Find the envelope of nodes that needs to be discarded in order
# for every axis to have the desired number of outlets/inlets.
slices = []
cuts = [[0,0], [0,0], [0,0]]
io_idx = 0

# Iterate over the natural axes (x, y, z).
for axis in range(0, 3):
    # Scan lower end of the current axis.
    outlets = int(expected_io[io_idx])
    if outlets > 0:
        for i in range(0, geo.shape[2 - axis]):
            tmp = fluid[make_slice(axis, i)]
            if np.sum(tmp) == 0:
                continue
            _, num = ndimage.label(tmp)
            if outlets == num:
                start = i
                cuts[axis][0] = i
                break
    else:
        start = 0

    # Scan higher end of the current axis.
    io_idx += 1
    outlets = int(expected_io[io_idx])
    if outlets > 0:
        for i in range(1, geo.shape[2 - axis]):
            tmp = fluid[make_slice(axis, geo.shape[2 - axis] - i)]
            if np.sum(tmp) == 0:
                continue
            _, num = ndimage.label(tmp)
            if outlets == num:
                end = geo.shape[2 - axis] - i + 1
                cuts[axis][1] = i
                break
    else:
        end = None

    slices.append(slice(start, end))
    io_idx += 1

# Discard envelope if necessary,
geo = geo[list(reversed(slices))]

# Start building a new config file.

# Size is stored in the physical order corresponding to how data in the
# underlying array is laid out.
config['size'] = geo.shape
# The axes positions, as well as the fields below, use the natural axis order
# -- xyz. This is the  opposite of the storage order in the LB arrays.
config['axes'] = axes
config['cuts'] = cuts

name_to_idx = {'x': 0, 'y': 1, 'z': 2}
targets = [0, 0, 0]
actual = [0, 1, 2]

# Reorder axes.
for i, a in enumerate(axes):
    ai = name_to_idx[a]
    targets[i] = ai

for i, tg in enumerate(targets):
    if actual[i] != tg:
        other = actual.index(tg)
        geo = np.swapaxes(geo, i, other)
        t = actual[i]
        actual[i] = tg
        actual[other] = t

assert 0 in actual
assert 1 in actual
assert 2 in actual

np.save(fname_out + '.npy', geo)
json.dump(config, open(fname_out + '.config', 'w'))

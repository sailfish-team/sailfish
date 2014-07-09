#!/usr/bin/env pyhton
"""
Preprocesses a geometry .npy file.

Usage:
    ./process_geometry.py <in> <out> <axes> <exp>

Expects in.config and in.npy to be present.
Arguments:
  out: base filename for outputs
  axes: a string such 'xyz' indicating the target axis order
  exp: a sequence of 6 digits (two per axis), indicating the expected
       number of outlets/inlets, e.g. 012000.
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

fname_in, fname_out, axes, exp = sys.argv[1:5]

config = json.load(open(fname_in + '.config', 'r'))
if os.path.exists(fname_in + '.npy.gz'):
    geo = np.load(gzip.GzipFile(fname_in + '.npy.gz', 'r'))
else:
    geo = np.load(open(fname_in + '.npy'))
fluid = np.logical_not(geo)

def make_slice(axis, pos):
    ret = []
    for i in range(0, 3):
        if axis == i:
            ret.append(pos)
        else:
            ret.append(slice(None))
    return ret

# Find the envelope of nodes that needs to be discarded in order
# for every axis to have the desired number of outlets/inlets.
padding = []
slices = []
idx = 0
for axis in range(0, 3):
    # Scan lower end of the current axis.
    outlets = int(exp[idx])
    if outlets > 0:
        for i in range(0, geo.shape[axis]):
            tmp = fluid[make_slice(axis, i)]
            if np.sum(tmp) == 0:
                continue
            _, num = ndimage.label(tmp)
            if outlets == num:
                padding.append(0)
                start = i
                break
    else:
        start = 0
        padding.append(1)   # geometry already has 1 node of padding

    # Scan higher end of the current axis.
    idx += 1
    outlets = int(exp[idx])
    if outlets > 0:
        for i in range(1, geo.shape[axis]):
            tmp = fluid[make_slice(axis, geo.shape[axis] - i)]
            if np.sum(tmp) == 0:
                continue
            _, num = ndimage.label(tmp)
            if outlets == num:
                padding.append(0)
                end = geo.shape[axis] - i + 1
                break
    else:
        end = None
        padding.append(1)   # geometry already has 1 node of padding

    slices.append(slice(start, end))
    idx += 1

config['orig_size'] = geo.shape

# Discard envelope if necessary,
geo = geo[slices]

# Start building a new config file.
config['size'] = geo.shape
config['padding'] = padding
config['axes'] = axes

name_to_idx = {'x': 0, 'y': 1, 'z': 2}
targets = [0, 0, 0]
actual = [0, 1, 2]

# Reoder axes.
for i, a in enumerate(axes):
    ai = name_to_idx[a]
    targets[i] = ai

for i, tg in enumerate(targets):
    if actual[i] != tg:
        other = actual.index(tg)
        geo = np.swapaxes(geo, i, other)
        t = actual[i]
        actual[i] = other
        actual[other] = t

np.save(fname_out + '.npy', geo)
json.dump(config, open(fname_out + '.config', 'w'))

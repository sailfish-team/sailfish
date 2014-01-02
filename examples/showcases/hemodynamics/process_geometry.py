#!/usr/bin/env pyhton
"""

Usage:
    ./process_geometry.py <in> <out> <axes> <exp>

Expects in.config and in.npy to be present.
<axes> is a string such 'xyz' indicating the target axis order.
<exp> is a sequence of 6 letters (two per axis), indicating the expected
    number of outlets/inlets.
"""

import json
import sys
import numpy as np
from scipy import ndimage

if len(sys.argv) < 5:
    print 'Usage: ./process_geometry.py <in> <out> <axes> <exp>'
    sys.exit(0)

fname_in = sys.argv[1]
fname_out = sys.argv[2]
axes = sys.argv[3]
exp = sys.argv[4]

config = json.load(open(fname_in + '.config', 'r'))
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

padding = []
slices = []
idx = 0
for axis in range(0, 3):
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

geo = geo[slices]
config['size'] = geo.shape
config['padding'] = padding
config['axes'] = axes


name_to_idx = {'x': 0, 'y': 1, 'z': 2}
targets = [0, 0, 0]
actual = [0, 1, 2]

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

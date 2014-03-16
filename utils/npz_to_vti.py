#!/usr/bin/python

"""Converts npz data files to VTK files.

Usage:
    ./npz_to_vti.py <file.npz> [config.json]
"""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import argparse
import json
import os
import sys
import numpy as np

os.environ['ETS_TOOLKIT'] = 'null'
from tvtk.api import tvtk

parser = argparse.ArgumentParser()
parser.add_argument('--dt', type=float, default=0.0)
args, remaining = parser.parse_known_args()

filename = remaining[0]
src_data = np.load(filename)

dt = 1.0 if args.dt == 0.0 else args.dt

max_len = 0
for field in src_data.files:
    max_len = max(len(src_data[field].shape), max_len)

field = src_data[src_data.files[0]]
shape = None
origin = [0.0, 0.0, 0.0]
spacing = [1.0, 1.0, 1.0]
config = {}

if len(sys.argv) > 2:
    config = json.load(open(sys.argv[2]))
    axes = {'x': 0, 'y': 1, 'z': 2}

    # Maps physical axis to LB axis.
    amap = [0, 1, 2]
    if 'axes' in config:
        for i, a in enumerate(config['axes']):
            amap[axes[a]] = i

    for i, ((xmin, xmax), lb_extent) in enumerate(zip(config['bounding_box'], config['size'])):
        # -2 is due to padding
        scale = (xmax - xmin) / (lb_extent - 2)
        origin[i] = xmin - 0.5 * scale
        spacing[i] = scale

        if 'slices' in config:
            start = config['slices'][amap[i]][0]
            if start is not None:
                origin[i] += start * scale


def reorder(data):
    """Reorders the array so that the physical axes are: z, y, x."""
    if 'axes' not in config:
        return data
    ret = data
    axes = config['axes']
    x = axes.index('x')
    ret = np.swapaxes(ret, x, 2)
    axes[x] = axes[2]
    axes[2] = 'x'
    ret = np.swapaxes(ret, axes.index('y'), 1)
    return ret

# 3D
if max_len == 4:
    dim = 3
    if len(field.shape) == 3:
        shape = reorder(field).shape
    else:
        shape = reorder(field).shape[1:]
# 2D:
elif max_len == 3:
    dim = 2
    if len(field.shape) == 2:
        shape = reorder(field).shape
    else:
        shape = reorder(field).shape[1:]
else:
    raise ValueError('Unexpected field shape length %d' % max_len)

first = True
idata = tvtk.ImageData(spacing=spacing, origin=origin)
# Only process scalar fields.
for field in src_data.files:
    if len(src_data[field].shape) == max_len:
        continue

    if first:
        idata.point_data.scalars = reorder(src_data[field]).flatten()
        idata.point_data.scalars.name = field
        first = False
    else:
        t = idata.point_data.add_array(reorder(src_data[field]).flatten())
        idata.point_data.get_array(t).name = field

# Only process vector fields.
for field in src_data.files:
    if len(src_data[field].shape) != max_len:
        continue

    f = src_data[field]

    if dim == 3:
        tmp = idata.point_data.add_array(np.c_[reorder(f[0]).flatten() * spacing[0]/dt,
                                               reorder(f[1]).flatten() * spacing[1]/dt,
                                               reorder(f[2]).flatten() * spacing[2]/dt])
    else:
        tmp = idata.point_data.add_array(np.c_[reorder(f[0]).flatten() * spacing[0]/dt,
                                               reorder(f[1]).flatten() * spacing[1]/dt,
                                               np.zeros_like(f[0].flatten())])
    idata.point_data.get_array(tmp).name = field

if dim == 3:
    idata.dimensions = list(reversed(shape))
else:
    idata.dimensions = list(reversed(shape)) + [1]

out_filename = filename.replace('.npz', '.vti')
w = tvtk.XMLImageDataWriter(input=idata, file_name=out_filename)
w.write()

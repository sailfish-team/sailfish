#!/usr/bin/python

"""Converts npz data files to VTK files.

Usage:
    ./npz_to_vti.py <file.npz>
"""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import argparse
import os
import sys
import numpy as np

os.environ['ETS_TOOLKIT'] = 'null'
from tvtk.api import tvtk

parser = argparse.ArgumentParser()
parser.add_argument('--dx', type=float, default=0.0)
parser.add_argument('--dt', type=float, default=0.0)
args, remaining = parser.parse_known_args()

filename = remaining[0]
src_data = np.load(filename)

dx = 1.0 if args.dx == 0.0 else args.dx
dt = 1.0 if args.dt == 0.0 else args.dt

max_len = 0
for field in src_data.files:
    max_len = max(len(src_data[field].shape), max_len)

field = src_data[src_data.files[0]]
shape = None

# 3D
if max_len == 4:
    dim = 3
    if len(field.shape) == 3:
        shape = field.shape
    else:
        shape = field.shape[1:]
# 2D:
elif max_len == 3:
    dim = 2
    if len(field.shape) == 2:
        shape = field.shape
    else:
        shape = field.shape[1:]
else:
    raise ValueError('Unexpected field shape length %d' % max_len)

first = True
idata = tvtk.ImageData(spacing=(dx, dx, dx), origin=(0, 0, 0))
# Only process scalar fields.
for field in src_data.files:
    if len(src_data[field].shape) == max_len:
        continue

    if first:
        idata.point_data.scalars = src_data[field].flatten()
        idata.point_data.scalars.name = field
        first = False
    else:
        t = idata.point_data.add_array(src_data[field].flatten())
        idata.point_data.get_array(t).name = field

# Only process vector fields.
for field in src_data.files:
    if len(src_data[field].shape) != max_len:
        continue

    f = src_data[field]

    if dim == 3:
        tmp = idata.point_data.add_array(np.c_[f[0].flatten() * dx/dt,
                                               f[1].flatten() * dx/dt,
                                               f[2].flatten() * dx/dt])
    else:
        tmp = idata.point_data.add_array(np.c_[f[0].flatten() * dx/dt,
                                               f[1].flatten() * dx/dt,
                                               np.zeros_like(f[0].flatten())])
    idata.point_data.get_array(tmp).name = field

if dim == 3:
    idata.dimensions = list(reversed(shape))
else:
    idata.dimensions = list(reversed(shape)) + [1]

out_filename = filename.replace('.npz', '.vti')
w = tvtk.XMLImageDataWriter(input=idata, file_name=out_filename)
w.write()

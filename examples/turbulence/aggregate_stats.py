#!/usr/bin/python
#
# Performs time-averaging of Reynolds stats files.
# No space-averaging is done.
#
# Usage:
#  ./agregate_stats.py <output> <input ...>
#
# <input> can be filenames or filename,weight. The weights
# are automatically normalized.


from collections import defaultdict
import numpy as np
import sys

from sailfish import io

data = {}
weight_sum = 0.0

sources = []

for fname in sys.argv[2:]:
    fname, _, weight = fname.partition(',')
    if weight:
        weight = float(weight)
    else:
        weight = 1.0
    a = np.load(fname)
    sources.append(io.iter_from_filename(fname))

    print "\r", fname, np.max(a['uz_m1']),
    for field in a.files:
        if field in data:
            data[field] += weight * a[field]
        else:
            data[field] = weight * a[field]
    weight_sum += 1

for v in data.itervalues():
    v /= weight_sum

np.savez(sys.argv[1], sources=sources, **data)

#!/usr/bin/python
#
# ./agregate_stats.py <output> <input ...>

from collections import defaultdict
import numpy as np
import sys

data = {}
cnt = 0

for fname in sys.argv[2:]:
    a = np.load(fname)
    for field in a.files:
        if field in data:
            data[field] += a[field]
        else:
            data[field] = a[field]
    cnt += 1

for v in data.itervalues():
    v /= cnt

np.savez(sys.argv[1], **data)

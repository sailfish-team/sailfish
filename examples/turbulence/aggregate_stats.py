#!/usr/bin/python
#
# Performs time-averaging of Reynolds stats files.
# No space-averaging is done.
#
# Usage:
#  ./agregate_stats.py <output> <input ...>

from collections import defaultdict
import numpy as np
import sys

data = {}
cnt = 0

for fname in sys.argv[2:]:
    a = np.load(fname)
    print "\r", fname, np.max(a['uz_m1']),
    for field in a.files:
        if field in data:
            data[field] += a[field]
        else:
            data[field] = a[field]
    cnt += 1

for v in data.itervalues():
    v /= cnt

np.savez(sys.argv[1], **data)

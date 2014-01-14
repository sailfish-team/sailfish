#!/usr/bin/python
#
# Performs space averaging of Reynolds stats files
# and generates a time series. Every step in the timeseries
# corresponds to a profile averaged over the whole domain
# and N time steps, where N is the number of time steps
# stored in a single stats files processed as the input.
#
# The output data is useful for estimating convergence.
#
# Usage:
#  ./aggregate_timeseries.py <axis> <output> <input ..>

from collections import defaultdict
import numpy as np
import sys

axis = int(sys.argv[1])

data = {}
cnt = 0

for fname in sys.argv[3:]:
    a = np.load(fname)
    for field in a.files:
        avg = np.average(a[field], axis=axis)
        if field in data:
            data[field] = np.vstack([data[field], avg])
        else:
            data[field] = avg

np.savez(sys.argv[2], **data)



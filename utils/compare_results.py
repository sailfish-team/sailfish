#!/usr/bin/python

from __future__ import print_function
import numpy as np
import sys

data_a = np.load(sys.argv[1])
data_b = np.load(sys.argv[2])

if data_a.files != data_b.files:
    print('Different fields: %s vs %s' % (
            data_a.files, data_b.files), file=sys.stderr)
    sys.exit(1)

err = 0

for f in data_a.files:
    if not np.all(np.nan_to_num(data_a[f]) == np.nan_to_num(data_b[f])):
        print('Difference in field "%s", max deviation is: %e.' % (
            f, np.nanmax(np.abs(data_a[f] - data_b[f]))), file=sys.stderr)
        err += 1

sys.exit(err)

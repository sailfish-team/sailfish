#!/usr/bin/python

import numpy as np
import sys

data_a = np.load(sys.argv[1])
data_b = np.load(sys.argv[2])

if data_a.files != data_b.files:
    print >>sys.stderr, 'Different fields: %s vs %s' % (
            data_a.files, data_b.files)
    sys.exit(1)

err = 0

for f in data_a.files:
    if not np.all(np.nan_to_num(data_a[f]) == np.nan_to_num(data_b[f])):
        print >>sys.stderr, 'Difference in field "%s", max deviation is: %e.' % (
            f, np.nanmax(np.abs(data_a[f] - data_b[f])))
        err += 1

sys.exit(err)

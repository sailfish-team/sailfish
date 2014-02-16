#/usr/bin/python
#
# Prints the sum of all finite values of all fields on a given
# slice. Arguments: axis, position

import sys
import numpy as np

axes = 'z', 'y', 'x'
axis = axes.index(sys.argv[1])
pos = int(sys.argv[2])
data = np.load(sys.argv[3])

slc = []
for i in range(0, 3):
    if i == axis:
        slc.append(slice(pos, pos+1))
    else:
        slc.append(slice(None))

res = []
for k, f in data.iteritems():
    if len(f.shape) == 4:
        for i, ax in enumerate(('x', 'y', 'z')):
            slc2 = [i] + slc
            res.append((k + ax, np.nansum(f[slc2])))
    else:
        res.append((k, np.nansum(f[slc])))


print '  '.join(['%s: %.4f' % (x, y) for x, y in res])

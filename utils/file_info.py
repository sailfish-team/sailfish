#!/usr/bin/python
#
# Prints basic information about a Sailfish output file.

import sys
import numpy as np

data = np.load(sys.argv[1])
print 'Fields:', ', '.join(data.files)
nz, ny, nx = data[data.files[0]].shape
print 'Shape (ZYX): %d x %d x %d' % (nz, ny, nx)

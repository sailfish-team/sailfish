#!/usr/bin/python

"""Converts uncompressed npz files into compressed ones."""

import sys
import numpy as np

for fn in sys.argv[1:]:
    a = np.load(fn)
    np.savez_compressed(fn, **a)

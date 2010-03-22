#!/usr/bin/python

# A simple script to run a Sailfish simulation in the benchmark mode for multiple
# values of the block size.  Example invocation:
#
# ./block_size_tuner.py 16 64 16 ../examples/lbm_ldc.py --lat_nx=256 --lat_ny=128 --max_iters=10000 --every=1000 2>/dev/null
#
# The first three arguments are the starting block size, the ending block size the
# step size.  The remaining arguments indicate the simulation to run.  In the above
# example, the simulation would be run for the following block sizes: 16, 32, 48, 64.

import subprocess
import sys

start = int(sys.argv[1])
end = int(sys.argv[2])
step = int(sys.argv[3])
cmd = sys.argv[4:]

for i in range(start, end+1, step):
    out = subprocess.Popen(cmd + ['--benchmark', '--block_size=%d' % i], stdout=subprocess.PIPE).communicate()[0]
    out = out.split('\n')

    for line in reversed(out):
        if line:
            out = line
            break

    print i, out.split()[-1]

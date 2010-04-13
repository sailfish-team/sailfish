#!/usr/bin/python -u

import os
import sys
import numpy
import math
import matplotlib
import optparse
import glob
import operator
from collections import namedtuple, deque
from optparse import OptionGroup, OptionParser, OptionValueError

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from matplotlib.font_manager import fontManager, FontProperties 
font = FontProperties(size='xx-small')

output = sys.argv[1]
path = sys.argv[2]
files = []

if len(sys.argv) > 3:
    for prefix in sys.argv[3:]:
        files.extend(glob.glob(os.path.join(path, prefix + '*')))
else:
    files.extend(glob.glob(os.path.join(path, '*')))

data = {}
data2 = {}
ids = set()

TestRes = namedtuple('TestRes', 'time mlups')

for file in files:
    fname = os.path.basename(file)
    data[fname] = {}
    with open(file) as fp:
        for line in fp:
            id, testtime, mlups = line.split()
            mlups = float(mlups)
            testtime = float(testtime)
            ids.add(id)

            # If there are multiple tests, take the average of the performance
            # value and keep the latest time.
            if id in data[fname]:
                data[fname][id] = TestRes(testtime, 0.5 * (mlups + data[fname][id].mlups))
            else:
                data[fname][id] = TestRes(testtime, mlups)
            
for k, v in data.items():
    data[k] = deque(sorted(v.items(), key=lambda x: x[1].time))
    data2[k] = list(data[k])

done = False
ordered_ids = []

while not done:
    min_ = sys.maxint
    kmin = None

    for k, v in data.iteritems():
        try:
            if v[0][1].time < min_:
                kmin = v[0][0]
                min_ = v[0][1].time
        except IndexError:
            pass

    if kmin is None:
        break

    ordered_ids.append(kmin)
    for k, v in data.iteritems():
        try:
            if v[0][0] == kmin:
                v.popleft()
        except IndexError:
            pass

id2idx = dict(map(reversed, enumerate(ordered_ids)))

for name, v in sorted(data2.items()):
    xvec = []
    yvec = []

    for id, testres in v:
        xvec.append(id2idx[id])
        yvec.append(testres.mlups)

    plt.plot(xvec, yvec, '.-', label=name)

plt.gca().yaxis.grid(True)
plt.gca().xaxis.grid(True)

plt.xlabel('time (arbitrary units, higher values represent newer commits)')
plt.ylabel('MLUPS')
plt.legend(loc='upper left', prop=font)
#plt.show()
plt.savefig(os.path.join(output), format='pdf')

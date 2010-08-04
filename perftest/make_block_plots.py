import os
import sys
import numpy as np
import matplotlib
import optparse
import glob
from collections import namedtuple, deque
from optparse import OptionGroup, OptionParser, OptionValueError

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from matplotlib.font_manager import fontManager, FontProperties 
font = FontProperties(size='xx-small')

output = sys.argv[1]
path = sys.argv[2]

for file in glob.glob(os.path.join(path, '*')):
    print file

    fname = os.path.basename(file)
    dat = np.loadtxt(file)

    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    ax1.grid('on')
    
    ax1.bar(dat[:,0]-5, dat[:,1], 10)
    ax1.set_xlabel('block size')
    ax1.set_ylabel('MLUPS')

    ax2.bar(dat[:,0]+7, dat[:,2], 10, color='r')
    ax2.set_ylabel('occupancy')
    ax2.xaxis.set_ticks(dat[:,0])


    plt.savefig(os.path.join(output, fname) + '.pdf')
    plt.clf()
    plt.cla()


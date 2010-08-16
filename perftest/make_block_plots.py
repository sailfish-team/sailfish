#!/usr/bin/python
import os
import numpy as np
import matplotlib
import glob
from optparse import OptionParser

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(size='xx-small')

def make_plot(file):
    fig = plt.figure()
    subplot(file, fig)

def subplot(file, fig, num=111):
    if type(num) is tuple:
        ax1 = fig.add_subplot(*num)
    else:
        ax1 = fig.add_subplot(num)
    ax2 = ax1.twinx()

    fname = os.path.basename(file)
    dat = np.loadtxt(file)

    ax1.grid('on')

    ax1.bar(dat[:,0]-5, dat[:,1], 10)
    ax1.set_xlabel('block size')
    ax1.set_ylabel('MLUPS')

    ax2.bar(dat[:,0]+7, dat[:,2], 10, color='r')
    ax2.set_ylabel('occupancy')
    ax2.xaxis.set_ticks(dat[:,0])

    ax2.set_title(fname)

def make_summary(path):
    files = list(sorted(glob.glob(os.path.join(path, '*'))))

    cols = 2
    rows = len(files) / cols
    fig = plt.figure(figsize=(9 * cols, rows * 5))

    for i, file in enumerate(files):
        subplot(file, fig, (rows, cols, i))

    fig.subplots_adjust(left=0.05, bottom=0.015, right=0.95, top=0.985)
    return plt

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('--summary', dest='summary', help='generate a single output file', action='store_true', default=False)
    options, args = parser.parse_args()

    output = args[0]
    path = args[1]

    if options.summary:
        plot = make_summary(path)
        plot.savefig(os.path.join(output, 'block_summary.pdf'))#,
#                bbox_inches='tight')
    else:
        for file in glob.glob(os.path.join(path, '*')):
            print file

            make_plot(file)

            fname = os.path.basename(file)
            plt.savefig(os.path.join(output, fname) + '.pdf')
            plt.clf()
            plt.cla()


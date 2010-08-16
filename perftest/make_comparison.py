#!/usr/bin/python

import os
import sys
import numpy as np
import matplotlib
import glob

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties
font = FontProperties(size='xx-small')

def subplot(fig, devices, model, val, num=111):
    if type(num) is tuple:
        ax1 = fig.add_subplot(*num)
    else:
        ax1 = fig.add_subplot(num)

    locs = np.float32(range(0, len(devices)))

    ax1.grid('on')
    ax1.bar(locs, val, 0.5)
    ax1.set_xlabel('device')
    ax1.set_ylabel('MLUPS')
    ax1.set_xlim(-0.5, len(locs))
    ax1.set_xticks(locs + 0.25)
    ax1.set_xticklabels(devices)
    ax1.set_title(model)

def comparison_plot(path):
    devices = sorted(os.listdir(path))
    perf_map = {}

    for device in devices:
        base = os.path.join(path, device, 'blocksize')
        for file in glob.glob(os.path.join(base, '*')):
            model = os.path.basename(file)
            data = np.loadtxt(file)

            perf_map.setdefault(model, []).append(np.max(data[:,1]))

    cols = 2
    rows = len(perf_map.keys()) / cols
    fig = plt.figure(figsize=(9 * cols, rows * 5))

    for i, (model, val) in enumerate(sorted(perf_map.iteritems())):
        subplot(fig, devices, model, val, (rows, cols, i))

    fig.subplots_adjust(left=0.05, bottom=0.015, right=0.95, top=0.985)

if __name__ == '__main__':
    output = sys.argv[1]
    path = sys.argv[2]
    comparison_plot(path)
    plt.savefig(output)

import os

import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt


def make_subplot(file, title_):
    data = pylab.csv2rec(file, delimiter=' ', names=('re', 'sim', 'th'))
    figW = 5.5
    figH = 4.5
    fig = plt.figure(subplotpars=mpl.figure.SubplotParams(left=0.125, bottom=0.130))
    ax = fig.add_subplot(111)

    pl_sim = ax.loglog(data['re'], data['sim'], 'bo-')
    pl_th = ax.loglog(data['re'], data['th'], 'ro-')
    ax.legend((pl_sim, pl_th), ('simulation results', 'theoretical approximation'), 'best')
    ax.grid('.')
    ax.grid(which='minor')
    ax.set_xlim(data[0][0], data[-1][0])
    ax.set_ylabel('drag coefficient')
    ax.set_xlabel('Reynolds number')
    ax.set_title(title_)

def make_plot(basepath, files, titles):
    for i, sp in enumerate(zip(files, titles)):
        make_subplot(os.path.join(basepath, sp[0]), sp[1])





import os

from pylab import *

def make_subplot(file, title_):
    data = csv2rec(file, delimiter=' ', names=('visc', 'y'))
    subplots_adjust(left=0.16, bottom=0.13, right=0.95)
    semilogx(data['visc'], data['y'], 'bo-')
    grid('.')
    grid(which='minor')
    xlim(data[0][0], data[-1][0])
    ylabel('max velocity / theoretical max velocity - 1')
    xlabel('viscosity')
    title(title_)


def make_plot(basepath, files, titles):
    for i, sp in enumerate(zip(files, titles)):
        figure(i+1)
        make_subplot(os.path.join(basepath, sp[0]), sp[1])
    show()





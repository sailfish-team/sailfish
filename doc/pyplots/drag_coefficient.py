import os

from pylab import *

def make_subplot(file, title_):
	data = csv2rec(file, delimiter=' ', names=('re', 'sim', 'th'))
	loglog(data['re'], data['sim'], 'bo-')
	loglog(data['re'], data['th'], 'ro-')
	grid('.')
	grid(which='minor')
	xlim(data[0][0], data[-1][0])
	ylabel('drag coefficient')
	xlabel('Reynolds number')
	title(title_)

def make_plot(basepath, files, titles):
    for i, sp in enumerate(zip(files, titles)):
        figure(i+1)
        make_subplot(os.path.join(basepath, sp[0]), sp[1])
    show()





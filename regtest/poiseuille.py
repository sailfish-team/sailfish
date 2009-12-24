#!/usr/bin/python -u

import os
import sys
import numpy
import math
import matplotlib
import optparse
from optparse import OptionGroup, OptionParser, OptionValueError

matplotlib.use('cairo')
import matplotlib.pyplot as plt

sys.path.append('.')
from lbm_poiseuille import LPoiSim, LBMGeoPoiseuille
import geo

MAX_ITERS = 10000
POINTS = 30

class LTestPoiSim(LPoiSim):
	def __init__(self, args, defaults):
		super(LTestPoiSim, self).__init__(LBMGeoPoiseuille, args, defaults)
		self.clear_hooks()
		self.add_iter_hook(self.options.max_iters-1, self.save_output)

	def save_output(self):
#		self.result = (numpy.max(self.vy[16,1:self.geo.lat_w-1]) / max(self.geo.get_velocity_profile())) - 1.0
		self.res_maxv = numpy.max(self.geo.mask_array_by_fluid(self.vy))
		self.th_maxv = max(self.geo.get_velocity_profile())
		self.result = self.res_maxv / self.th_maxv - 1.0

bcs = [x.name for x in geo.SUPPORTED_BCS if geo.LBMGeo.NODE_WALL in x.supported_types]
defaults = {
		'stationary': True,
		'batch': True,
		'quiet': True,
		'test': True,
		'lat_w': 64,
		'lat_h': 64,
	}

def run_test(bc, drive, precision):
	xvec = []
	yvec = []
	prof_sim = []
	prof_th = []

	basepath = os.path.join('regtest/results/poiseuille', drive, precision)

	if not os.path.exists(basepath):
		os.makedirs(basepath)

	f = open(os.path.join(basepath, '%s.dat' % bc), 'w')

	print '* Testing "%s" for visc' % bc,

	for visc in numpy.logspace(-3, -1, num=POINTS):
		print '%f ' % visc,

		iters = int(1000 / visc)
		xvec.append(visc)

		defaults['bc_wall'] = bc
		defaults['visc'] = visc
		defaults['max_iters'] = iters
		defaults['precision'] = precision

		sim = LTestPoiSim([], defaults)
		sim.run()

		yvec.append(sim.result)

		prof_sim.append(sim.get_profile())
		prof_th.append(sim.geo.get_velocity_profile())

		print >>f, visc, sim.result

	print


	plt.gca().yaxis.grid(True)
	plt.gca().xaxis.grid(True)
	plt.gca().xaxis.grid(True, which='minor')
	plt.gca().yaxis.grid(True, which='minor')
	plt.gca().set_xbound(0, len(prof_sim[0])-1)

	for i in range(0, len(prof_sim)):
		plt.clf()
		plt.plot(prof_th[i] - prof_sim[i], 'bo-')
		plt.title('visc = %f' % xvec[i])
		plt.savefig(os.path.join(basepath, '%s-profile%d.pdf' % (bc, i)), format='pdf')

		f2 = open(os.path.join(basepath, '%s-profile%d.dat' % (bc, i)), 'w')
		for j in range(0, len(prof_sim[i])):
			print >>f2, prof_sim[i][j], prof_th[i][j]
		f2.close()

		plt.clf()
		plt.cla()

	f.close()

	plt.clf()
	plt.cla()

	plt.semilogx(xvec, yvec, 'bo-')
	plt.title('%d iters' % MAX_ITERS)
	plt.gca().yaxis.grid(True)
	plt.gca().yaxis.grid(True, which='minor')
	plt.gca().xaxis.grid(True)
	plt.gca().xaxis.grid(True, which='minor')
	plt.ylabel('max velocity / theoretical max velocity - 1')
	plt.xlabel('viscosity')
	plt.savefig(os.path.join(basepath, '%s.pdf' % bc), format='pdf')

parser = OptionParser()
parser.add_option('--precision', dest='precision', help='precision (single, double)', type='choice', choices=['single', 'double'], default='single')
(options, args) = parser.parse_args()

print 'Running tests for %s precision' % options.precision

for bc in bcs:
	run_test(bc, 'force', options.precision)

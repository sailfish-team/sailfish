#!/usr/bin/python -u

import sys
import numpy
import math
import matplotlib

matplotlib.use('cairo')
import matplotlib.pyplot as plt

sys.path.append('.')
from lbm_poiseuille import LPoiSim, LBMGeoPoiseuille
import geo

MAX_ITERS = 50000
POINTS = 30

class LTestPoiSim(LPoiSim):
	def __init__(self, visc, bc, static=False, lat_w=64, lat_h=64, max_iters=MAX_ITERS):
		args = ['--test', '--visc=%f' % visc, '--quiet', '--boundary=%s' % bc,
				'--lat_w=%d' % lat_w, '--lat_h=%d' % lat_h, '--batch']
		if static:
			args.append('--static')
		super(LTestPoiSim, self).__init__(LBMGeoPoiseuille, args)
		self.clear_hooks()
		self.options.max_iters = max_iters
		self.add_iter_hook(self.options.max_iters-1, self.save_output)

	def save_output(self):
		self.result = numpy.max(self.vy[16,1:self.geo.lat_w-1]) / max(self.geo.get_velocity_profile())
		self.res_maxv = numpy.max(self.geo.mask_array_by_fluid(self.vy))
		self.th_maxv = max(self.geo.get_velocity_profile())

bcs = geo.BCS_MAP.keys()

for bc in bcs:
	xvec = []
	yvec = []
	yvec2 = []
	prof_sim = []
	prof_th = []

	f = open('regtest/results/poiseuille-%s.dat' % bc, 'w')

	print '* Testing "%s" for visc' % bc,

	for visc in numpy.logspace(-3, -1, num=POINTS):
		print '%f ' % visc,

		sim = LTestPoiSim(visc, bc)
		sim.run()

		xvec.append(visc)
		yvec.append(sim.result)

		sim2 = LTestPoiSim(visc, bc, static=True)
		sim2.run()

		yvec2.append(sim2.result)

		prof_sim.append(sim2.get_profile())
		prof_th.append(sim2.geo.get_velocity_profile())

		print >>f, visc, sim.result, sim2.result

	print

	f.close()

	plt.clf()
	plt.cla()

	args = []

	plt.gca().yaxis.grid(True)
	plt.gca().xaxis.grid(True)
	plt.gca().xaxis.grid(True, which='minor')
	plt.gca().yaxis.grid(True, which='minor')
	plt.gca().set_xbound(0, len(prof_sim[0])-1)

	for i in range(0, len(prof_sim)):
		plt.clf()
		plt.plot(prof_th[i] - prof_sim[i], 'bo-')
		plt.title('visc = %f' % xvec[i])
		plt.savefig('regtest/results/poiseuille-%s-profile%d.pdf' % (bc, i), format='pdf')

	plt.clf()
	plt.cla()

	plt.semilogx(xvec, yvec, 'bo-')
	plt.title('%d iters' % MAX_ITERS)
	plt.gca().yaxis.grid(True)
	plt.gca().yaxis.grid(True, which='minor')
	plt.gca().xaxis.grid(True)
	plt.gca().xaxis.grid(True, which='minor')
	plt.ylabel('max velocity / theoretical max velocity')
	plt.xlabel('viscosity')
	plt.savefig('regtest/results/poiseuille-%s.pdf' % bc, format='pdf')

	plt.clf()
	plt.semilogx(xvec, yvec2, 'bo-')
	plt.title('%d iters' % MAX_ITERS)
	plt.ylabel('max velocity / theoretical max velocity')
	plt.xlabel('viscosity')
	plt.savefig('regtest/results/poiseuille-%s-static.pdf' % bc, format='pdf')

	xvec = []
	yvec = []

	print '  Testing error scaling'

	for lat_w in range(64, 1024, 64):
		sim = LTestPoiSim(0.005, bc, static=True, lat_w=lat_w, lat_h=32)
		sim.run()

		rel_err = math.sqrt(numpy.max(
			numpy.power(sim.vx[16,1:lat_w-1], 2) +
			numpy.power(sim.vy[16,1:lat_w-1] - sim.geo.get_velocity_profile()[1:lat_w-1], 2))) / LBMGeoPoiseuille.maxv

		xvec.append(1.0/sim.geo.get_chan_width())
		yvec.append(rel_err)

	f = open('regtest/results/poiseuille-%s-error.dat' % bc, 'w')
	for x, y in zip(xvec, yvec):
		print >>f, x, y
	f.close()

	plt.clf()
	plt.plot(xvec, yvec, 'bo-')
	plt.title('Relative error')
	plt.ylabel('error')
	plt.xlabel('lattice spacing')
	plt.savefig('regtest/results/poiseuille-%s-error.pdf' % bc, format='pdf')

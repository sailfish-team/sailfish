#!/usr/bin/python -u

import sys
import numpy
import matplotlib

matplotlib.use('cairo')
import matplotlib.pyplot as plt

sys.path.append('.')
from lbm_poiseuille import LPoiSim, LBMGeoPoiseuille

MAX_ITERS = 50000

class LTestPoiSim(LPoiSim):
	def __init__(self, visc):
		args = ['--test', '--visc=%f' % visc, '--quiet']
		super(LTestPoiSim, self).__init__(LBMGeoPoiseuille, args)
		self.clear_hooks()
		self.options.max_iters = MAX_ITERS
		self.add_iter_hook(self.options.max_iters-1, self.save_output)

	def save_output(self):
		self.result = numpy.max(self.geo.mask_array_by_fluid(self.vy)) / max(self.geo.get_velocity_profile())

xvec = []
yvec = []

f = open('regtest/results/poiseuille.dat', 'w')

for visc in numpy.logspace(-3, -1, num=10):
	sim = LTestPoiSim(visc)
	sim.run()

	xvec.append(visc)
	yvec.append(sim.result)

	print >>f, visc, sim.result

f.close()

prof_sim = sim.get_profile()
prof_th = sim.geo.get_velocity_profile()

plt.plot(prof_th - prof_sim, 'ro-')
plt.title('Velocity profile difference for visc = %f' % xvec[-1])
plt.gca().yaxis.grid(True)
plt.gca().yaxis.grid(True, which='minor')
plt.gca().set_xbound(0, len(prof_sim)-1)
plt.savefig('regtest/results/poiseuille-profile.pdf', format='pdf')

plt.clf()
plt.cla()

plt.semilogx(xvec, yvec, 'bo-')
plt.title('Simulation convergence at %d iters' % MAX_ITERS)
plt.gca().yaxis.grid(True)
plt.gca().yaxis.grid(True, which='minor')
plt.gca().xaxis.grid(True)
plt.gca().xaxis.grid(True, which='minor')
plt.ylabel('max velocity / theoretical max velocity')
plt.xlabel('viscosity')
plt.savefig('regtest/results/poiseuille.pdf', format='pdf')



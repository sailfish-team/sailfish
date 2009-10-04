#!/usr/bin/python -u

import sys
import numpy

sys.path.append('.')
from lbm_poiseuille import LPoiSim, LBMGeoPoiseuille

result = 0

class LTestPoiSim(LPoiSim):
	def __init__(self, visc):
		args = ['--test', '--visc=%f' % visc, '--quiet']
		super(LTestPoiSim, self).__init__(LBMGeoPoiseuille, args)
		self.clear_hooks()
		self.options.max_iters = 50000
		self.add_iter_hook(self.options.max_iters-1, self.save_output)

	def save_output(self):
		global result
		result = numpy.max(self.geo.mask_array_by_fluid(self.vy)) / self.geo.maxv

for visc in numpy.logspace(-3, -1, num=10):
	sim = LTestPoiSim(visc)
	sim.run()

	print visc, result





#!/usr/bin/python -u

import os
import sys
import numpy
import math
import matplotlib
import optparse
from optparse import OptionGroup, OptionParser, OptionValueError
import time

import git
import pycuda
import pycuda.autoinit

matplotlib.use('cairo')
import matplotlib.pyplot as plt

from sailfish import geo
from sailfish import sym

from examples import lbm_ldc
from examples import lbm_poiseuille
from examples import lbm_poiseuille_3d

# Default settings.
defaults = {
    'benchmark': True,
    'quiet': True,
    'verbose': False,
    'max_iters': 10000,
    'every': 1000
}

# Tests to run.
tests = {
    '2d_ldc_small': {
        'options': {'lat_nx': 128, 'lat_ny': 128},
        'run': lambda settings: lbm_ldc.LDCSim(lbm_ldc.LBMGeoLDC, settings),
    },

    '2d_ldc_large': {
        'options': {'lat_nx': 1024, 'lat_ny': 1024},
        'run': lambda settings: lbm_ldc.LDCSim(lbm_ldc.LBMGeoLDC, settings),
    },

    '2d_poiseuille_small': {
        'options': {'lat_nx': 128, 'lat_ny': 128},
        'run': lambda settings: lbm_poiseuille.LPoiSim(lbm_poiseuille.LBMGeoPoiseuille, defaults=settings),  
    },

    '2d_poiseuille_large': {
        'options': {'lat_nx': 1024, 'lat_ny': 1024},
        'run': lambda settings: lbm_poiseuille.LPoiSim(lbm_poiseuille.LBMGeoPoiseuille, defaults=settings),  
    },

    '3d_poiseuille_d3q13': {
        'options': {'lat_nx': 128, 'lat_ny': 128, 'lat_nz': 128, 'grid': 'D3Q13'},
        'run': lambda settings: lbm_poiseuille_3d.LPoiSim(lbm_poiseuille_3d.LBMGeoPoiseuille, defaults=settings),
    },

    '3d_poiseuille_d3q15': {
        'options': {'lat_nx': 128, 'lat_ny': 128, 'lat_nz': 128, 'grid': 'D3Q15'},
        'run': lambda settings: lbm_poiseuille_3d.LPoiSim(lbm_poiseuille_3d.LBMGeoPoiseuille, defaults=settings),
    },

    '3d_poiseuille_d3q19': {
        'options': {'lat_nx': 128, 'lat_ny': 128, 'lat_nz': 128, 'grid': 'D3Q19'},
        'run': lambda settings: lbm_poiseuille_3d.LPoiSim(lbm_poiseuille_3d.LBMGeoPoiseuille, defaults=settings),
    },
}

repo = git.Repo('.')
head = repo.commits()[0]

def run_test(name):
    global tests, defaults, head

    if name not in tests:
        raise ValueError('Test %s not found' % name)

    settings = {}
    settings.update(defaults)
    settings.update(tests[name]['options'])
    sim = tests[name]['run'](settings)
    sim.run()

    basepath = os.path.join('perftest', 'results', pycuda.autoinit.device.name().replace(' ','_'))
    path = os.path.join(basepath, name)
    if not os.path.exists(basepath):
        os.makedirs(basepath)

    f = open(path, 'a')
    print >>f, head.id, time.time(), sim._bench_avg
    f.close()

if len(sys.argv) > 1:
    for name in sys.argv[1:]:
        run_test(name)
else:
    for name in tests.iterkeys():
        run_test(name)

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
from examples import sc_phase_separation
from examples.binary_fluid import sc_separation_2d
from examples.binary_fluid import fe_separation_2d
from examples.binary_fluid import fe_viscous_fingering

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

    '2d_sc_phase_sep_small': {
        'options': {'lat_nx': 128, 'lat_ny': 128},
        'run': lambda settings: sc_phase_separation.SCSim(sc_phase_separation.GeoSC, defaults=settings),
    },

    '2d_sc_phase_sep_large': {
        'options': {'lat_nx': 1024, 'lat_ny': 1024},
        'run': lambda settings: sc_phase_separation.SCSim(sc_phase_separation.GeoSC, defaults=settings),
    },

    '2d_bin_sc_phase_sep_small': {
        'options': {'lat_nx': 128, 'lat_ny': 128},
        'run': lambda settings: sc_separation_2d.SCSim(sc_separation_2d.GeoSC, defaults=settings),
    },

    '2d_bin_sc_phase_sep_large': {
        'options': {'lat_nx': 1024, 'lat_ny': 1024},
        'run': lambda settings: sc_separation_2d.SCSim(sc_separation_2d.GeoSC, defaults=settings),
    },

    '2d_bin_fe_sep_small': {
        'options': {'lat_nx': 128, 'lat_ny': 128},
        'run': lambda settings: fe_separation_2d.FESim(fe_separation_2d.GeoFE, defaults=settings),
    },

    '2d_bin_fe_sep_large': {
        'options': {'lat_nx': 1024, 'lat_ny': 1024},
        'run': lambda settings: fe_separation_2d.FESim(fe_separation_2d.GeoFE, defaults=settings),
    },

    '3d_bin_fe_fingering': {
        'options': {'lat_nx': 448, 'lat_ny': 48, 'lat_nz': 38},
        'run': lambda settings: fe_viscous_fingering.FEFingerSim(fe_viscous_fingering.GeoFEFinger, defaults=settings),
    },
}

repo = git.Repo('.')
head = repo.commits()[0]

def run_test(name):
    global tests, defaults, head

    print '* %s' % name

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
    done = set()

    for name in sys.argv[1:]:
        if name in tests:
            run_test(name)
        else:
            # Treat test name as a prefix if an exact match has not been found.
            for x in tests:
                if len(name) < len(x) and name == x[0:len(name)] and x not in done:
                    run_test(x)
                    done.add(x)
else:
    for name in tests.iterkeys():
        run_test(name)

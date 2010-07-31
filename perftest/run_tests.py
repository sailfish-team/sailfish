#!/usr/bin/python -u

# Usage: perftest/run_tests.py <testsuite_name>

import sys

from examples import lbm_ldc
from examples import lbm_poiseuille
from examples import lbm_poiseuille_3d
from examples import sc_phase_separation
from examples.binary_fluid import sc_separation_2d
from examples.binary_fluid import fe_separation_2d
from examples.binary_fluid import fe_viscous_fingering

from models import single_fluid

from tests import run_suite

model_tests = {
    'd2q9_bgk': {
        'options': {'lat_nx': 512, 'lat_ny': 512, 'model': 'bgk', 'grid': 'D2Q9'},
        'run': lambda settings: single_fluid.TestSim(single_fluid.TestGeo2D, settings),
    },

    'd2q9_mrt': {
        'options': {'lat_nx': 512, 'lat_ny': 512, 'model': 'mrt', 'grid': 'D2Q9'},
        'run': lambda settings: single_fluid.TestSim(single_fluid.TestGeo2D, settings),
    },

    'd3q13_mrt': {
        'options': {'lat_nx': 128, 'lat_ny': 64, 'lat_nz': 64, 'model': 'mrt', 'grid': 'D3Q13'},
        'run': lambda settings: single_fluid.TestSim(single_fluid.TestGeo3D, settings),
    },

    'd3q15_bgk': {
        'options': {'lat_nx': 128, 'lat_ny': 64, 'lat_nz': 64, 'model': 'bgk', 'grid': 'D3Q15'},
        'run': lambda settings: single_fluid.TestSim(single_fluid.TestGeo3D, settings),
    },

    'd3q15_mrt': {
        'options': {'lat_nx': 128, 'lat_ny': 64, 'lat_nz': 64, 'model': 'mrt', 'grid': 'D3Q15'},
        'run': lambda settings: single_fluid.TestSim(single_fluid.TestGeo3D, settings),
    },

    'd3q19_bgk': {
        'options': {'lat_nx': 128, 'lat_ny': 64, 'lat_nz': 64, 'model': 'bgk', 'grid': 'D3Q19'},
        'run': lambda settings: single_fluid.TestSim(single_fluid.TestGeo3D, settings),
    },

    'd3q19_mrt': {
        'options': {'lat_nx': 128, 'lat_ny': 64, 'lat_nz': 64, 'model': 'mrt', 'grid': 'D3Q19'},
        'run': lambda settings: single_fluid.TestSim(single_fluid.TestGeo3D, settings),
    },

}

# Tests to run.
example_tests = {
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

args = sys.argv[1:]
suite = globals()[args[0]]
run_suite(suite, args[1:])

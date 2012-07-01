#!/usr/bin/python
"""Verifies translation invariance for binary fluids.

In simulations with periodic boundary conditions, when the whole domain is
shifted by constant vector, the result should be the same up to that constant
shift.  This test verifies that this is indeed the case.
"""
import os
import shutil
import tempfile
import unittest

import numpy as np

from sailfish.geo import LBGeometry2D, LBGeometry3D
from sailfish.subdomain import Subdomain2D, Subdomain3D
from sailfish.controller import LBSimulationController
from sailfish.lb_binary import LBBinaryFluidShanChen, LBBinaryFluidFreeEnergy
from sailfish import io
from regtest.subdomains import util


SHIFT = [17, 4, 13]
output = ''
tmpdir = tempfile.mkdtemp()
mem_align = 32
block_size = 64
MAX_ITERS_2D = 1000
MAX_ITERS_3D = 100

def match_fields(reference, shifted):
    for axis in range(0, len(reference.shape)):
        shifted = np.roll(shifted, -SHIFT[axis], axis)
    np.testing.assert_array_almost_equal(reference, shifted)

def shift_array(array):
    for axis in range(0, len(array.shape)):
        array = np.roll(array, SHIFT[axis], axis)
    return array

#
# Shan-Chen model.
#

class SCTestDomain2D(Subdomain2D):
    shift = False

    def initial_conditions(self, sim, hx, hy):
        np.random.seed(1234);

        rho = 1.0 + np.random.rand(*sim.rho.shape) / 1000.0
        phi = 1.0 + np.random.rand(*sim.phi.shape) / 1000.0

        if self.shift:
            rho = shift_array(rho)
            phi = shift_array(phi)

        sim.rho[:] = rho
        sim.phi[:] = phi

    def boundary_conditions(self, hx, hy):
        pass


class SCTestDomain3D(Subdomain3D):
    shift = False

    def initial_conditions(self, sim, hx, hy, hz):
        np.random.seed(1234);

        rho = 1.0 + np.random.rand(*sim.rho.shape) / 1000.0
        phi = 1.0 + np.random.rand(*sim.phi.shape) / 1000.0

        if self.shift:
            rho = shift_array(rho)
            phi = shift_array(phi)

        sim.rho[:] = rho
        sim.phi[:] = phi

    def boundary_conditions(self, hx, hy, hz):
        pass


class SCTestSim2D(LBBinaryFluidShanChen):
    subdomain = SCTestDomain2D

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'cuda_cache': False,
            'lat_nx': 254,
            'lat_ny': 230,
            'grid': 'D2Q9',
            'G': 1.2,
            'visc': 1.0/6.0,
            'periodic_x': True,
            'periodic_y': True,
            'output': output,
            'quiet': True,
            'max_iters': MAX_ITERS_2D,
            'every': 500})

class SCTestSim3D(LBBinaryFluidShanChen):
    subdomain = SCTestDomain3D

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'cuda_cache': False,
            'lat_nx': 62,
            'lat_ny': 48,
            'lat_nz': 40,
            'grid': 'D3Q19',
            'G': 1.2,
            'visc': 1.0/6.0,
            'periodic_x': True,
            'periodic_y': True,
            'periodic_z': True,
            'output': output,
            'quiet': True,
            'max_iters': MAX_ITERS_3D,
            'every': 50})


class TestShanChenShift(unittest.TestCase):
    def test_shift_2d(self):
        global output
        output = os.path.join(tmpdir, 'baseline')
        ctrl = LBSimulationController(SCTestSim2D, LBGeometry2D).run(ignore_cmdline=True)
        digits = io.filename_iter_digits(MAX_ITERS_2D)
        ref = np.load(io.filename(output, digits, 0, MAX_ITERS_2D))

        output = os.path.join(tmpdir, 'shifted')
        SCTestDomain2D.shift = True
        LBSimulationController(SCTestSim2D, LBGeometry2D).run(ignore_cmdline=True)
        shifted = np.load(io.filename(output, digits, 0, MAX_ITERS_2D))
        match_fields(ref['rho'], shifted['rho'])
        match_fields(ref['phi'], shifted['phi'])
        self.assertFalse(np.any(np.isnan(ref['rho'])))
        self.assertFalse(np.any(np.isnan(ref['phi'])))
        self.assertFalse(np.any(np.isnan(shifted['rho'])))
        self.assertFalse(np.any(np.isnan(shifted['phi'])))

    def test_shift_3d(self):
        global output
        output = os.path.join(tmpdir, 'baseline')
        ctrl = LBSimulationController(SCTestSim3D, LBGeometry3D).run(ignore_cmdline=True)
        digits = io.filename_iter_digits(MAX_ITERS_3D)
        ref = np.load(io.filename(output, digits, 0, MAX_ITERS_3D))

        output = os.path.join(tmpdir, 'shifted')
        SCTestDomain3D.shift = True
        LBSimulationController(SCTestSim3D, LBGeometry3D).run(ignore_cmdline=True)
        shifted = np.load(io.filename(output, digits, 0, MAX_ITERS_3D))
        match_fields(ref['rho'], shifted['rho'])
        match_fields(ref['phi'], shifted['phi'])
        self.assertFalse(np.any(np.isnan(ref['rho'])))
        self.assertFalse(np.any(np.isnan(ref['phi'])))
        self.assertFalse(np.any(np.isnan(shifted['rho'])))
        self.assertFalse(np.any(np.isnan(shifted['phi'])))
#
# Free-energy model.
#

class FETestDomain2D(Subdomain2D):
    shift = False

    def initial_conditions(self, sim, hx, hy):
        np.random.seed(1234)

        phi = np.random.rand(*sim.phi.shape) / 100.0
        if self.shift:
            phi = shift_array(phi)

        sim.rho[:] = 1.0
        sim.phi[:] = phi

    def boundary_conditions(self, hx, hy):
        pass


class FETestDomain3D(Subdomain3D):
    shift = False

    def initial_conditions(self, sim, hx, hy, hz):
        np.random.seed(1234)

        phi = np.random.rand(*sim.phi.shape) / 100.0
        if self.shift:
            phi = shift_array(phi)

        sim.rho[:] = 1.0
        sim.phi[:] = phi

    def boundary_conditions(self, hx, hy, hz):
        pass


class FETestSim2D(LBBinaryFluidFreeEnergy):
    subdomain = FETestDomain2D

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'cuda_cache': False,
            'lat_nx': 254,
            'lat_ny': 256,
            'grid': 'D2Q9',
            'kappa': 2e-4,
            'Gamma': 25.0,
            'A': 1e-4,
            'tau_a': 4.5,
            'tau_b': 0.8,
            'tau_phi': 1.0,
            'periodic_x': True,
            'periodic_y': True,
            'output': output,
            'quiet': True,
            'max_iters': 1000,
            'every': 500})


class FETestSim3D(LBBinaryFluidFreeEnergy):
    subdomain = FETestDomain3D

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'block_size': block_size,
            'mem_alignment': mem_align,
            'cuda_cache': False,
            'lat_nx': 62,
            'lat_ny': 48,
            'lat_nz': 40,
            'grid': 'D3Q19',
            'kappa': 2e-4,
            'Gamma': 25.0,
            'A': 1e-4,
            'tau_a': 4.5,
            'tau_b': 0.8,
            'tau_phi': 1.0,
            'periodic_x': True,
            'periodic_y': True,
            'periodic_z': True,
            'output': output,
            'quiet': True,
            'max_iters': 100,
            'every': 50})


class TestFreeEnergyShift(unittest.TestCase):
    def test_shift_2d(self):
        global output
        output = os.path.join(tmpdir, 'baseline')
        LBSimulationController(FETestSim2D, LBGeometry2D).run(ignore_cmdline=True)
        digits = io.filename_iter_digits(MAX_ITERS_2D)
        ref = np.load(io.filename(output, digits, 0, MAX_ITERS_2D))

        output = os.path.join(tmpdir, 'shifted')
        FETestDomain2D.shift = True
        LBSimulationController(FETestSim2D, LBGeometry2D).run(ignore_cmdline=True)
        shifted = np.load(io.filename(output, digits, 0, MAX_ITERS_2D))
        match_fields(ref['rho'], shifted['rho'])
        match_fields(ref['phi'], shifted['phi'])
        self.assertFalse(np.any(np.isnan(ref['rho'])))
        self.assertFalse(np.any(np.isnan(ref['phi'])))
        self.assertFalse(np.any(np.isnan(shifted['rho'])))
        self.assertFalse(np.any(np.isnan(shifted['phi'])))

    def test_shift_3d(self):
        global output
        output = os.path.join(tmpdir, 'baseline')
        LBSimulationController(FETestSim3D, LBGeometry3D).run(ignore_cmdline=True)
        digits = io.filename_iter_digits(MAX_ITERS_3D)
        ref = np.load(io.filename(output, digits, 0, MAX_ITERS_3D))

        output = os.path.join(tmpdir, 'shifted')
        FETestDomain3D.shift = True
        LBSimulationController(FETestSim3D, LBGeometry3D).run(ignore_cmdline=True)
        shifted = np.load(io.filename(output, digits, 0, MAX_ITERS_3D))
        match_fields(ref['rho'], shifted['rho'])
        match_fields(ref['phi'], shifted['phi'])
        self.assertFalse(np.any(np.isnan(ref['rho'])))
        self.assertFalse(np.any(np.isnan(ref['phi'])))
        self.assertFalse(np.any(np.isnan(shifted['rho'])))
        self.assertFalse(np.any(np.isnan(shifted['phi'])))

def tearDownModule():
    shutil.rmtree(tmpdir)


if __name__ == '__main__':
    args = util.parse_cmd_line()
    block_size = args.block_size
    if block_size < mem_align:
        mem_align = block_size
    unittest.main()


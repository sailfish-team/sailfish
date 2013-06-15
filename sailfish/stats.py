"""Auxilliary classes for computing flow statistics."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from sailfish.lb_base import ScalarField, LBMixIn
import numpy as np

class FlowStatsMixIn(LBMixIn):
    """When mixed with an LBFluidSim-descendant class, provides easy access
    to various flow statistics"""


class KineticEnergyEnstrophyMixIn(FlowStatsMixIn):
    aux_code = ['data_processing.mako']

    @classmethod
    def fields(cls):
        return [ScalarField('v_sq', gpu_array=True, init=0.0),
                ScalarField('vort_sq', gpu_array=True, init=0.0)]

    def before_main_loop(self, runner):
        gpu_map = runner.gpu_geo_map()
        gpu_v = runner.gpu_field(self.v)
        gpu_vsq = runner.gpu_field(self.v_sq)
        gpu_vortsq = runner.gpu_field(self.vort_sq)

        self._vsq_vort_kernel = runner.get_kernel(
            'ComputeSquareVelocityAndVorticity',
            [gpu_map] + gpu_v + [gpu_vsq, gpu_vortsq],
            'PPPPPP')
        self._arr_vsq = runner.backend.get_array(gpu_vsq)
        self._arr_vortsq = runner.backend.get_array(gpu_vortsq)

    def compute_ke_enstropy(self, runner):
        """Computes kinetic energy and estrophy densities on the compute device.

        :rvalue: kinetic energy, enstrophy (per node)
        """
        b = runner.backend
        b.run_kernel(self._vsq_vort_kernel, runner._kernel_grid_full)
        div = 2.0 * runner._spec.num_nodes

        # Compute the sum in double precision.
        kinetic_energy = b.array.sum(self._arr_vsq,
                                     dtype=np.float64).get() / div
        enstrophy = b.array.sum(self._arr_vortsq,
                                dtype=np.float64).get() / div

        return kinetic_energy, enstrophy

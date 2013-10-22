"""2D flow around a square cylinder in a channel.

Lift and drag coefficients of the cylinder are measured using the
momentum exchange method.

Fully developed parabolic profile is prescribed at the inflow and
a constant pressure condition is prescribed at the outflow.

The results can be compared with:
    [1] M. Breuer, J. Bernsdorf, T. Zeiser, F. Durst
    Accurate computations of the laminar flow past a square cylinder
    based on two different methods: lattice-Boltzmann and finite-volume
    Int. J. of Heat and Fluid Flow 21 (2000) 186-196.
"""

import numpy as np
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTHalfBBWall, NTEquilibriumVelocity, NTEquilibriumDensity, DynamicValue, NTFullBBWall
from sailfish.controller import LBSimulationController
from sailfish.lb_base import ForceObject
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S

# Geometry set to match [1].
H = 160
L = int(6.25 * H)
D = int(0.02 * L)
visc = 0.016666666666666666

# St = f D / u_max

class BoxSubdomain(Subdomain2D):
    bc = NTHalfBBWall
    max_v = 0.05

    def boundary_conditions(self, hx, hy):
        walls = (hy == 0) | (hy == self.gy - 1)
        self.set_node(walls, NTHalfBBWall)

        hhy = S.gy - self.bc.location
        self.set_node((hx == 0) & np.logical_not(walls),
                      NTEquilibriumVelocity(
                          DynamicValue(4.0 * self.max_v / H**2 * hhy * (H - hhy), 0.0)))
        self.set_node((hx == self.gx - 1) & np.logical_not(walls),
                      NTEquilibriumDensity(1))
        l = L / 4

        box = ((hx > l - D/2) & (hx <= l + D/2) &
               (hy > (H - D) / 2) & (hy <= (H + D) / 2))
        self.set_node(box, NTFullBBWall)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vy[:] = 0.0

        hhy = hy - self.bc.location
        sim.vx[:] = 4.0 * self.max_v / H**2 * hhy * (H - hhy)


class BoxSimulation(LBFluidSim):
    subdomain = BoxSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': L,
            'lat_ny': H,
            'precision': 'double',
            'max_iters': 1000000,
            'visc': visc})

    def __init__(self, *args, **kwargs):
        super(BoxSimulation, self).__init__(*args, **kwargs)

        margin = 5
        self.add_force_oject(ForceObject(
            (L / 4 - D / 2 - margin, (H - D) / 2 - margin),
            (L / 4 + D / 2 + margin, (H + D) / 2 + margin)))

        print '%d x %d | box: %d' % (L, H, D)
        print 'Re = %2.f' % (BoxSubdomain.max_v * D / self.config.visc)

    def record_value(self, iteration, force, C_D, C_L):
        print runner._sim.iteration, force[0], force[1], C_D, C_L

    prev_f = None
    every = 500
    def after_step(self, runner):

        if self.iteration % self.every == 0:
            runner.update_force_objects()
            for fo in self.force_objects:
                runner.backend.from_buf(fo.gpu_force_buf)
                f = fo.force()

                # Compute drag and lift coefficients.
                C_D = (2.0 * f[0] / (D * BoxSubdomain.max_v**2))
                C_L = (2.0 * f[1] / (D * BoxSubdomain.max_v**2))
                self.record_value(runner._sim.iteration, f, C_D, C_L)

                if self.prev_f is None:
                    self.prev_f = np.array(f)
                else:
                    f = np.array(f)

                    # Terminate simulation when steady state has
                    # been reached.
                    diff = np.abs(f - self.prev_f) / np.abs(f)

                    if np.all(diff < 1e-6):
                        runner._quit_event.set()
                    self.prev_f = f



if __name__ == '__main__':
    ctrl = LBSimulationController(BoxSimulation)
    ctrl.run()

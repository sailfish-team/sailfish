#!/usr/bin/env python -u
"""
Simulates the flow around a 2D cylinder using the Immersed Boundary Method.

The domain is not periodic and the flow is driven by a body force.

The simulation results can be used to compute the Strouhal number (f D / u),
where D is the cylinder diameter, f is the vortex street frequency, and u is
the flow velocity. Sample values from He, Doolen, "Lattice Boltzmann method
on a curvilinear coordinate system: Vortex shedding behind a circular cylinder",
Phys. Rev. E 56/1:
  Re   St
  ----------
  50   0.121
  100  0.161
  150  0.179

Note that St and Re should be calculated based on the actual far-field flow
velocity.

To measure frequency:
 - collect stats from the output once the flow is in a steady state
 - plot(np.fft.fftfreq(stat[:,1].size, d=20)[:100],
        np.abs(np.fft.rfft(stat[:,1]))[:100])
 - find top peak, corresponding x value is frequency

Lift force on the cylinder oscillates at the vortex shedding frequency (f),
while the drag force oscillates at 2f.

Re ranges for flow around a cylinder:
 . 40 - 150:  laminar vortex street
 . < 3e5:     laminar boundary layer, turbulent wake
 . < 3.5e6:   boundary layer transitions to turbulent
 . > 3.5e6:   turbulent vortex street

Note that once turbulence starts, the 3D structure becomes important and the
flow cannot be modeled with the current simulation.
"""

import math
import numpy as np
from sailfish.subdomain import Subdomain2D
from sailfish.node_type import NTFullBBWall, NTEquilibriumDensity, NTCopy, NTRegularizedVelocity
from sailfish.controller import LBSimulationController
from sailfish.lb_single import LBIBMFluidSim, Particle

# Cylinder radius.
R_CYL = 10

class CylinderSubdomain(Subdomain2D):
    def boundary_conditions(self, hx, hy):
        wall_map = (hy == 0) | (hy == self.gy - 1)
        self.set_node(wall_map, NTFullBBWall)
        nwall = np.logical_not(wall_map)
        inflow = nwall & (hx == 0)
        self.set_node(inflow, NTRegularizedVelocity((0.03, 0.0)))
        outflow = nwall & (hx == self.gx - 1)
        self.set_node(outflow, NTCopy)

    def initial_conditions(self, sim, hx, hy):
        sim.rho[:] = 1.0
        sim.vy[:] = 0.0
        sim.vx[:] = 0.0

        # Cylinder position.
        cx = 0.25 * self.config.lat_nx
        cy = 0.5 * self.config.lat_ny
        N = 50     # number of particles

        for i in range(N):
            x = cx + R_CYL * math.cos(i / float(N) * 2.0 * math.pi)
            y = cy + R_CYL * math.sin(i / float(N) * 2.0 * math.pi)
            sim.add_particle(Particle((x, y), stiffness=0.01, ref_position=(x, y)))

class CylinderSimulation(LBIBMFluidSim):
    subdomain = CylinderSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'lat_nx': 512,
            'lat_ny': 128,
            'visc': 0.01,
            'perf_stats_every': 500,
        })

    def __init__(self, config):
        super(CylinderSimulation, self).__init__(config)

        Re = 150
        D = 2 * R_CYL
        max_v = Re / D * config.visc
        force = max_v / D**2 * 8 * config.visc
        self.add_body_force((force, 0.0))

        self.config.logger.info('v_max:%.3e  Re:%d  F:%.3e' % (
            max_v, Re, force))

    def after_step(self, runner):
        every = 20
        mod = self.iteration % every

        if mod == every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            ly = self.config.lat_ny / 2
            lx = int(self.config.lat_nx * 0.75)
            print self.iteration, self.vy[ly, lx], self.vx[ly, lx]

if __name__ == '__main__':
    ctrl = LBSimulationController(CylinderSimulation)
    ctrl.run()

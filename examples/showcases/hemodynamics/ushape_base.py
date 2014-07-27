"""
Flow through a U-shaped pipe.

The pipe dimensions are:
 - diameter   D: 1 in = 2.54e-2 m
 - bow radius R: 3 in
 - arm length L: 10 in

Inflow center is: -0.254, 0.0635, 0
                  -0.254, -0.063461, 0
(original geometry coordinates)

Original dimensions:
 X - inflow direction (longest; ~ L + R)
 Y - flow direction in the bow area
 Z - depth (shortest -- comparable to D)

Geometry bounding box:
 X: -0.25659:0.076243
 Y: -0.0762:0.076239
 Z: -0.0127:0.0127

Simulation dimensions (z, y, x order):
 X - flow direction in the bow area
 Y - inflow direction
 Z - depth

Other physical parameters:
 - U_avg = 1.3111e-2 m/s
 - U_max = 2*U_avg
 - visc = 3.33e-6

Approximate flow length: 2 * L + pi * R_1

The oscillation period (in lattice units) should be significantly longer
than this so that pressure wave propagation effects are not visible.
"""

import numpy as np

from sailfish.lb_base import LBMixIn
from sailfish.vis_mixin import Vis2DSliceMixIn

import common

class UshapeSubdomain(common.InflowOutflowSubdomain):
    inflow_loc = [-0.254, -0.063461, 0]
    inflow_rad = 2.54e-2

    def _inflow_outflow(self, hx, hy, hz, wall_map):
        inlet = None
        outlet = None

        # This needs to cover both the boundary conditions case, where
        # hx, hy, hz can refer to ghost nodes, as well as the initial
        # conditions case where hx, hy, hz will not cover ghost nodes.
        if np.min(hx) <= 0 and np.min(hy) <= 0:
            inlet = np.logical_not(wall_map) & (hy == 0) & (hx < self.gx / 2)

        if np.max(hx) >= self.gx - 1 and np.min(hy) <= 0:
            outlet = np.logical_not(wall_map) & (hy == 0) & (hx > self.gx / 2)

        return inlet, outlet

#    def _velocity_profile(self, hx, hy, hz, wall_map):



class DynamicViscosity(LBMixIn):
    """Dynamically adjusts viscosity while the simulation is running."""
    # iteration number -> new viscosity
    viscosity_map = {}

    def after_step(self, runner):
        if self.iteration not in viscosity_map:
            return

        runner.config.visc = viscosity_map[self.iteration]
        runner._update_compute_code()
        runner._prepare_compute_kernels()
        self._vis_update_kernels(runner)


class UshapeBaseSim(common.HemoSim, Vis2DSliceMixIn):
    subdomain = UshapeSubdomain
    phys_diam = 2.54e-2     # meters
    lb_v = 0.001            # for oscillatory flow

    @classmethod
    def update_defaults(cls, defaults):
        super(UshapeBaseSim, cls).update_defaults(defaults)
        defaults.update({
            'max_iters': 500000,
            'checkpoint_every': 200000,
            'every': 50000,
            'from_': 0,
            'model': 'bgk',

            # Subdomains configuration.
            'subdomains': 1,
            'node_addressing': 'direct',
            'conn_axis': 'y',
            'geometry': 'geo/ushape_802_0.00125.npy.gz',

            # Start the simulation from a lower Reynolds numbers to
            # remove spurious oscillations. During the simulation, the
            # viscosity is going to be dynamically adjusted.
            'reynolds': 100,
            'velocity': 'constant',
        })

    @classmethod
    def modify_config(cls, config):
        if not config.geometry:
            return

        wall_map = cls.load_geometry(config.geometry)

        # Longest dimension determines configuration to use.
        size = wall_map.shape[1]

        if not config.base_name:
            config.base_name = 'results/re%d_ushape_%d_%s' % (
                config.reynolds, size, config.velocity)

        # Smooth out the wall.
        smooth = {
            2002: 1500,
            1669: 1270,
            1002: 768,
            802: 610,
            398: 310
        }

        for i in range(1, smooth[size]):
            wall_map[:,i,:] = wall_map[:,0,:]

        # Make it symmetric.
        hw = wall_map.shape[1] / 2
        wall_map[:,:,-hw:] = wall_map[:,:,:hw][:,:,::-1]

        # Override lattice size based on the geometry file.
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape

        # Add ghost nodes.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map

        super(UshapeBaseSim, cls).modify_config(config)

    prev_rho = None
    prev_v = None
    def after_step(self, runner):
        every = 1000
        mod = self.iteration % every

        return

        # Used for convergence analysis.
        if mod == every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            self.config.logger.info(
                'sums it=%d: %e %e %e %e' % (
                    self.iteration, np.nansum(self.rho), np.nansum(self.vx),
                    np.nansum(self.vy), np.nansum(self.vz)))

            if self.prev_rho is not None:
                nodes = np.sum(np.logical_not(np.isnan(self.rho)))

                drho = np.abs(self.rho - self.prev_rho)
                dv = ((self.vx - self.prev_v[0])**2 +
                      (self.vy - self.prev_v[1])**2 +
                      (self.vz - self.prev_v[2])**2)
                vn = (self.prev_v[0]**2 + self.prev_v[1]**2 + self.prev_v[2]**2)

                rho_rel = np.nansum(drho / self.prev_rho)
                rho_rel2 = np.nansum(drho) / np.nansum(self.prev_rho)

                v_rel = np.sqrt(np.nansum(dv / vn))
                v_rel2 = np.sqrt(np.nansum(dv) / np.nansum(vn))

                self.config.logger.info(
                    'errs it=%d: %e %e %e %e' % (
                        self.iteration, rho_rel / nodes, rho_rel2 / nodes,
                        v_rel / nodes, v_rel2 / nodes))

            self.prev_rho = self.rho.copy()
            self.prev_v = (self.vx.copy(), self.vy.copy(), self.vz.copy())

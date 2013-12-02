# Input location is: -0.254, 0.0635, 0
#  X - flow direction
#  Y - other flow direction
#  Z - short dimension

import numpy as np

from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.vis_mixin import Vis2DSliceMixIn

import common

class UshapeSubdomain(common.InflowOutflowSubdomain):
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


class UshapeSim(common.HemoSim, Vis2DSliceMixIn):
    subdomain = UshapeSubdomain
    phys_diam = 2.54e-2     # meters
    lb_v = 0.001   # for oscillatory

    @classmethod
    def update_defaults(cls, defaults):
        super(UshapeSim, cls).update_defaults(defaults)
        defaults.update({
            'max_iters': 31563 * 10,
            'checkpoint_every': 200000,
            'every': 31563 / 20,
            'from_': 31563 * 5,
            'model': 'bgk',

            'log': 're100_ushape_802_osc.log',
            'checkpoint_file': 're100_ushape_802_osc',
            'output': 're100_ushape_802_osc',

            # Subdomains configuration.
            'subdomains': 1,
            'node_addressing': 'direct',
            'conn_axis': 'y',
            'geometry': 'geo/ushape_802_0.00125.npy.gz',

            'velocity': 'oscillatory',

            # Start the simulation from a lower Reynolds numbers to
            # remove spurious oscillations. During the simulation, the
            # viscosity is going to be dynamically adjusted.
            'reynolds': 100,
        })

    @classmethod
    def modify_config(cls, config):
        if not config.geometry:
            return

        wall_map = cls.load_geometry(config.geometry)
        wall_map = np.rollaxis(wall_map, 2)  # make z the smallest dimension

        # Remove wall blocking the inlet/outlet
        # 2002: 17,  1669: 14,  1002: 9,  802: 7,  402: 4
        wall_map = wall_map[:,7:,:]

        # Smooth out the wall...
        # 2002: 1500,  1669: 1270,  1002: 768,  802: 610,  402: 310
        for i in range(1, 610):
            wall_map[:,i,:] = wall_map[:,0,:]

        # Make it symmetric.
        # 2002: 459,  1669: 394,  1002: 230,  802: 184,  402: 92
        hw = wall_map.shape[2] / 2
        wall_map[:,:,-hw:] = wall_map[:,:,:hw][:,:,::-1]

        # Override lattice size based on the geometry file.
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape

        # Geometry is:
        # - diameter: 1in
        # - bow radius: 3in
        # - arm length: 10in

        # U_avg = 1.3111e-2 m/s, U_max = 2*U_avg
        # visc = 3.33 e-6
        # D_h = 2.54e-2 m
        #
        # Flow length: 1860 [lattice units] (2 * L + pi * R_1 + 0.5) * 60
        # The oscillation period (in lattice units) should be significantly longer
        # than this so that pressure wave propagation effects are not visible.

        # Add ghost nodes.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map

        super(UshapeSim, cls).modify_config(config)

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

        # Dynamically update viscosity to set Re = 1000.
        if self.iteration == 20000:
            runner.config.visc = runner.config.visc / 10.0
            runner._update_compute_code()
            runner._prepare_compute_kernels()
            self._vis_update_kernels(runner)


if __name__ == '__main__':
    LBSimulationController(UshapeSim, EqualSubdomainsGeometry3D).run()

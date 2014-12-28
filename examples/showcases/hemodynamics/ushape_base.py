"""
Flow through a U-shaped pipe.

The pipe dimensions are:
 - diamter    D: 1 in = 2.54e-2 m
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
from sailfish import io

import common

class UshapeSubdomain(common.InflowOutflowSubdomain):
    # Physical coordinates in meters.
    inflow_loc = [-0.254, -0.063461, 0]
    inflow_rad = 2.54e-2 / 2.0

    def _inflow_outflow(self, hx, hy, hz, wall_map):
        inlet = None
        outlet = None

        # This needs to cover both the boundary conditions case, where
        # hx, hy, hz can refer to ghost nodes, as well as the initial
        # conditions case where hx, hy, hz will not cover ghost nodes.
        if np.min(hy) <= 0 and np.min(hx) <= 0:
            inlet = np.logical_not(wall_map) & (hx == 0) & (hy < self.gy / 2)

        if np.max(hy) >= self.gy - 1 and np.min(hx) <= 0:
            outlet = np.logical_not(wall_map) & (hx == 0) & (hy > self.gy / 2)

        return inlet, outlet


class DynamicViscosity(LBMixIn):
    """Dynamically adjusts viscosity while the simulation is running."""
    # iteration number -> new viscosity
    viscosity_map = {}

    def after_step(self, runner):
        if self.iteration not in self.viscosity_map:
            return

        runner.config.visc = self.viscosity_map[self.iteration]
        runner._update_compute_code()
        runner._prepare_compute_kernels()
        self._vis_update_kernels(runner)


class UshapeBaseSim(common.HemoSim, Vis2DSliceMixIn):
    subdomain = UshapeSubdomain
    phys_diam = 2.54e-2     # meters
    lb_v = 0.001            # for oscillatory flow

    smooth = {
        # Update the 2 following values to be compatible with the
        # process_geometry script.
        2002: 1500,
        1669: 1270,
        1241: 968,
        993: 768,
        795: 610,
        894: 697,
        596: 465,
        398: 310,
        266: 204
    }

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
            'geometry': 'geo/proc_ushape_800.npy.gz',

            # Start the simulation from a lower Reynolds numbers to
            # remove spurious oscillations. During the simulation, the
            # viscosity is going to be dynamically adjusted.
            'reynolds': 100,
            'velocity': 'constant',
        })

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--midslice_every', type=int, default=0,
                help='how often to save data from the middle of the '
                'simulation domain')

    @classmethod
    def set_walls(cls, config):
        if not config.geometry:
            return

        wall_map = cls.load_geometry(config.geometry)

        # Longest dimension determines configuration to use.
        size = wall_map.shape[2]

        if not config.base_name:
            config.base_name = 'results/re%d_ushape_%d_%s' % (
                config.reynolds, size, config.velocity)

        # Smooth out the wall.
        for i in range(1, cls.smooth[size]):
            wall_map[:,:,i] = wall_map[:,:,0]

        # Make it symmetric.
        hw = wall_map.shape[1] / 2
        wall_map[:,-hw:,:] = wall_map[:,:hw,:][:,::-1,:]

        # Override lattice size based on the geometry file.
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape

        # Add ghost nodes.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map

    @classmethod
    def get_diam(cls, config):
        _, ys, _ = config._wall_map.shape
        return 2.0 * np.sqrt(np.sum(np.logical_not(config._wall_map[:,:(ys/2),1])) / np.pi)

    @classmethod
    def modify_config(cls, config):
        cls.set_walls(config)
        super(UshapeBaseSim, cls).modify_config(config)

    def _get_midslice(self, runner):
        diam = int(runner._subdomain.inflow_rad / self.config._converter.dx * 2)
        wall_nonghost = self.config._wall_map[1:-1,1:-1,1:-1]
        size = wall_nonghost.shape[1]
        self.midslice = [slice(None)]
        m = size / 2
        if size % 2 == 0:
            # Extracts a 2-node wide slice.
            self.midslice.append(slice(m - 1, m + 1))
        else:
            # Extracts a 3-node wide slice.
            self.midslice.append(slice(m - 1, m + 2))
        self.midslice.append(slice(-(diam + 10), None))
        return wall_nonghost

    midslice = None
    mask = None
    def after_step(self, runner):
        super(UshapeBaseSim, self).after_step(runner)

        # This assumes that the domain is split along the Y axis,
        # and the whole pipe fits in the last subdomain.
        if self.config.midslice_every == 0 or (
                runner._spec.id != self.config.subdomains - 1):
            return

        # Extract macroscopic fields from the middle of the simulation
        # domain and save them to a file.
        mod = self.iteration % self.config.midslice_every
        if mod == self.config.midslice_every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            if self.midslice is None:
                wall_nonghost = self._get_midslice(runner)
                hx, hy, hz = runner._subdomain._get_mgrid()
                hx = hx[self.midslice]
                hy = hy[self.midslice]
                hz = hz[self.midslice]
                self.mask = wall_nonghost[hz, hy, hx]

            ms = self.midslice
            rho = self.rho[ms]
            vx = self.vx[ms]
            vy = self.vy[ms]
            vz = self.vz[ms]
            rho[self.mask] = np.nan
            vx[self.mask] = np.nan
            vy[self.mask] = np.nan
            vz[self.mask] = np.nan

            np.savez(io.filename(self.config.base_name +
                '_midslice', 7, 0, self.iteration),
                rho=rho, vx=vx, vy=vy, vz=vz)

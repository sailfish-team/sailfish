import numpy as np

from sailfish.node_type import DynamicValue
from sailfish.controller import LBSimulationController
from sailfish.geo import EqualSubdomainsGeometry3D
from sailfish.sym import D3Q19, S

import ushape_base
import common

class UshapeZyxSubdomain(ushape_base.UshapeSubdomain):
    # The direction of the flow is x+.
    _flow_orient = D3Q19.vec_to_dir([0, 0, 1])

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

    def velocity_profile(self, hx, hy, hz, wall_map):
        """Returns a velocity profile array."""
        (zm, ym, xm), diam = self._velocity_params(hx, hy, hz, wall_map)
        radius_sq = (diam / 2.0)**2
        r = np.sqrt((hz - 0.5 - zm)**2 + (hy - 0.5 - ym)**2)
        v = self._inflow_velocity(initial=True) * 2.0 * (1.0 - r**2 / radius_sq)
        return v

    def sym_velocity_profile(self, v, xm, ym, zm, diam):
        radius_sq = (diam / 2.0)**2
        return DynamicValue(
            2.0 * v * (1.0 - ((S.gz - zm)**2 + (S.gy - ym)**2) / radius_sq),
            0.0,
            0.0)


class UshapeSim(ushape_base.UshapeBaseSim):
    lb_v = 0.05
    subdomain = UshapeZyxSubdomain

    @classmethod
    def update_defaults(cls, defaults):
        super(UshapeSim, cls).update_defaults(defaults)
        defaults.update({
            'velocity': 'constant',
            'geometry': 'geo/proc_ushape_802_zyx.npy.gz',
            # For benchmarking, disable error checking.
        })

    @classmethod
    def set_walls(cls, config):
        if not config.geometry:
            return

        wall_map = cls.load_geometry(config.geometry)

        # Longest dimension determines configuration to use.
        size = wall_map.shape[2]

        if not config.base_name:
            config.base_name = 'results/re%d_ushape_zyx_%d_%s' % (
                config.reynolds, size, config.velocity)

        # Smooth out the wall.
        smooth = {
            # Update the 3 following values to be compatible with the
            # process_geometry script.
            2002: 1500,
            1669: 1270,
            1241: 968,
            993: 768,
            795: 610,
            398: 310
        }

        for i in range(1, smooth[size]):
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
        super(UshapeSim, cls).modify_config(config)

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

if __name__ == '__main__':
    LBSimulationController(UshapeSim, EqualSubdomainsGeometry3D).run()

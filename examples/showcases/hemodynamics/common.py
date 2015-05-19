from functools import partial
import math
import numpy as np
import gzip

from sailfish.node_type import NTRegularizedVelocity, DynamicValue, NTDoNothing, LinearlyInterpolatedTimeSeries, NTFullBBWall, NTRegularizedDensity, NTEquilibriumDensity, NTEquilibriumVelocity
from sailfish.subdomain import Subdomain3D
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S, D3Q19
from sailfish import util
from sympy import sin, Piecewise
import converter

class InflowOutflowSubdomain(Subdomain3D):
    # Vector pointing in the direction of the flow (x+).
    _flow_orient = D3Q19.vec_to_dir([1, 0, 0])
    _outlet_orient = D3Q19.vec_to_dir([1, 0, 0])
    oscillatory_amplitude = 0.1
    # Number of iterations after which the oscillatory profile
    # starts to become active.
    oscillatory_delay = 100000

    # Number of iterations used for a linear ramp-up of flow velocity
    # in the 'constant_rampup' mode.
    rampup_time = 500000

    bc_velocity = NTEquilibriumVelocity
    bc_outflow = partial(NTEquilibriumDensity, 1.0)

    # Location of the center of the inflow in physical coordinates, using
    # the original axis ordering of the mesh.
    inflow_loc = [0, 0, 0]

    # Radius of the inflow, in physical units.
    inflow_rad = 0

    # Override this to return a boolean array selecting inflow
    # and outflow nodes, or None if the inflow/outflow is not contained in
    # the current subdomain. wall_map is the part of the global wall map
    # corresponding to the current subdomain (with ghosts).
    def _inflow_outflow(self, hx, hy, hz, wall_map):
        raise NotImplementedError

    def sym_velocity_profile(self, v, xm, ym, zm, diam):
        radius_sq = (diam / 2.0)**2

        # Add 0.5 to the grid symbols to indicate that the node is located in the
        # middle of the grid cell.
        # The velocity vector direction matches the flow orientation vector.

        if self._flow_orient == D3Q19.vec_to_dir([1, 0, 0]):
            vv = 2.0 * v * (1.0 - ((S.gz + 0.5 - zm)**2 + (S.gy + 0.5 - ym)**2) / radius_sq)
            return DynamicValue(vv, 0.0, 0.0)
        elif self._flow_orient == D3Q19.vec_to_dir([0, 1, 0]):
            vv = 2.0 * v * (1.0 - ((S.gz + 0.5 - zm)**2 + (S.gx + 0.5 - xm)**2) / radius_sq)
            return DynamicValue(0.0, vv, 0.0)
        else:
            raise ValueError('Unsupported orientation: %d' % self._flow_orient)

    def velocity_profile(self, hx, hy, hz, wall_map):
        """Returns a velocity profile array."""
        (zm, ym, xm), diam = self._velocity_params(hx, hy, hz, wall_map)
        radius_sq = (diam / 2.0)**2

        # Add 0.5 to the grid arrays to indicate that the node is located in the
        # middle of the grid cell.
        if self._flow_orient == D3Q19.vec_to_dir([1, 0, 0]):
            r = np.sqrt((hz + 0.5 - zm)**2 + (hy + 0.5 - ym)**2)
        elif self._flow_orient == D3Q19.vec_to_dir([0, 1, 0]):
            r = np.sqrt((hz + 0.5 - zm)**2 + (hx + 0.5- xm)**2)
        else:
            raise ValueError('Unsupported orientation: %d' % self._flow_orient)
        v = self._inflow_velocity(initial=True) * 2.0 * (1.0 - r**2 / radius_sq)
        return v

    def _inflow_velocity(self, initial=False):
        """Returns an expression describing the mean inflow velocity.

        When used with the parabolic profile, the maximum velocity will be
        twice the returned value. The velocity can be a function of time for
        unsteady flows.

        initial: whether velocity for initial conditions should be returned'
        """
        conv = self.config._converter

        if self.config.velocity == 'constant':
            return conv.velocity_lb
        elif self.config.velocity == 'constant_rampup':
            if initial:
                return 0.0
            return conv.velocity_lb * Piecewise(
                (S.time / self.rampup_time, S.time < self.rampup_time),
                (1.0, True))

        elif self.config.velocity == 'oscillatory':
            if initial:
                return conv.velocity_lb

            return conv.velocity_lb * (
                1 + self.oscillatory_amplitude * Piecewise(
                    (0, S.time < self.oscillatory_delay),
                    (sin(2.0 * np.pi * conv.freq_lb * (S.time -
                                                       self.oscillatory_delay)), True)))
        elif self.config._velocity_profile is not None:
            data = self.config._velocity_profile
            t = data[:, 0]
            v = data[:, 1]
            v_phys_min, v_phys_max = np.min(v), np.max(v)

            if self.config.velocity == 'external_init' or initial:
                return conv.velocity_lb * v[0] / v_phys_max

            v *= 1.0 / v_phys_max * conv.velocity_lb

            v_lb_min, v_lb_max = np.min(v), np.max(v)
            Re_min = v_lb_min * conv.len_lb / conv.visc_lb
            Re_max = v_lb_max * conv.len_lb / conv.visc_lb

            self.config.logger.info('External velocity range: LB: [%.4e, %.4e]'
                                    '  physical: [%.4e, %.4e]  Re: [%d, %d]' %
                                    (np.min(v), np.max(v), v_phys_min,
                                    v_phys_max, Re_min, Re_max))

            phys_step = t[1] - t[0]
            return LinearlyInterpolatedTimeSeries(v, step_size=phys_step /
                                                  self.config._converter.dt)

    # Assumes the global wall map is stored in config._wall_map.
    def _wall_map(self, hx, hy, hz):
        return self.select_subdomain(self.config._wall_map, hx, hy, hz)

    # Only used with node_addressing = 'indirect'.
    def load_active_node_map(self, hx, hy, hz):
        self.set_active_node_map_from_wall_map(self._wall_map(hx, hy, hz))

    def _velocity_params(self, hx, hy, hz, wall_map):
        """Finds the center of the inlet and its diameter."""
        diam = self.inflow_rad / self.config._converter.dx * 2
        loc = self.config._coord_conv.to_lb(self.inflow_loc, round_=False)

        assert diam > 0
        return loc, diam

    def boundary_conditions(self, hx, hy, hz):
        self.config.logger.info(self.config._converter.info_lb)
        wall_map = self._wall_map(hx, hy, hz)
        inlet, outlet = self._inflow_outflow(hx, hy, hz, wall_map)

        self.set_node(wall_map, NTFullBBWall)
        if inlet is not None:
            (zm, ym, xm), diam = self._velocity_params(hx, hy, hz, wall_map)
            self.config.logger.info('.. setting inlet, center at (x=%d, y=%d, z=%d), diam=%f',
                                    xm, ym, zm, diam)
            self.config.logger.info('.. using the "%s" velocity profile',
                                    self.config.velocity)
            self.config.logger.info('.. using the "%s" BC', self.bc_velocity.__name__)
            v = self._inflow_velocity()
            self.set_node(inlet, self.bc_velocity(
                self.sym_velocity_profile(v, xm, ym, zm, diam),
                orientation=self._flow_orient))

        if outlet is not None:
            self._set_outlet(outlet, hx, hy, hz)

    def _set_single_outlet(self, outlet, orient, hx, hy, hz):
        bc = self.bc_outflow(orientation=orient)
        self.config.logger.info('.. setting outlet using the "%s" BC', bc.__class__.__name__)
        self.set_node(outlet, bc)

    def _set_outlet(self, outlet, hx, hy, hz):
        if type(outlet) is np.ndarray:
            self._set_single_outlet(outlet, self._outlet_orient, hx, hy, hz)
        else:
            for out, orient in zip(outlet, self._outlet_orient):
                self._set_single_outlet(out, orient, hx, hy, hz)

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        wall_map = self._wall_map(hx, hy, hz)
        inlet, _ = self._inflow_outflow(hx, hy, hz, wall_map)

        if inlet is not None:
            v = self.velocity_profile(hx, hy, hz, wall_map)

            if self._flow_orient == D3Q19.vec_to_dir([0, 1, 0]):
                sim.vy[inlet] = v[inlet]
            elif self._flow_orient == D3Q19.vec_to_dir([1, 0, 0]):
                sim.vx[inlet] = v[inlet]
            else:
                raise ValueError('Unsupported flow orientation: %d' %
                                 self._flow_orient)

class HemoSim(LBFluidSim):
    phys_visc = 3.33e-6
    phys_diam = None   # Set in a subclass.
    phys_freq = 1.0    # NOT angular frequency.
    lb_v = 0.025

    @classmethod
    def update_defaults(cls, defaults):
        defaults.update({
            'grid': 'D3Q19',
            'perf_stats_every': 1000,
            'block_size': 128,

            # Data layout. Optimize for size.
            'access_pattern': 'AA',
            'node_addressing': 'indirect',
            'compress_intersubdomain_data': True,

            # Output and checkpointing.
            'checkpoint_every': 500000,
            'final_checkpoint': True,
        })

    @classmethod
    def add_options(cls, group, dim):
        group.add_argument('--geometry', type=str,
                           default='',
                           help='file defining the geometry')
        group.add_argument('--velocity', type=str,
                           choices=['constant', 'constant_rampup',
                                    'oscillatory', 'external',
                                    'external_init'],
                           default='constant')
        group.add_argument('--velocity_profile', type=str,
                           default='', help='external velocity profile in a '
                           'two column data format: time [s], velocity [m/s]')
        group.add_argument('--reynolds', type=float,
                           default=10.0, help='Reynolds number; only works '
                           'for --velocity=constant|oscillatory')
        group.add_argument('--errors_every', type=int, default=0,
                           help='Number of iterations between data collection '
                           'used to calculate errors.')
        group.add_argument('--errors_reference', type=str, default='',
                           help='Reference fields against which errors will be '
                           'calculated.')

    @classmethod
    def get_diam(cls, config):
        dx = np.average(np.array(config._coord_conv.dx))
        return cls.subdomain.inflow_rad * 2.0 / dx

    @classmethod
    def set_walls(cls, config):
        if not config.geometry:
            return

        wall_map = util.load_array(config.geometry)

        # Override lattice size based on the geometry file.
        config.lat_nz, config.lat_ny, config.lat_nx = wall_map.shape

        # Add ghost nodes.
        wall_map = np.pad(wall_map, (1, 1), 'constant', constant_values=True)
        config._wall_map = wall_map

    @classmethod
    def get_uconv(cls, config):
        uconv = None
        if config.velocity_profile and config.velocity.startswith('external'):
            config._velocity_profile = np.loadtxt(config.velocity_profile)
            t_cycle = np.max(config._velocity_profile[:,0])
            # Unit conversion based on velocity.
            uconv = converter.UnitConverter(cls.phys_visc,
                                            cls.phys_diam,
                                            velocity=np.max(np.abs(config._velocity_profile[:,1])),
                                            freq=1.0 / t_cycle)
        else:
            # Unit conversion based on Reynolds number.
            uconv = converter.UnitConverter(cls.phys_visc,
                                            cls.phys_diam,
                                            Re=config.reynolds,
                                            freq=cls.phys_freq)
        return uconv

    @classmethod
    def modify_config(cls, config):
        cls.set_walls(config)

        # Instantiate a coordinate converter.
        if config.geometry.endswith('.npy'):
            geo_config_fn = config.geometry.replace('.npy', '.config')
        else:
            geo_config_fn = config.geometry.replace('.npy.gz', '.config')

        import json
        geo_config = json.load(open(geo_config_fn, 'r'))
        config._coord_conv = converter.CoordinateConverter(geo_config)

        # Automatically compute the LB viscosity.
        uconv = cls.get_uconv(config)
        config._converter = uconv
        uconv.set_lb(velocity=cls.lb_v, length=cls.get_diam(config))
        config.visc = uconv.visc_lb

    def __init__(self, config):
        super(HemoSim, self).__init__(config)
        if self.config.errors_reference:
            self.config.logger.info('Loading reference data for error calculation.')
            data = np.load(self.config.errors_reference)
            self.prev_v = data['v']
            self.prev_rho = data['rho']

    prev_rho = None
    prev_v = None
    def track_errors(self, runner):
        mod = self.iteration % self.config.errors_every

        # Used for convergence analysis.
        if mod == self.config.errors_every - 1:
            self.need_sync_flag = True
        elif mod == 0:
            self.config.logger.info(
                'sums it=%d: %e %e %e %e' % (
                    self.iteration, np.nansum(self.rho), np.nansum(self.vx),
                    np.nansum(self.vy), np.nansum(self.vz)))

            nodes = np.sum(np.logical_not(np.isnan(self.rho)))
            if self.prev_rho is not None:
                drho = np.abs(self.rho - self.prev_rho)
                dv = np.sqrt((self.vx - self.prev_v[0])**2 +
                      (self.vy - self.prev_v[1])**2 +
                      (self.vz - self.prev_v[2])**2)
                vn = np.sqrt(self.prev_v[0]**2 + self.prev_v[1]**2 + self.prev_v[2]**2)

                rho_rel = np.nansum(drho / self.prev_rho)
                rho_rel2 = np.nansum(drho) / np.nansum(self.prev_rho)

                v_rel = np.nansum(dv / vn)
                v_rel2 = np.nansum(dv) / np.nansum(vn)

                self.config.logger.info(
                    'errs it=%d: %e %e %e %e' % (
                        self.iteration, rho_rel / nodes, rho_rel2 / nodes,
                        v_rel / nodes, v_rel2 / nodes))
            else:
                self.config.logger.info('total fluid nodes: %d' % nodes)

            if not self.config.errors_reference:
                self.prev_rho = self.rho.copy()
                self.prev_v = (self.vx.copy(), self.vy.copy(), self.vz.copy())

    def after_step(self, runner):
        if self.config.errors_every > 0:
            self.track_errors(runner)

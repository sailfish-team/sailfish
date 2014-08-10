import math
import numpy as np
import gzip

from sailfish.node_type import NTRegularizedVelocity, DynamicValue, NTDoNothing, LinearlyInterpolatedTimeSeries, NTFullBBWall, NTRegularizedDensity, NTEquilibriumDensity
from sailfish.subdomain import Subdomain3D
from sailfish.lb_single import LBFluidSim
from sailfish.sym import S, D3Q19
from sympy import sin


class CoordinateConverter(object):
    """Converts between physical coordinates and LB coordinates"""

    def __init__(self, config):
        """Initializes the converter.

        :param config: dictionary of settings from the .config file
        """
        self.dx = []
        ax = config['axes']
        self.axes = [ax.index('x'), ax.index('y'), ax.index('z')]

        self.padding = []
        self.phys_min_x = []

        for i, size in enumerate(config['size']):
            self.padding.append(config['padding'][2 * i])
            size -= config['padding'][2 * i]
            size -= config['padding'][2 * i + 1]

            # Physical size BEFORE cutting nodes from the envelope.
            phys_size = config['bounding_box'][i]

            if 'cuts' in config:
                # -2 to get rid of padding
                size += config['cuts'][i][0] + config['cuts'][i][1]

            dx = (phys_size[1] - phys_size[0]) / size
            self.dx.append(dx)
            if 'cuts' in config:
                self.phys_min_x.append(phys_size[0] + config['cuts'][i][0] * dx)
            else:
                self.phys_min_x.append(phys_size[0])


    def to_lb(self, phys_pos, round_=True):
        lb_pos = [0, 0, 0]
        for i, phys_x in enumerate(phys_pos):
            lb_pos[self.axes[i]] = (self.padding[i] + (phys_x - self.phys_min_x[i]) / self.dx[i])

        if round_:
            lb_pos = [int(round(x)) for x in lb_pos]
        return lb_pos

    def from_lb(self, lb_pos):
        phys_pos = [0, 0, 0]
        for i, lb_x in enumerate(lb_pos):
            i = self.axes.index(i)
            phys_pos[i] = (self.dx[i] * (lb_x - self.padding[i]) + self.phys_min_x[i])
        return phys_pos


class UnitConverter(object):
    """Performs unit conversions."""

    def __init__(self, visc=None, length=None, velocity=None, Re=None, freq=None):
        """Initializes the converter.

        :param visc: physical viscosity
        :param length: physical reference length
        :param velocity: physical reference velocity
        :param Re: Reynolds number
        :param freq: physical frequency
        """
        self._phys_visc = visc
        self._phys_len = length
        self._phys_vel = velocity
        self._phys_freq = freq

        if Re is not None:
            if visc is None:
                self._phys_visc = length * velocity / Re
            elif length is None:
                self._phys_len = Re * visc / velocity
            elif velocity is None:
                self._phys_vel = Re * visc / length

        self._lb_visc = None
        self._lb_len = None
        self._lb_vel = None

    def set_lb(self, visc=None, length=None, velocity=None):
        if visc is not None:
            self._lb_visc = visc
        if length is not None:
            self._lb_len = length
        if velocity is not None:
            self._lb_vel = velocity

        self._update_missing_parameters()

    def _update_missing_parameters(self):
        if (self._lb_visc is None and self._lb_len is not None and
            self._lb_vel is not None):
            self._lb_visc = self._lb_len * self._lb_vel / self.Re
            return

        if (self._lb_len is None and self._lb_visc is not None and
            self._lb_vel is not None):
            self._lb_len = self.Re * self._lb_visc / self._lb_vel
            return

        if (self._lb_vel is None and self._lb_len is not None and
            self._lb_visc is not None):
            self._lb_vel = self.Re * self._lb_visc / self._lb_len
            return

    @property
    def Re(self):
        return self._phys_len * self._phys_vel / self._phys_visc

    @property
    def Womersley(self):
        return math.sqrt(self._phys_freq * self._phys_len**2 / self.phys_visc)

    @property
    def Re_lb(self):
        return self._lb_len * self._lb_vel / self._lb_visc

    @property
    def Womersley_lb(self):
        return math.sqrt(self.freq_lb * self.len_lb**2 / self.visc_lb)

    @property
    def visc_lb(self):
        return self._lb_visc

    @property
    def velocity_lb(self):
        return self._lb_vel

    @property
    def len_lb(self):
        return self._lb_len

    @property
    def freq_lb(self):
        if self._phys_freq is None:
            return 1.0
        return self._phys_freq * self.dt

    @property
    def dx(self):
        if self._lb_len is None:
            return 0
        return self._phys_len / self._lb_len

    @property
    def dt(self):
        if self._lb_visc is None:
            return 0
        return self._lb_visc / self._phys_visc * self.dx**2

    @property
    def info_lb(self):
        return ('Re:%.2f  Wo:%.2f  visc:%.3e  vel:%.3e  len:%.3e  T:%d  '
                'dx:%.4e  dt:%.4e' % (
                    self.Re_lb, self.Womersley_lb, self.visc_lb, self.velocity_lb,
                    self.len_lb, 1.0 / self.freq_lb, self.dx, self.dt))


class InflowOutflowSubdomain(Subdomain3D):
    # Vector pointing in the direction of the flow.
    _flow_orient = D3Q19.vec_to_dir([0, 1, 0])
    oscillatory_amplitude = 0.1

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
        return None, None

    def _inflow_velocity(self, initial=False):
        """Returns an expression describing the max inflow velocity

        The velocity can be as a function of time for unsteady flows.

        initial: whether velocity for initial conditions should be returned'
        """
        if self.config.velocity == 'constant' or initial:
            return self.config._converter.velocity_lb
        elif self.config.velocity == 'oscillatory':
            return self.config._converter.velocity_lb * (
                1 + self.oscillatory_amplitude * sin(
                    2.0 * np.pi * self.config._converter.freq_lb * S.time))
        elif self.config._velocity_profile is not None:
            data = self.config._velocity_profile
            t = data[:, 0]
            v = data[:, 1]
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

    def _velocity_profile(self, hx, hy, hz, wall_map):
        """Returns a velocity profile array."""
        (zm, ym, xm), loc, diam = self._velocity_params(hx, hy, hz, wall_map)
        r = np.sqrt((hz - 0.5 - zm)**2 + (hx - 0.5 - xm)**2)

        R = diam / 2.0
        v = vel * 2.0 * (1.0 - r**2 / R**2)
        return v

    def boundary_conditions(self, hx, hy, hz):
        self.config.logger.info(self.config._converter.info_lb)
        wall_map = self._wall_map(hx, hy, hz)
        inlet, outlet = self._inflow_outflow(hx, hy, hz, wall_map)

        self.set_node(wall_map, NTFullBBWall)
        if inlet is not None:
            (zm, ym, xm), diam = self._velocity_params(hx, hy, hz, wall_map)
            radius_sq = (diam / 2.0)**2
            self.config.logger.info('.. setting inlet, center at (%d, %d, %d), diam=%f',
                                    xm, ym, zm, diam)
            self.config.logger.info('.. using the "%s" velocity profile',
                                    self.config.velocity)
            v = self._inflow_velocity()
            # Vector pointing into the flow domain. The direction of the flow is y+.
            self.set_node(inlet, NTRegularizedVelocity(
                DynamicValue(0.0,
                             2.0 * v *
                             (1.0 - ((S.gz - zm)**2 + (S.gx - xm)**2) / radius_sq),
                             0.0),
                orientation=self._flow_orient))

        if outlet is not None:
            self.config.logger.info('.. setting outlet')
            self._set_outlet(outlet, hx, hy, hz)

    def _set_outlet(self, outlet, hx, hy, hz):
        self.set_node(outlet, NTEquilibriumDensity(1.0, orientation=self._flow_orient))

    def velocity_profile(self, hx, hy, hz, wall_map, inlet):
        (zm, ym, xm), diam = self._velocity_params(hx, hy, hz, wall_map)
        radius_sq = (diam / 2.0)**2
        r = np.sqrt((hz - 0.5 - zm)**2 + (hx - 0.5 - xm)**2)
        v = self._inflow_velocity(initial=True) * 2.0 * (1.0 - r**2 / radius_sq)
        return v

    def initial_conditions(self, sim, hx, hy, hz):
        sim.rho[:] = 1.0
        wall_map = self._wall_map(hx, hy, hz)
        inlet, outlet = self._inflow_outflow(hx, hy, hz, wall_map)

        if inlet is not None:
            v = self.velocity_profile(hx, hy, hz, wall_map, inlet)
            sim.vy[inlet] = v[inlet]


class HemoSim(LBFluidSim):
    phys_visc = 3.33e-6
    phys_diam = 0.0  # Set in a subclass.
    phys_freq = 1.0  # NOT angular frequency.
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
        LBFluidSim.add_options(group, dim)
        group.add_argument('--geometry', type=str,
                           default='',
                           help='file defining the geometry')
        group.add_argument('--velocity', type=str,
                           choices=['constant', 'oscillatory', 'external'],
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

    @classmethod
    def load_geometry(cls, fname):
        if fname.endswith('.gz'):
            return np.load(gzip.GzipFile(fname))
        else:
            return np.load(fname)

    @classmethod
    def modify_config(cls, config):
        converter = None
        if config.velocity_profile and config.velocity == 'external':
            config._velocity_profile = np.loadtxt(config.velocity_profile)
            # Unit conversion based on velocity.
            converter = UnitConverter(cls.phys_visc, cls.phys_diam,
                                      velocity=np.max(np.abs(config._velocity_profile[:,1])),
                                      freq=cls.phys_freq)
        else:
            # Unit conversion based on Reynolds number.
            converter = UnitConverter(cls.phys_visc, cls.phys_diam,
                                      Re=config.reynolds, freq=cls.phys_freq)
        _, _, xs = config._wall_map.shape
        diam = 2.0 * np.sqrt(np.sum(np.logical_not(config._wall_map[:,1,:(xs/2)])) / np.pi)
        # Automatically compute the LB viscosity.
        converter.set_lb(velocity=cls.lb_v, length=diam)
        config.visc = converter.visc_lb
        config._converter = converter

        # Instantiate a coordinate converter.
        if config.geometry.endswith('.npy'):
            geo_config_fn = config.geometry.replace('.npy', '.config')
        else:
            geo_config_fn = config.geometry.replace('.npy.gz', '.config')

        import json
        geo_config = json.load(open(geo_config_fn, 'r'))
        config._coord_conv = CoordinateConverter(geo_config)

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

            self.prev_rho = self.rho.copy()
            self.prev_v = (self.vx.copy(), self.vy.copy(), self.vz.copy())

    def after_step(self, runner):
        if self.config.errors_every > 0:
            self.track_errors(runner)

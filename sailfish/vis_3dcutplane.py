# TODO(michalj): Port this visualization backend.
raise NotImplementedError("This visualization backend has not been ported yet.")

class Fluid3DVisCutplane(Fluid2DVis):
    name = 'cutplane'
    dims = [3]

    @classmethod
    def add_options(cls, group):
        return False

    def __init__(self, sim):
        Fluid2DVis.__init__(self, sim)
        self.shape =  tuple(reversed(sim.shape))
        self._scr_scale = sim.options.scr_scale
        self._cut_dim = 2
        self._cut_pos = [self.shape[0] / 2, self.shape[1] / 2, self.shape[2] / 2]
        self._reset_display()

    @property
    def _slice_args(self):
        args = []

        for i in range(2, -1, -1):
            if i == self._cut_dim:
                args.append(self._cut_pos[self._cut_dim])
            else:
                args.append(slice(None))
        return args

    def get_field_vals(self, field):
        v = []
        for f in field.vals:
            a = f()[self._slice_args]
            v.append(a)
        return v

    @property
    def vx(self):
        return self.sim.velocity[self._dims[0]][self._slice_args]

    @property
    def vy(self):
        return self.sim.velocity[self._dims[1]][self._slice_args]

    @property
    def geo_map(self):
        return self.sim.geo.map[self._slice_args]

    def _2d_to_3d_loc(self, x, y):
        """Translate a location on the currently visible slice into a location in the 3D lattice.

        Args:
          x, y: coordinates on the current 2D slice

        Returns:
          x, y, z: coordinates in the simulation domain
        """
        loc = [0,0,0]
        loc[self._cut_dim] = self._cut_pos[self._cut_dim]
        loc[self._dims[0]] = x
        loc[self._dims[1]] = y
        return loc

    def _reset_display(self):
        dims = set([0,1,2])
        dims.remove(self._cut_dim)
        dims = sorted(list(dims))

        self._dims = dims
        self.set_mode(int(self.shape[dims[0]] * self._scr_scale),
                int(self.shape[dims[1]] * self._scr_scale))

        # For compatibility with other functions for 2D.
        self.lat_nx = self.shape[dims[0]]
        self.lat_ny = self.shape[dims[1]]

    def _process_misc_event(self, event):
        if event.type == pygame.KEYDOWN:
            # Select the axis normal to the cutplane.
            if event.key == pygame.K_x:
                self._cut_dim = 0
                self._reset_display()
            elif event.key == pygame.K_y:
                self._cut_dim = 1
                self._reset_display()
            elif event.key == pygame.K_z:
                self._cut_dim = 2
                self._reset_display()
            # Move the cutplane along the selected axis.
            elif event.key == pygame.K_QUOTE:
                if self._cut_pos[self._cut_dim] < self.shape[self._cut_dim]-1:
                    self._cut_pos[self._cut_dim] += 1
            elif event.key == pygame.K_SEMICOLON:
                if self._cut_pos[self._cut_dim] > 0:
                    self._cut_pos[self._cut_dim] -= 1

    def _visualize(self):
        ret = Fluid2DVis._visualize(self)
        dim_names = ('X', 'Y', 'Z')
        ret.append('cut {0} @ {1}'.format(dim_names[self._cut_dim], self._cut_pos[self._cut_dim]))
        return ret

    def _draw_tracers(self, tx, ty, sw, sh, width, height):
        pass

    def _draw_wall(self, event):
        x, y, z = self._2d_to_3d_loc(*self._get_loc(event))
        self.sim.geo.set_geo((x, y, z),
                self._draw_type == 1 and geo.LBMGeo.NODE_WALL or geo.LBMGeo.NODE_FLUID,
                update=True)

backend=Fluid3DVisCutplane

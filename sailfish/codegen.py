import os
import sys
from mako.lookup import TemplateLookup

def _convert_to_double(src):
    """Converts all single-precision floating point literals to double
    precision ones.

    :param src: string containing the C code to convert
    """
    import re
    t = re.sub('([0-9]+\.[0-9]*(e-?[0-9]*)?)f([^a-zA-Z0-9\.])', '\\1\\3',
               src.replace('float', 'double'))
    t = t.replace('logf(', 'log(')
    t = t.replace('expf(', 'exp(')
    t = t.replace('powf(', 'pow(')
    return t


class BlockCodeGenerator(object):
    """Generates CUDA/OpenCL code for a simulation."""

    #: The command to use to automatically format the compute unit source code.
    _format_cmd = (
        r"sed -i -e '{{:s;N;\#//#{{p ;d}}; \!#!{{p;d}} ; s/\n//g;t s}}' {path} ; "
        r"sed -i -e 's/}}/}}\n\n/g' {path} ; indent -linux -sob -l120 {path} ; "
        r"sed -i -e '/^$/{{N; s/\n\([\t ]*}}\)$/\1/}}' "
                r"-e '/{{$/{{N; s/{{\n$/{{/}}' {path}")
    # The first sed call removes all newline characters except for those terminating lines
    # that are preprocessor directives (starting with #) or single line comments (//).

    @classmethod
    def add_options(cls, group):
        group.add_argument('--precision',
                help='precision (single, double)', type=str,
                choices=['single', 'double'], default='single')
        group.add_argument('--save_src',
                help='file to save the CUDA/OpenCL source code to',
                type=str, default='')
        group.add_argument('--use_src', type=str, default='',
                help='CUDA/OpenCL source to use instead of the automatically '
                     'generated one')
        group.add_argument('--noformat_src', dest='format_src',
                help='do not format the generated source code',
                action='store_false', default=True)
        group.add_argument('--use_mako_cache',
                help='cache the generated Mako templates in '
                     '/tmp/sailfish_modules-$USER', action='store_true',
                default=False)

    def __init__(self, simulation):
        self._sim = simulation

    def get_code(self, kernel_file, use_mako_cache=False):
        # Clear all locale settings, we do not want them affecting the
        # generated code in any way.
        import locale
        locale.setlocale(locale.LC_ALL, 'C')

        if use_mako_cache:
            import pwd
            lookup = TemplateLookup(directories=sys.path,
                    module_directory='/tmp/sailfish_modules-%s' %
                            pwd.getpwuid(os.getuid())[0])
        else:
            lookup = TemplateLookup(directories=sys.path)

        code_tmpl = lookup.get_template(os.path.join('sailfish/templates',
                                        self._sim.kernel_file))
        ctx = self._build_context()
        src = code_tmpl.render(**ctx)

        if self._is_double_precision():
            src = _convert_to_double(src)

        return src

    def save_code(self, code, dest_path, reformat=True):
        with open(dest_path, 'w') as fsrc:
            print >>fsrc, code

        if reformat:
            os.system(self._format_cmd.format(path=dest_path))

    def _build_context(self):
        ctx = {}
        ctx['dim'] = self.grid.dim
        ctx['block_size'] = self.options.block_size

        # Size of the lattice.
        ctx['lat_ny'] = self.options.lat_ny
        ctx['lat_nx'] = self.options.lat_nx
        ctx['lat_nz'] = self.options.lat_nz

        # Actual size of the array, including any padding.
        ctx['arr_nx'] = self.arr_nx
        ctx['arr_ny'] = self.arr_ny
        ctx['arr_nz'] = self.arr_nz
        ctx['periodic_x'] = int(self.options.periodic_x)
        ctx['periodic_y'] = int(self.options.periodic_y)
        ctx['periodic_z'] = int(self.options.periodic_z)
        ctx['num_params'] = len(self.geo_params)
        ctx['geo_params'] = self.geo_params
        ctx['tau'] = self.tau
        ctx['visc'] = self.float(self.options.visc)
        ctx['backend'] = self.options.backend
        ctx['dist_size'] = self.get_dist_size()
        ctx['pbc_offsets'] = [{-1: self.options.lat_nx,
                                1: -self.options.lat_nx},
                              {-1: self.options.lat_ny*self.arr_nx,
                                1: -self.options.lat_ny*self.arr_nx},
                              {-1: self.options.lat_nz*self.arr_ny*self.arr_nx,
                                1: -self.options.lat_nz*self.arr_ny*self.arr_nx}]
        ctx['bnd_limits'] = [self.options.lat_nx, self.options.lat_ny, self.options.lat_nz]
        ctx['loc_names'] = ['gx', 'gy', 'gz']
        ctx['periodicity'] = [int(self.options.periodic_x), int(self.options.periodic_y),
                            int(self.options.periodic_z)]
        ctx['grid'] = self.grid
        ctx['sim'] = self
        ctx['model'] = self.lbm_model
        ctx['bgk_equilibrium'] = self.equilibrium
        ctx['bgk_equilibrium_vars'] = self.equilibrium_vars
        ctx['constants'] = self.constants
        ctx['grids'] = [self.grid]

        ctx['simtype'] = 'lbm'
        ctx['forces'] = self.forces
        ctx['force_couplings'] = self._force_couplings
        ctx['force_for_eq'] = self._force_term_for_eq
        ctx['image_fields'] = self.image_fields
        ctx['precision'] = self.options.precision

        # TODO: Find a more general way of specifying whether sentinels are
        # necessary.
        ctx['propagation_sentinels'] = (self.options.bc_wall == 'halfbb')

        self._update_ctx(ctx)
        ctx.update(self.geo.get_defines())
        ctx.update(self.backend.get_defines())

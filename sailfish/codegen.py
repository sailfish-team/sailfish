"""Run-time CUDA/OpenCL code generation."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import os
import sys
import tempfile
from mako.lookup import TemplateLookup

import sailfish.io

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
        group.add_argument('--block_size', type=int, default=64,
                help='size of the block of threads on the compute device')

    def __init__(self, simulation):
        self._sim = simulation

    @property
    def config(self):
        return self._sim.config

    def get_code(self, subdomain_runner):
        if self.config.use_src:
            source_fn = sailfish.io.source_filename(self.config.use_src,
                    subdomain_runner._block.id)
            self.config.logger.debug(
                    "Using code from '{0}'.".format(source_fn))
            with open(source_fn, 'r') as f:
                src = f.read()
            return src

        # Clear all locale settings, we do not want them affecting the
        # generated code in any way.
        import locale
        locale.setlocale(locale.LC_ALL, 'C')

        if self.config.use_mako_cache:
            import pwd
            lookup = TemplateLookup(directories=sys.path,
                    module_directory='{0}/sailfish_modules-{1}'.format(
                        tempfile.gettempdir(), pwd.getpwuid(os.getuid())[0]))
        else:
            lookup = TemplateLookup(directories=sys.path)

        code_tmpl = lookup.get_template(os.path.join('sailfish/templates',
                                        self._sim.kernel_file))
        ctx = self._build_context(subdomain_runner)
        src = code_tmpl.render(**ctx)

        if self.is_double_precision():
            src = _convert_to_double(src)

        if self.config.save_src:
            self.save_code(src,
                    sailfish.io.source_filename(self.config.save_src,
                        subdomain_runner._block.id),
                    self.config.format_src)

        return src

    def save_code(self, code, dest_path, reformat=True):
        with open(dest_path, 'w') as fsrc:
            print >>fsrc, code

        if reformat:
            os.system(self._format_cmd.format(path=dest_path))

    def is_double_precision(self):
        return self.config.precision == 'double'

    def _build_context(self, subdomain_runner):
        ctx = {}
        ctx['block_size'] = self.config.block_size
        ctx['propagation_sentinels'] = True

        self._sim.update_context(ctx)
        subdomain_runner.update_context(ctx)

        return ctx

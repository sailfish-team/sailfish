"""Run-time CUDA/OpenCL code generation."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import os
import sys
import tempfile
import mako.exceptions
from mako.lookup import TemplateLookup
from mako.template import Template

from sailfish.lb_base import LBMixIn
import sailfish.io

def _convert_to_double(src):
    """Converts all single-precision floating point literals to double
    precision ones.

    :param src: string containing the C code to convert
    """
    import re
    t = re.sub('([0-9]+\.[0-9]*(e-?[0-9]*)?)f([^a-zA-Z0-9\.])', '\\1\\3',
               src.replace('float', 'double'))
    t = _remove_math_function_suffix(t)
    return t

def _remove_math_function_suffix(t):
    t = t.replace('logf(', 'log(')
    t = t.replace('expf(', 'exp(')
    t = t.replace('powf(', 'pow(')
    t = t.replace('sinf(', 'sin(')
    t = t.replace('cosf(', 'cos(')
    t = t.replace('tanhf(', 'tan(')
    return t

def _use_intrinsics(t):
    t = t.replace('logf(', '__logf(')
    t = t.replace('expf(', '__expf(')
    t = t.replace('powf(', '__powf(')
    t = t.replace('sinf(', '__sinf(')
    t = t.replace('cosf(', '__cosf(')
    t = t.replace('sqrtf(', '__fsqrt_rz(')
    return t

class BlockCodeGenerator(object):
    """Generates CUDA/OpenCL code for a simulation of a subdomain."""

    #: The command to use to automatically format the compute unit source code.
    _format_cmd = (
        r"{sed} -i -e '{{:s;N;\#//#{{p ;d}}; \!#!{{p;d}} ; s/\n//g;t s}}' {path} ; "
        r"{sed} -i -e 's/}}/}}\n\n/g' {path} ; {indent} -linux -sob -l120 {path} ; "
        r"{sed} -i -e '/^$/{{N; s/\n\([\t ]*}}\)$/\1/}}' "
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
        group.add_argument('--mem_alignment', type=int, default=32,
                help='number of bytes to which rows of node should be aligned;'
                ' this affects performance but can also increase memory usage '
                'as the lattice size in the X dimension will be padded up to '
                'the nearest multiple of this value')
        group.add_argument('--indent', type=str, default='indent',
                help='path to the GNU indent program')
        group.add_argument('--sed', type=str, default='sed',
                           help='path to the GNU sed program')
        group.add_argument('--use_intrinsics', action='store_true',
                           default=False, help='whether to use intrinsic '
                           'implementation of transcendental functions')

    def __init__(self, simulation):
        self._sim = simulation

    @property
    def config(self):
        return self._sim.config

    def get_code(self, subdomain_runner, target_type):
        if self.config.use_src:
            source_fn = sailfish.io.source_filename(self.config.use_src,
                    subdomain_runner._spec.id)
            self.config.logger.debug(
                    "Using code from '{0}'.".format(source_fn))
            with open(source_fn, 'r') as f:
                src = f.read()
            return src

        # Clear all locale settings, we do not want them affecting the
        # generated code in any way.
        import locale
        locale.setlocale(locale.LC_ALL, 'C')

        template_dirs = [
                os.getcwd(),
                os.path.join(os.getcwd(), os.path.dirname(sys.argv[0])),
                os.path.join(
                    os.path.realpath(os.path.dirname(__file__)),
                    'templates')]

        if self.config.use_mako_cache:
            import pwd
            lookup = TemplateLookup(directories=template_dirs,
                        module_directory='{0}/sailfish_modules-{1}'.format(
                        tempfile.gettempdir(), pwd.getpwuid(os.getuid())[0]))
        else:
            lookup = TemplateLookup(directories=template_dirs)

        code_tmpl = lookup.get_template(self._sim.kernel_file)
        ctx = self._build_context(subdomain_runner)
        try:
            src = code_tmpl.render(**ctx)
        except:
            print mako.exceptions.text_error_template().render()
            return ''

        aux_sources = list(self._sim.aux_code)
        # Allow mixin classes to provide their own aux_code values.
        for c in self._sim.__class__.mro()[1:]:
            if issubclass(c, LBMixIn) and hasattr(c, 'aux_code'):
                for fn in c.aux_code:
                    # Do not allow duplicate files.
                    if fn not in aux_sources:
                        aux_sources.append(fn)

        for aux in aux_sources:
            if aux.count('\n') > 0:
                code_tmpl = Template(aux)
            else:
                code_tmpl = lookup.get_template(aux)
            try:
                src += '\n' + code_tmpl.render(**ctx)
            except:
                print mako.exceptions.text_error_template().render()
                return ''

        if self.is_double_precision():
            src = _convert_to_double(src)

        if self.config.use_intrinsics:
            src = _use_intrinsics(src)

        # TODO(michalj): Consider using native_ or half_ functions here.
        if target_type == 'opencl':
            src = _remove_math_function_suffix(src)

        if self.config.save_src:
            self.save_code(src,
                    sailfish.io.source_filename(self.config.save_src,
                        subdomain_runner._spec.id),
                    self.config.format_src)

        return src

    def save_code(self, code, dest_path, reformat=True):
        with open(dest_path, 'w') as fsrc:
            print >>fsrc, code

        if reformat:
            os.system(self._format_cmd.format(
                path=dest_path,
                indent=self.config.indent,
                sed=self.config.sed))

    def is_double_precision(self):
        return self.config.precision == 'double'

    def _build_context(self, subdomain_runner):
        ctx = {}
        ctx['block_size'] = self.config.block_size
        ctx['propagation_sentinels'] = True
        ctx['unit_test'] = self.config.unit_test

        self._sim.update_context(ctx)
        subdomain_runner.update_context(ctx)

        return ctx

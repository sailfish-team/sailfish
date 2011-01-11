"""A dummy Sailfish backend.  Used for testing."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

class DummyBackend(object):

    @classmethod
    def add_options(cls, group):
        return 0

    def __init__(self, options=None):
        self.buffers = {}

    def alloc_buf(self, size=None, like=None):
        return 0

    def to_buf(self, cl_buf, source=None):
        pass

    def from_buf(self, cl_buf, target=None):
        pass

    def build(self, source):
        pass

    def get_kernel(self, prog, name, block, args, args_format, shared=None, fields=[]):
        return None

    def run_kernel(self, kernel, grid_size):
        return None

    def sync(self):
        pass

    def get_defines(self):
        return {
            'shared_var': '__shared__',
            'kernel': '__global__',
            'global_ptr': '',
            'const_ptr': '',
            'device_func': '__device__',
            'const_var': '__constant__',
        }


backend=DummyBackend

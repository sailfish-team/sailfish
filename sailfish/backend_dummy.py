"""A dummy Sailfish backend.  Used for testing."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import numpy as np

class DummyBackend(object):

    @classmethod
    def add_options(cls, group):
        return 0

    def __init__(self, options=None):
        self.buffers = {}
        self.arrays = {}

    def alloc_buf(self, size=None, like=None, wrap_in_array=True):
        return like

    def alloc_async_host_buf(self, shape, dtype):
        return np.zeros(shape, dtype=dtype)

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

    def get_reduction_kernel(self, reduce_expr, map_expr, neutral, *args):
        return lambda : None

    def sync(self):
        pass

    def to_buf_async(self, *args):
        pass

    def from_buf_async(self, *args):
        pass

    def get_defines(self):
        return {
            'warp_size': 32,
            'supports_shuffle': False,
            'shared_var': '__shared__',
            'kernel': '__global__',
            'global_ptr': '',
            'const_ptr': '',
            'device_func': '__device__',
            'const_var': '__constant__',
        }

    def make_stream(self):
        return DummyStream()

    def make_event(self, stream, timing=False):
        return DummyEvent()

    def sync_stream(self, *streams):
        pass

class DummyStream(object):
    def synchronize(self):
        pass

class DummyEvent(object):
    pass

backend=DummyBackend

"""Auxilliary classes for on-line data visualization."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import namedtuple
from sailfish.lb_base import ScalarField, LBMixIn
from sailfish.util import ArrayPair
import numpy as np

import netifaces
import zmq

class Slice(object):
    def __init__(self, pair, kernel):
        self.pair = pair
        self.kernel = kernel

class Vis2DSliceMixIn(LBMixIn):
    """Extracts 2D slices of 3D fields for on-line visualization."""
    aux_code = ['data_processing.mako']

    def before_main_loop(self, runner):
        self._vis_every = 10
        self._axis = 0
        self._position = 0

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.PAIR)
        port = self._sock.bind_to_random_port('tcp://*')

        for iface in netifaces.interfaces():
            addr = netifaces.ifaddresses(iface).get(
                netifaces.AF_INET, [{}])[0].get('addr', '')
            if addr:
                self.config.logger.info('Visualization server at %s:%d', addr,
                                        port)

        gpu_map = runner.gpu_geo_map()
        gpu_v = runner.gpu_field(self.v)

        self._buf_sizes = (runner._spec.ny * runner._spec.nz,
                           runner._spec.nx * runner._spec.nz,
                           runner._spec.nx * runner._spec.ny)
        self._buf_shapes = ((runner._spec.nz, runner._spec.ny),
                            (runner._spec.nz, runner._spec.nx),
                            (runner._spec.ny, runner._spec.ny))

        # The buffer has to be large enough to hold any slice.
        buffer_size = max(self._buf_sizes)
        def _make_buf(size):
            h = np.zeros(size, dtype=runner.float)
            return ArrayPair(h, runner.backend.alloc_buf(like=h))

        self._slices = []

        targets = [self.vx, self.vy, self.vz]
        targets.extend(self._scalar_fields)
        self._names = ['vx', 'vy', 'vz']

        gpu_targets = gpu_v
        for f in self._scalar_fields:
            gpu_targets.append(runner.gpu_field(f.buffer))
            self._names.append(f.abstract.name)

        for gf in gpu_targets:
            pair = _make_buf(buffer_size)
            self._slices.append(
                Slice(pair, runner.get_kernel(
                    'ExtractSliceField',
                    [self._axis, self._position, gpu_map, gf,
                     pair.gpu], 'iiPPP')))

        self._vis_targets = targets

    def after_step(self, runner):
        mod = self.iteration % self._vis_every
        if mod == self._vis_every - 1:
            self.need_fields_flags = True
        elif mod == 0:
            md = {
                'names': self._names,
                'dtype': str(runner.float),
                'shape': self._buf_shapes[self._axis]
            }

            try:
                self._sock.send_json(md, zmq.SNDMORE | zmq.NOBLOCK)
            except zmq.ZMQError:
                # This most likely indicates that the socket is not connected.
                # Bail out early to avoid running kernels to extract slice data.
                return

            grid = (self._buf_sizes[self._axis] + self.config.block_size - 1) / self.config.block_size
            for sl in self._slices:
                runner.backend.run_kernel(sl.kernel, grid)

            selector = slice(0, self._buf_sizes[self._axis])
            for i, sl in enumerate(self._slices):
                runner.backend.from_buf(sl.pair.gpu)
                try:
                    self._sock.send(sl.pair.host[selector], zmq.NOBLOCK |
                                    (zmq.SNDMORE if i != len(self._slices) - 1
                                     else 0))
                except zmq.ZMQError:
                    self.config.logger.error(
                        'Failed to send visualization info for "%s".',
                        self._vis_targets[i].name)
                    return


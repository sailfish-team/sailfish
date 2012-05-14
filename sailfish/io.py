"""Input/Output for LB simulations."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

import math
import numpy as np
import operator
import os
import ctypes
from ctypes import Structure, c_uint16, c_int32, c_uint8, c_bool

class VisConfig(Structure):
    MAX_NAME_SIZE = 64
    _fields_ = [('iteration', c_int32), ('block', c_uint16), ('field', c_uint8),
            ('all_blocks', c_bool), ('fields', c_uint8), ('field_name',
                type(ctypes.create_string_buffer(MAX_NAME_SIZE)))]

class LBOutput(object):
    def __init__(self, config, subdomain_id, *args, **kwargs):
        self._scalar_fields = {}
        self._vector_fields = {}

        # Additional scalar fields used for visualization.
        self._visualization_fields = {}
        self.basename = config.output
        self.subdomain_id = subdomain_id

    def register_field(self, field, name, visualization=False):
        if visualization:
            self._visualization_fields[name] = field
        else:
            if type(field) is list:
                self._vector_fields[name] = field
            else:
                self._scalar_fields[name] = field

    def save(self, i):
        pass

    def dump_dists(self, i):
        pass


class VisualizationWrapper(LBOutput):
    """Passes data to a visualization engine, and handles saving it to a
    file."""
    format_name = 'vis'

    # TODO(michalj): Add support for visualization fields different from these
    # used for the output file.
    def __init__(self, config, block, vis_config, output_cls):
        self._output = output_cls(config, block.id)
        self._vis_buffer = block.vis_buffer
        self._vis_config = vis_config
        self._geo_buffer = block.vis_geo_buffer
        self._first_save = True
        self.block = block
        self.nodes = reduce(operator.mul, block.size)
        self._dim = len(self.block.size)

    def register_field(self, field, name, visualization=False):
        self._output.register_field(field, name, visualization)

    def save(self, i):
        self._output.save(i)

        if self._first_save:
            self._scalar_names = self._output._scalar_fields.keys()
            self._vis_names    = self._output._visualization_fields.keys()
            self._vector_names = self._output._vector_fields.keys()
            self._scalar_len = len(self._scalar_names)
            self._vis_len    = len(self._vis_names)
            self._vector_len = len(self._vector_names) * self._dim
            self._field_names = self._scalar_names + self._vis_names
            component_names = ['_x', '_y', '_z']
            for name in self._vector_names:
                for i in range(self._dim):
                    self._field_names.append(
                        '{0}{1}'.format(name, component_names[i]))

            self._vis_config.fields = self._scalar_len + self._vis_len + self._vector_len
            self._first_save = False

        # Only update the buffer if the block to which we belong is
        # currently being visualized.
        if self._vis_config.all_blocks or self.block.id == self._vis_config.block:
            self._vis_config.iteration = i
            requested_field = self._vis_config.field
            self._vis_config.field_name = self._field_names[requested_field]

            if requested_field < self._scalar_len:
                name = self._scalar_names[requested_field]
                field = self._output._scalar_fields[name]
            elif requested_field < self._scalar_len + self._vis_len:
                requested_field -= self._scalar_len
                name = self._vis_names[requested_field]
                field = self._output._visualization_fields[name]()
            else:
                requested_field -= self._scalar_len + self._vis_len
                idx = requested_field / self._dim
                name = self._vector_names[idx]
                component = requested_field % self._dim
                field = self._output._vector_fields[name][component]

            self._vis_buffer[0:self.nodes] = np.ravel(field)
            self._geo_buffer[0:self.nodes] = np.ravel(self.block.runner.visualization_map())


def filename_iter_digits(max_iters=0):
    """Returns the number of digits used to represent the iteration in the filename"""
    if max_iters:
        return str(int(math.log10(max_iters)) + 1)
    else:
        return str(7)

def filename(base, digits, subdomain_id, it, suffix='.npz'):
    return ('{0}.{1}.{2:0' + str(digits) + 'd}{3}').format(base, subdomain_id,
            it, suffix)

def merged_filename(base, digits, it, suffix='.npz'):
    return ('{0}.{1:0' + str(digits) + 'd}{2}').format(base, it, suffix)

def dists_filename(base, digits, subdomain_id, it, suffix='.npy'):
    return filename(base + '_dists', digits, subdomain_id, it, suffix=suffix)

def subdomains_filename(base):
    return base + '.subdomains'

def source_filename(filename, subdomain_id):
    base, ext = os.path.splitext(filename)
    return '{0}.{1}.{2}'.format(base, subdomain_id, ext)


class VTKOutput(LBOutput):
    """Saves simulation data in VTK files."""
    format_name = 'vtk'

    def __init__(self, config):
        LBOutput.__init__(self)
        self.fname = config.output
        self.digits = filename_iter_digits(config.max_iters)

    def save(self, i):
        # FIXME: Port this class.
        raise NotImplementedError('This class has not been ported yet.')

        from enthought.tvtk.api import tvtk
        idata = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))

        fields = self._scalar_fields.keys() + self._vector_fields.keys()
        ffld = fields[0]
        fields = fields[1:]

        idata.point_data.scalars = self.sim.output_fields[ffld].flatten()
        idata.point_data.scalars.name = ffld

        for fld in fields:
            tmp = idata.point_data.add_array(self.sim.output_fields[fld].flatten())
            idata.point_data.get_array(tmp).name = fld

        idata.update()

        for k, v in self.sim.output_vectors.iteritems():
            if self.sim.gridata.dim == 3:
                tmp = idata.point_data.add_array(np.c_[v[0].flatten(),
                                                 v[1].flatten(), v[2].flatten()])
            else:
                tmp = idata.point_data.add_array(np.c_[v[0].flatten(),
                                                 v[1].flatten(), np.zeros_like(v[0].flatten())])
            idata.point_data.get_array(tmp).name = k

        if self.sim.grid.dim == 3:
            idata.dimensions = list(reversed(self.sim.output_fields[ffld].shape))
        else:
            idata.dimensions = list(reversed(self.sim.output_fields[ffld].shape)) + [1]
        w = tvtk.XMLPImageDataWriter(input=idata,
                                     file_name=('%s%0' + self.digits + 'd.xml') % (self.fname, i))
        w.write()


class NPYOutput(LBOutput):
    """Saves simulation data as np arrays."""
    format_name = 'npy'

    def __init__(self, config, subdomain_id):
        LBOutput.__init__(self, config, subdomain_id)
        self.digits = filename_iter_digits(config.max_iters)

    def save(self, i):
        fname = filename(self.basename, self.digits, self.subdomain_id, i, suffix='')
        data = {}
        data.update(self._scalar_fields)
        data.update(self._vector_fields)
        np.savez(fname, **data)

    def dump_dists(self, dists, i):
        fname = dists_filename(self.basename, self.digits, self.subdomain_id, i)
        np.save(fname, dists)


class MatlabOutput(LBOutput):
    """Saves simulation data as Matlab .mat files."""
    format_name = 'mat'

    def __init__(self, config, subdomain_id):
        LBOutput.__init__(self, config, subdomain_id)
        self.digits = filename_iter_digits(config.max_iters)

    def save(self, i):
        import scipy.io
        fname = filename(self.basename, self.digits, self.subdomain_id, i, suffix='')
        data = {}
        data.update(self._scalar_fields)
        data.update(self._vector_fields)
        scipy.io.savemat(fname, data)

    def dump_dists(self, dists, i):
        import scipy.io
        fname = dists_filename(self.basename, self.digits, self.subdomain_id, i)
        scipy.io.savemat(dists)


_OUTPUTS = [NPYOutput, VTKOutput, MatlabOutput]

format_name_to_cls = {}
for output_class in _OUTPUTS:
    format_name_to_cls[output_class.format_name] = output_class

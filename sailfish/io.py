import math
import numpy

# TODO: Correctly process vector and scalar fields in these clases.
class HDF5FlatOutput(object):
    """Saves simulation data in a HDF5 file."""
    format_name = 'h5flat'

    def __init__(self, fname, sim):
        self.sim = sim
        import tables
        self.h5file = tables.openFile(fname, mode='w')
        self.h5grp = self.h5file.createGroup('/', 'results', 'simulation results')
        self.h5file.setNodeAttr(self.h5grp, 'viscosity', sim.options.visc)
        self.h5file.setNodeAttr(self.h5grp, 'sample_rate', sim.options.every)
        self.h5file.setNodeAttr(self.h5grp, 'model', sim.lbm_model)

    def save(self, i):
        h5t = self.h5file.createGroup(self.h5grp, 'iter%d' % i, 'iteration %d' % i)
        self.h5file.createArray(h5t, 'v', numpy.dstack(self.sim.velocity), 'velocity')
        self.h5file.createArray(h5t, 'rho', self.sim.rho, 'density')


class HDF5NestedOutput(HDF5FlatOutput):
    """Saves simulation data in a HDF5 file."""
    format_name = 'h5nested'

    def __init__(self, fname, sim):
        super(HDF5NestedOutput, self).__init__(fname, sim)
        import tables
        desc = {
            'iter': tables.Float32Col(pos=0),
            'vx': tables.Float32Col(pos=1, shape=sim.vx.shape),
            'vy': tables.Float32Col(pos=2, shape=sim.vy.shape),
            'rho': tables.Float32Col(pos=4, shape=sim.rho.shape)
        }

        if sim.grid.dim == 3:
            desc['vz'] = tables.Float32Col(pos=2, shape=sim.vz.shape)

        self.h5tbl = self.h5file.createTable(self.h5grp, 'results', desc, 'results')

    def save(self, i):
        record = self.h5tbl.row
        record['iter'] = i
        record['vx'] = self.sim.vx
        record['vy'] = self.sim.vy
        if self.sim.grid.dim == 3:
            record['vz'] = self.sim.vz
        record['rho'] = self.sim.rho
        record.append()
        self.h5tbl.flush()


def _get_fname_digits(max_iters=0):
    if max_iters:
        return str(int(math.log10(max_iters)) + 1)
    else:
        return str(7)


class VTKOutput(object):
    """Saves simulation data in VTK files."""
    format_name = 'vtk'

    def __init__(self, fname, sim):
        self.fname = fname
        self.sim = sim
        self.digits = _get_fname_digits(sim.options.max_iters)

    def save(self, i):
        from enthought.tvtk.api import tvtk
        idata = tvtk.ImageData(spacing=(1, 1, 1), origin=(0, 0, 0))

        # FIXME: Gracefully handle the case when there are no scalar fields.
        fields = self.sim.output_fields.keys()
        ffld = fields[0]
        fields = fields[1:]

        idata.point_data.scalars = self.sim.output_fields[ffld].flatten()
        idata.point_data.scalars.name = ffld

        for fld in fields:
            tmp = idata.point_data.add_array(self.sim.output_fields[fld].flatten())
            idata.point_data.get_array(tmp).name = fld

        idata.update()

        for k, v in self.sim.output_vectors.iteritems():
            if self.sim.grid.dim == 3:
                tmp = idata.point_data.add_array(numpy.c_[v[0].flatten(),
                                                 v[1].flatten(), v[2].flatten()])
            else:
                tmp = idata.point_data.add_array(numpy.c_[v[0].flatten(),
                                                 v[1].flatten(), numpy.zeros_like(v[0].flatten())])
            idata.point_data.get_array(tmp).name = k

        if self.sim.grid.dim == 3:
            idata.dimensions = list(reversed(self.sim.output_fields[ffld].shape))
        else:
            idata.dimensions = list(reversed(self.sim.output_fields[ffld].shape)) + [1]
        w = tvtk.XMLPImageDataWriter(input=idata,
                                     file_name=('%s%0' + self.digits + 'd.xml') % (self.fname, i))
        w.write()


class NPYOutput(object):
    """Saves simulation data as numpy arrays."""
    format_name = 'npy'

    def __init__(self, fname, sim):
        self.fname = fname
        self.sim = sim
        self.digits = _get_fname_digits(sim.options.max_iters)

    def save(self, i):
        fname = ('%s%0' + self.digits + 'd') % (self.fname, i)
        data = {}
        data.update(self.sim.output_fields)
        data.update(self.sim.output_vectors)
        numpy.savez(fname, **data)


class MatlabOutput(object):
    """Ssves simulation data as Matlab .mat files."""
    format_name = 'mat'

    def __init__(self, fname, sim):
        self.fname = fname
        self.sim = sim
        self.digits = _get_fname_digits(sim.options.max_iters)

    def save(self, i):
        import scipy.io
        fname = ('%s%0' + self.digits + 'd.mat') % (self.fname, i)
        data = {}
        data.update(self.sim.output_fields)
        data.update(self.sim.output_vectors)
        scipy.io.savemat(fname, data)

OUTPUTS = [NPYOutput, HDF5FlatOutput, HDF5NestedOutput, VTKOutput, MatlabOutput]

format_name_to_cls = {}
for output_class in OUTPUTS:
    format_name_to_cls[output_class.format_name] = output_class

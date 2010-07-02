class Field(object):
    def __init__(self, name, values, negative=False, ranges=None):
        self.name = name
        if type(values) is tuple or type(values) is list:
            self.vals = values
        else:
            self.vals = (values,)

        self.negative = negative
        self.ranges = ranges

        if ranges is not None and len(ranges) != len(values):
            raise ValueError('A range has to be specified for every component of the field.')

class FluidVis(object):

    name = 'replace_this_in_derivative_classes'
    dim = []

    @classmethod
    def add_options(cls, group):
        pass

    def __init__(self, *args, **kwargs):
        self.display_infos = []
        self.vis_fields = []

    def add_info(self, info):
        self.display_infos.append(info)

    def add_field(self, field, description, negative=False):
        if type(field) is list or type(field) is tuple:
            self.vis_fields.append(Field(description, field, negative))
        else:
            if callable(field):
                self.vis_fields.append(Field(description, field, negative))
            else:
                self.vis_fields.append(Field(description, lambda: field, negative))

    @property
    def num_fields(self):
        return len(self.vis_fields)



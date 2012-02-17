"""Geometry encoding logic."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'GPL3'

from collections import defaultdict
import numpy as np


def bit_len(num):
    """Returns the minimal number of bits necesary to encode `num`."""
    length = 0
    while num:
        num >>= 1
        length += 1
    return max(length, 1)


class GeoEncoder(object):
    """Takes information about geometry as specified by the simulation and
    encodes it into buffers suitable for processing on a GPU.

    This is an abstract class.  Its implementations provide a specific encoding
    scheme."""
    def __init__(self, subdomain):
        self._type_id_map = {}
        self.subdomain = subdomain

    def encode(self):
        raise NotImplementedError("encode() should be implemented in a subclass")

    def update_context(self, ctx):
        raise NotImplementedError("update_context() should be implemented in a subclass")

    def _type_id(self, node_type):
        if node_type in self._type_id_map:
            return self._type_id_map[node_type]
        else:
            # Does not end with 0xff to make sure the compiler will not complain
            # that x < <val> always evaluates true.
            return 0xfffffffe


class GeoEncoderConst(GeoEncoder):
    """Encodes information about the type, optional parameters and orientation
    of a node into a single uint32.  Optional parameters such as velocities,
    densities, etc. are stored in const memory."""

    def __init__(self, subdomain):
        GeoEncoder.__init__(self, subdomain)

        self._bits_type = 0
        self._bits_param = 0
        self._type_map = None
        self._param_map = None
        self._geo_params = []
        # TODO: Generalize this.
        self._num_velocities = 0
        self.config = subdomain.block.runner.config

    def prepare_encode(self, type_map, param_map, param_dict):
        """
        Args:
          type_map: uint32 array representing node type information
        """
        uniq_types = np.unique(type_map)

        for i, node_type in enumerate(uniq_types):
            self._type_id_map[node_type] = i

        self._bits_type = bit_len(uniq_types.size)
        self._type_map = type_map
        self._param_map = param_map

        # Group parameters by type.
        type_dict = defaultdict(list)
        for param_hash, (node_type, val) in param_dict.iteritems():
            type_dict[node_type].append((param_hash, val))

        max_len = 0
        for node_type, values in type_dict.iteritems():
            l = len(values)
            if node_type == self.subdomain.NODE_VELOCITY:
                self._num_velocities = l
            max_len = max(max_len, l)
        self._bits_param = bit_len(max_len)

        # TODO(michalj): Generalize this to other node types.
        for param_hash, val in type_dict[self.subdomain.NODE_VELOCITY]:
            self._geo_params.extend(val)
        for param_hash, val in type_dict[self.subdomain.NODE_PRESSURE]:
            self._geo_params.append(val)

        self._type_dict = type_dict

    def encode(self):
        assert self._type_map is not None

        # TODO: optimize this using numpy's built-in routines
        param = np.zeros_like(self._type_map)
        for node_type, values in self._type_dict.iteritems():
            for i, (hash_value, _) in enumerate(values):
                param[self._param_map == hash_value] = i

        orientation = np.zeros_like(self._type_map)
        cnt = np.zeros_like(self._type_map)

        for i, vec in enumerate(self.subdomain.grid.basis):
            l = len(list(vec)) - 1
            shifted_map = self._type_map
            for j, shift in enumerate(vec):
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            cnt[(shifted_map == self.subdomain.NODE_WALL)] += 1
            # FIXME: we're currently only processing the primary directions
            # here
            if vec.dot(vec) == 1:
                idx = np.logical_and(self._type_map != self.subdomain.NODE_FLUID,
                        shifted_map == self.subdomain.NODE_FLUID)
                orientation[idx] = self.subdomain.grid.vec_to_dir(list(vec))

            # Mark any nodes completely surrounded by walls as unused.
            self._type_map[(cnt == self.subdomain.grid.Q)] = self.subdomain.NODE_UNUSED

        # Remap type IDs.
        max_type_code = max(self._type_id_map.keys())
        type_choice_map = np.zeros(max_type_code+1, dtype=np.uint32)
        for orig_code, new_code in self._type_id_map.iteritems():
            type_choice_map[orig_code] = new_code

        self._type_map[:] = self._encode_node(orientation, param,
                np.choose(np.int32(self._type_map), type_choice_map))

        # Drop the reference to the map array.
        self._type_map = None

    def update_context(self, ctx):
        ctx.update({
            'geo_fluid': self._type_id(self.subdomain.NODE_FLUID),
            'geo_wall': self._type_id(self.subdomain.NODE_WALL),
            'geo_slip': self._type_id(self.subdomain.NODE_SLIP),
            'geo_unused': self._type_id(self.subdomain.NODE_UNUSED),
            'geo_velocity': self._type_id(self.subdomain.NODE_VELOCITY),
            'geo_pressure': self._type_id(self.subdomain.NODE_PRESSURE),
            'geo_boundary': self._type_id(self.subdomain.NODE_BOUNDARY),
            'geo_ghost': self._type_id(self.subdomain.NODE_GHOST),
            'geo_misc_shift': self._bits_type,
            'geo_type_mask': (1 << self._bits_type) - 1,
            'geo_param_shift': self._bits_param,
            'geo_obj_shift': 0,
            'geo_dir_other': 0,
            'geo_num_velocities': self._num_velocities,
            'geo_params': self._geo_params
        })

    def _encode_node(self, orientation, param, node_type):
        """Encodes information for a single node into a uint32.

        The node code consists of the following bit fields:
          orientation | param_index | node_type
        """
        misc_data = (orientation << self._bits_param) | param
        return node_type | (misc_data << self._bits_type)


# TODO: Implement this class.
class GeoEncoderBuffer(GeoEncoder):
    pass

# TODO: Implement this class.
class GeoEncoderMap(GeoEncoder):
    pass

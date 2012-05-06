"""Geometry encoding logic."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

from collections import defaultdict
import numpy as np

from sailfish import util
import sailfish.node_type as nt

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
        # Maps LBNodeType.id to an internal ID used for encoding purposes.
        self._type_id_remap = {0: 0}  # fluid nodes are not remapped
        self.subdomain = subdomain

    def encode(self):
        raise NotImplementedError("encode() should be implemented in a subclass")

    def update_context(self, ctx):
        raise NotImplementedError("update_context() should be implemented in a subclass")

    def _type_id(self, node_type):
        if node_type in self._type_id_remap:
            return self._type_id_remap[node_type]
        else:
            # Does not end with 0xff to make sure the compiler will not complain
            # that x < <val> always evaluates true.
            return 0xfffffffe


class GeoEncoderConst(GeoEncoder):
    """Encodes node type and parameters into a single uint32.

    Optional parameters such as velocities, densities, etc. are stored in
    const memory and the packed value in the uint32 only contains an index
    inside a const memory array."""

    def __init__(self, subdomain):
        GeoEncoder.__init__(self, subdomain)

        # Set of all used node types, passed down to the Mako engine.
        self._node_types = set([nt._NTFluid])
        self._bits_type = 0
        self._bits_param = 0
        self._type_map = None
        self._param_map = None
        self._geo_params = []
        self.config = subdomain.block.runner.config

    def prepare_encode(self, type_map, param_map, param_dict):
        """
        Args:
          type_map: uint32 array of NodeType.ids
          param_map: array whose entries are keys in param_dict
          param_dict: maps entries from param_map to LBNodeType objects
        """
        uniq_types = list(np.unique(type_map))
        for nt_id in uniq_types:
            self._node_types.add(nt._NODE_TYPES[nt_id])

        # Initialize the node ID map used for remapping.
        for i, node_type in enumerate(uniq_types):
            self._type_id_remap[node_type] = i + 1

        self._bits_type = bit_len(len(uniq_types))
        self._type_map = type_map
        self._param_map = param_map
        self._param_dict = param_dict
        self._encoded_param_map = np.zeros_like(self._type_map)

        param_to_idx = dict()  # Maps entries in seen_params to ids.
        seen_params = set()
        param_items = 0

        # Refer to geo_block.Subdomain._verify_params for a list of allowed
        # ways of encoding nodes.
        for node_key, node_type in param_dict.iteritems():
            for param in node_type.params.itervalues():
                if util.is_number(param):
                    if param in seen_params:
                        idx = param_to_idx[param]
                    else:
                        seen_params.add(param)
                        self._geo_params.append(param)
                        idx = param_items
                        param_to_idx[param] = idx
                        param_items += 1
                elif type(param) is tuple:
                    if param in seen_params:
                        idx = param_to_idx[param]
                    else:
                        seen_params.add(param)
                        self._geo_params.extend(param)
                        idx = param_items
                        param_to_idx[param] = idx
                        param_items += len(param)
                else:
                    assert False
                    # FIXME: This needs to work with record arrays.
                    uniques = np.unique(param)
                    for value in uniques:
                        if value not in seen_params:
                            seen_params.add(value)
                            self._geo_params.extend(param)
                            param_items += len(param)

                self._encoded_param_map[param_map == node_key] = idx

                # TODO(kasiaj): Add support for sympy expressions.

        self._bits_param = bit_len(param_items)

    def encode(self):
        assert self._type_map is not None

        orientation = np.zeros_like(self._type_map)
        cnt = np.zeros_like(self._type_map)

        dry_types = self._type_map.dtype.type(nt.get_dry_node_type_ids())
        wet_types = self._type_map.dtype.type(nt.get_wet_node_type_ids())

        for i, vec in enumerate(self.subdomain.grid.basis):
            l = len(list(vec)) - 1
            shifted_map = self._type_map
            for j, shift in enumerate(vec):
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            cnt[util.in_anyd(shifted_map, dry_types)] += 1
            # FIXME: we're currently only processing the primary directions
            # here
            if vec.dot(vec) == 1:
                idx = np.logical_and(
                        util.in_anyd(self._type_map, dry_types),
                        util.in_anyd(shifted_map, wet_types))
                orientation[idx] = self.subdomain.grid.vec_to_dir(list(vec))

        # Remap type IDs.
        max_type_code = max(self._type_id_remap.keys())
        type_choice_map = np.zeros(max_type_code + 1, dtype=np.uint32)
        for orig_code, new_code in self._type_id_remap.iteritems():
            type_choice_map[orig_code] = new_code

        self._type_map[:] = self._encode_node(orientation,
                self._encoded_param_map,
                np.choose(np.int32(self._type_map), type_choice_map))

        # Drop the reference to the map array.
        self._type_map = None

    # XXX: Support different types of BCs here.
    def update_context(self, ctx):
        ctx.update({
            'node_types': self._node_types,
            'type_id_remap': self._type_id_remap,
            'nt_id_fluid': self._type_id(0),
            'nt_misc_shift': self._bits_type,
            'nt_type_mask': (1 << self._bits_type) - 1,
            'nt_param_shift': self._bits_param,
            'nt_dir_other': 0,  # used to indicate non-primary direction
                                # in orientation processing code
            'node_params': self._geo_params
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

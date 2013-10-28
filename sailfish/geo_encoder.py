"""Geometry encoding logic."""

__author__ = 'Michal Januszewski'
__email__ = 'sailfish-cfd@googlegroups.com'
__license__ = 'LGPL3'

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

    @property
    def dim(self):
        return self.subdomain.dim

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

    def tag_directions(self, tags):
        """
        :rvalue: True is there are any nodes supporting tagging, False otherwise
        """
        # For directions which are not periodic, keep the ghost nodes to avoid
        # detecting some missing directions.
        ngs = list(self.subdomain.spec._nonghost_slice)
        for i, periodic in enumerate(reversed(self.subdomain.spec._periodicity)):
            if not periodic:
                ngs[i] = slice(None)

        # Limit dry and wet types to these that are actually used in the simulation.
        uniq_types = set(np.unique(self._type_map.base))
        dry_types = list(set(nt.get_dry_node_type_ids()) & uniq_types)
        wet_types = list(set(nt.get_wet_node_type_ids()) & uniq_types)
        orient_types = list(set(nt.get_link_tag_node_type_ids()) & uniq_types)

        if not orient_types:
            return False

        # Convert to a numpy array.
        dry_types = self._type_map.dtype.type(dry_types)
        wet_types = self._type_map.dtype.type(wet_types)
        orient_types = self._type_map.dtype.type(orient_types)
        orient_map = util.in_anyd_fast(self._type_map[ngs], orient_types)
        l = self.subdomain.grid.dim - 1
        # Skip the stationary vector.
        for i, vec in enumerate(self.subdomain.grid.basis[1:]):
            shifted_map = self._type_map[ngs]
            for j, shift in enumerate(vec):
                if shift == 0:
                    continue
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            # If the given distribution points to a fluid node, tag it as
            # active.
            idx = orient_map & util.in_anyd_fast(shifted_map, wet_types)
            tags[ngs][idx] |= (1 << i)

        return True

    def detect_orientation(self, orientation):
        # Limit dry and wet types to these that are actually used in the simulation.
        uniq_types = set(np.unique(self._type_map.base))
        dry_types = list(set(nt.get_dry_node_type_ids()) & uniq_types)
        orient_types = list((set(nt.get_orientation_node_type_ids()) -
                            set(nt.get_link_tag_node_type_ids())) & uniq_types)

        if not orient_types:
            return

        # Convert to a numpy array.
        dry_types = self._type_map.dtype.type(dry_types)
        orient_types = self._type_map.dtype.type(orient_types)
        orient_map = util.in_anyd_fast(self._type_map, orient_types)
        l = self.subdomain.grid.dim - 1
        for vec in self.subdomain.grid.basis:
            # FIXME: we currently only process the primary directions
            if vec.dot(vec) != 1:
                continue
            shifted_map = self._type_map
            for j, shift in enumerate(vec):
                if shift == 0:
                    continue
                shifted_map = np.roll(shifted_map, int(-shift), axis=l-j)

            # Only set orientation where it's not already defined (=0).
            idx = orient_map & (shifted_map == 0) & (orientation == 0)
            orientation[idx] = self.subdomain.grid.vec_to_dir(list(vec))

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
        self._bits_scratch = 0
        self._type_map = None
        self._param_map = None
        self._geo_params = []
        self.config = subdomain.spec.runner.config
        self.scratch_space_size = 0
        self._unused_tag_bits = 0

    # TODO(michalj): Consider merging this funtionality into encode().
    def prepare_encode(self, type_map, param_map, param_dict, orientation,
                       detect_orientation):
        """
        :param type_map: uint32 array of NodeType.ids
        :param param_map: array whose entries are keys in param_dict
        :param param_dict: maps entries from param_map to LBNodeType objects
        :param detect_orientation: if True, will try to auto-detect the
            orientation of boundary nodes. This is used as an optimization to
            skip the relatively costly orientation detection in case there are
            no nodes requiring it.
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
        self._scratch_map = np.zeros_like(self._type_map)

        param_to_idx = dict()  # Maps entries in seen_params to ids.
        seen_params = set()
        param_items = 0
        self._symbol_map = {}  # Maps param indices to sympy expressions.

        # Refer to subdomain.Subdomain._verify_params for a list of allowed
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
                    self._encoded_param_map[param_map == node_key] = idx
                elif type(param) is tuple:
                    if param in seen_params:
                        idx = param_to_idx[param]
                    else:
                        seen_params.add(param)
                        self._geo_params.extend(param)
                        idx = param_items
                        param_to_idx[param] = idx
                        param_items += len(param)
                    self._encoded_param_map[param_map == node_key] = idx
                # Param is a structured numpy array.
                elif isinstance(param, np.ndarray):
                    nodes_idx = np.argwhere(param_map == node_key)

                    uniques = np.unique(param)
                    uniques.flags.writeable = False

                    for value in uniques:
                        if value in seen_params:
                            idx = param_to_idx[value]
                        else:
                            seen_params.add(value)
                            self._geo_params.extend(value)
                            idx = param_items
                            param_to_idx[value] = idx
                            param_items += len(value)

                        idxs = nodes_idx[param == value]
                        if idxs.shape[1] == 3:
                            self._encoded_param_map[idxs[:,0], idxs[:,1],
                                    idxs[:,2]] = idx
                        elif idxs.shape[1] == 2:
                            self._encoded_param_map[idxs[:,0], idxs[:,1]] = idx
                        else:
                            assert False, 'Unsupported dimension: {0}'.format(
                                    idxs.shape[1])

        self._non_symbolic_idxs = param_items

        # TODO(michalj): Verify that the type of the symbolic expression matches
        # that of the boundary condition (vector vs scalar, etc).
        # Second pass: only process symbolic expressions here.
        for node_key, node_type in param_dict.iteritems():
            for param in node_type.params.itervalues():
                if isinstance(param, nt.DynamicValue):
                    if param in seen_params:
                        idx = param_to_idx[value]
                    else:
                        seen_params.add(param)
                        idx = param_items
                        self._symbol_map[idx] = param
                        param_to_idx[param] = idx
                        param_items += 1
                    self._encoded_param_map[param_map == node_key] = idx

        self._bits_param = bit_len(param_items)

        # Maps node type ID to base offset within the scratch space array.
        self._scratch_space_base = {}
        type_to_node_count = {}
        # Generate unique (within node type) scratch space ids.
        for node_type in self._node_types:
            if node_type.scratch_space_size(self.dim) <= 0:
                continue

            def _selector(idx_list):
                return [slice(i, i+1) for i in idx_list]

            idx = np.argwhere(self._type_map == node_type.id)
            num_nodes = idx.shape[0]
            type_to_node_count[node_type.id] = num_nodes

            for i in xrange(num_nodes):
               self._scratch_map[_selector(idx[i,:])] = i

            self._scratch_space_base[node_type.id] = self.scratch_space_size

            # Accumulate size requirements of specific types into a global
            # size value for the whole scratch buffer.
            self.scratch_space_size += num_nodes * node_type.scratch_space_size(self.dim)

        if type_to_node_count:
            self._bits_scratch = bit_len(max(type_to_node_count.itervalues()))
        else:
            self._bits_scratch = 0

        self._have_link_tags = False

        if detect_orientation:
            # We do not reset the orientation array here as it is possible to
            # have orientation defined for some nodes and use autodetection for
            # others.
            self._have_link_tags = self.tag_directions(orientation)
            self.detect_orientation(orientation)

            # TODO: Actually drop these bits to save space in the node code.
            # It would be nice to use reduce here instead, but
            # bitwise_and.identity = 1 makes it impossible to use it.
            self._unused_tag_bits = int(np.bitwise_and.accumulate(
                orientation[orientation > 0])[-1])

            self.config.logger.debug('... orientation done.')

    def _subdomain_encode_node(self, orientation, node_type, param):
        """Helper method for use from Subdomain only.

        Use after encode() has been called, which initialized
        self._type_choice_map."""
        return self._encode_node(np.uint32(orientation), param,
                                 np.choose(np.int32(node_type),
                                           self._type_choice_map))

    def encode(self, orientation):
        """
        :param orientation: numpy array with the same layout as _type_map,
            indicating the orientation of different nodes; this array will be
            modified if detect_orientation is True.
        """
        assert self._type_map is not None
        self.config.logger.debug('Node type encoding...')

        # Remap type IDs.
        max_type_code = max(self._type_id_remap.keys())
        self._type_choice_map = np.zeros(max_type_code + 1, dtype=np.uint32)
        for orig_code, new_code in self._type_id_remap.iteritems():
            self._type_choice_map[orig_code] = new_code

        self._type_map[:] = self._encode_node(orientation,
                self._encoded_param_map,
                np.choose(np.int32(self._type_map), self._type_choice_map),
                self._scratch_map)
        self.config.logger.debug('... type map done.')

        # Drop the reference to the map array.
        self._type_map = None

    def get_param(self, location, values=1):
        """
        Returns 'values' float values which are pameters of the node at
        'location'.

        :param location: location of the node: x, y, [z] in the subdomain
                coordinate system (including ghost nodes)
        :param values: number of floating-point values to retrieve
        """
        idx = self._encoded_param_map[tuple(reversed(location))]
        return self._geo_params[idx:idx+values]

    def update_context(self, ctx):
        ctx.update({
            'use_link_tags': self.config.use_link_tags,
            'node_types': self._node_types,
            'type_id_remap': self._type_id_remap,
            'nt_id_fluid': self._type_id(0),
            'nt_misc_shift': self._bits_type,
            'nt_type_mask': (1 << self._bits_type) - 1,
            'nt_param_shift': self._bits_param,
            'nt_scratch_shift': self._bits_scratch,
            'nt_dir_other': 0,  # used to indicate non-primary direction
                                # in orientation processing code
            'node_params': self._geo_params,
            'symbol_idx_map': self._symbol_map,
            'non_symbolic_idxs': self._non_symbolic_idxs,
            'scratch_space': self.scratch_space_size > 0,
            'scratch_space_base': self._scratch_space_base,
            'unused_tag_bits': self._unused_tag_bits
        })

    def _encode_node(self, orientation, param, node_type, scratch_id=0):
        """Encodes information for a single node into a uint32.

        The node code consists of the following bit fields:
          orientation | scratch_index | param_index | node_type

                                                    ^_______ nt_misc_shift
                                      ^____ nt_param_shift + nt_misc_shift
                      ^_ nt_scratch_shift + nt_param_shift + nt_misc_shift

        """
        if (32 - self._bits_scratch < self.subdomain.grid.Q - 1 and
            self.config.use_link_tags and self._have_link_tags):
            raise ValueError('Not enough bits available to tag neighbor nodes.')

        misc_data = (orientation << self._bits_scratch) | scratch_id
        misc_data = (misc_data << self._bits_param) | param
        return (misc_data << self._bits_type) | node_type


# TODO: Implement this class.
class GeoEncoderBuffer(GeoEncoder):
    pass

# TODO: Implement this class.
class GeoEncoderMap(GeoEncoder):
    pass

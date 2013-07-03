<%
	import sailfish.node_type as nt
%>

<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>

typedef struct Dist {
%for i, dname in enumerate(grid.idx_name):
	float ${dname};
%endfor
} Dist;

%if model == 'mrt':
// Distribution in momentum space.
typedef struct DistM {
	%for i, dname in enumerate(grid.mrt_names):
		float ${dname};
	%endfor
} DistM;
%endif

// Functions for checking whether a node is of a given specific type.
%for nt_class in node_types:
	${device_func} inline bool is${nt_class.__name__}(unsigned int type) {
		return type == ${type_id_remap[nt_class.id]};
	}
%endfor

%if (nt.NTEquilibriumVelocity in node_types) or (nt.NTEquilibriumDensity in node_types):
${device_func} inline bool is_NTEquilibriumNode(unsigned int type) {
	return (false
	%if nt.NTEquilibriumVelocity in node_types:
		|| isNTEquilibriumVelocity(type)
	%endif
	%if nt.NTEquilibriumDensity in node_types:
		|| isNTEquilibriumDensity(type)
	%endif
	);
}
%endif

// Returns true is the node does not require any special processing
// to calculate macroscopic fields.
${device_func} inline bool NTUsesStandardMacro(unsigned int type) {
	return (false
		%for nt_class in node_types:
			%if nt_class.standard_macro:
				|| is${nt_class.__name__}(type)
			%endif
		%endfor
	);
}

// Wet nodes are nodes that undergo a standard collision procedure.
${device_func} inline bool isWetNode(unsigned int type) {
	return (false
		%for nt_class in node_types:
			%if nt_class.wet_node:
				|| is${nt_class.__name__}(type)
			%endif
		%endfor
	);
}

// Wet nodes are nodes that undergo a standard collision procedure.
${device_func} inline bool isExcludedNode(unsigned int type) {
	return (false
		%for nt_class in node_types:
			%if nt_class.excluded:
				|| is${nt_class.__name__}(type)
			%endif
		%endfor
	);
}

${device_func} inline bool isPropagationOnly(unsigned int type) {
	return (false
		%for nt_class in node_types:
			%if nt_class.propagation_only:
				|| is${nt_class.__name__}(type)
			%endif
		%endfor
	);
}

// Internal helper, do not use directly.
${device_func} inline void _storeNodeScratchSpace(unsigned int scratch_id,
	unsigned int num_values, float *buffer, ${global_ptr} float *g_buffer) {
	for (int i = 0; i < num_values; i++) {
		g_buffer[i + scratch_id * num_values] = buffer[i];

	}
}

// Internal helper, do not use directly.
${device_func} inline void _loadNodeScratchSpace(unsigned int scratch_id,
	unsigned int num_values, ${global_ptr} float *g_buffer, float *buffer) {
	for (int i = 0; i < num_values; i++) {
		buffer[i] = g_buffer[i + scratch_id * num_values];
	}
}

// Reads values from node scratch space (in global memory) into a local buffer.
//
// scratch_id: scratch space ID for nodes of type 'type'
// type: node type
// g_buffer: pointer to a buffer in the global memory used for scratch
//			 space
// buffer: pointer to a local buffer where the values will be saved
${device_func} inline void loadNodeScratchSpace(unsigned int scratch_id,
	 unsigned int type, ${global_ptr} float *g_buffer, float* buffer)
{
	switch (type) {
		%for nt_class in node_types:
			%if nt_class.scratch_space > 0:
				case ${type_id_remap[nt_class.id]}: {
					_loadNodeScratchSpace(scratch_id, ${nt_class.scratch_space_size(dim)},
								g_buffer + ${scratch_space_base[nt_class.id]},
								buffer);
					break;
				}
			%endif
		%endfor
	}
}

// Stores values from a local buffer into the node scratch space in global memory.
//
// Arguments: see loadNodeScratchSpace
${device_func} inline void storeNodeScratchSpace(unsigned int scratch_id,
	unsigned int type, float* buffer, ${global_ptr} float* g_buffer)
{
	switch (type) {
		%for nt_class in node_types:
			%if nt_class.scratch_space > 0:
				case ${type_id_remap[nt_class.id]}: {
					_storeNodeScratchSpace(scratch_id, ${nt_class.scratch_space_size(dim)},
								buffer, g_buffer + ${scratch_space_base[nt_class.id]});
					break;
				}
			%endif
		%endfor
	}
}

${device_func} inline unsigned int decodeNodeType(unsigned int nodetype) {
	return nodetype & ${nt_type_mask};
}

${device_func} inline unsigned int decodeNodeOrientation(unsigned int nodetype) {
	return nodetype >> ${nt_misc_shift + nt_param_shift + nt_scratch_shift};
}

// Returns the node's scratch ID, to be passed to (load,store)NodeScratchSpace as scratch_id.
${device_func} inline unsigned int decodeNodeScratchId(unsigned int nodetype) {
	return (nodetype >> ${nt_misc_shift + nt_param_shift}) & ${(1 << nt_scratch_shift)-1};
}

${device_func} inline unsigned int decodeNodeParamIdx(unsigned int nodetype) {
	return (nodetype >> ${nt_misc_shift}) & ${(1 << nt_param_shift)-1};
}

%if dim == 2:
	${device_func} inline int getGlobalIdx(int gx, int gy) {
		return gx + ${arr_nx} * gy;
	}

	${device_func} inline void decodeGlobalIdx(int gi, int *gx, int *gy) {
		*gx = gi % ${arr_nx};
		*gy = gi / ${arr_nx};
	}
%else:
	${device_func} inline int getGlobalIdx(int gx, int gy, int gz) {
		return gx + ${arr_nx} * gy + ${arr_nx * arr_ny} * gz;
	}

	${device_func} inline void decodeGlobalIdx(int gi, int *gx, int *gy, int *gz) {
		*gz = gi / ${arr_nx * arr_ny};
		int t = gi % ${arr_nx * arr_ny};
		*gy = t / ${arr_nx};
		*gx = t % ${arr_nx};
	}
%endif


${device_func} void die(void) {
	%if backend == 'cuda':
		asm("trap;");
	%else:
		return;
	%endif
}

${device_func} void checkInvalidValues(Dist* d, ${position_decl()}) {
	%if gpu_check_invalid_values:
		bool valid = true;
		%for dname in grid.idx_name:
			if (!isfinite(d->${dname})) {
				valid = false;
				printf("ERR(subdomain=${block.id}): Invalid value of ${dname} (%f) at: "
						%if dim == 2:
							"(%d, %d)"
						%else:
							"(%d, %d, %d)"
						%endif
						"\n", d->${dname}, ${position()});
			}
		%endfor

		if (!valid) {
			die();
		}
	%endif
}


<%
	def rel_offset(x, y, z=0):
		if grid.dim == 2:
			return x + y * arr_nx
		else:
			return x + arr_nx * (y + arr_ny * z)
%>

// Load the distributions from din to dout, for the node with the index 'idx'.
<%def name="get_dist_local()">
	%for i, dname in enumerate(grid.idx_name):
		dout->${dname} = ${get_dist('din', i, 'gi')};
	%endfor
</%def>

// Performs propagation when reading distributions from global memory.
// This implements the propagate-on-read scheme.
<%def name="get_unpropagated_dist()">
	%for i, (dname, ei) in enumerate(zip(grid.idx_name, grid.basis)):
		%if node_addressing == 'indirect':
			dout->${dname} = ${get_dist('din', i, 'nodes[dense_gi + %d]' % rel_offset(*(-ei)))}
		%else:
			dout->${dname} = ${get_dist('din', i, 'gi', offset=rel_offset(*(-ei)))};
		%endif
	%endfor
</%def>

// Implements the propagate-on-read scheme for the AA access pattern, where the
// distributions are not located in their natural slots, but the opposite ones
// (e.g. fNE is located where fSW normally is). This ensures that within a single
// timestep, the distributions are read from and written to the exact same places
// in global memory.
<%def name="get_unpropagated_dist_from_opposite_slots()">
	%for i, (dname, ei) in enumerate(zip(grid.idx_name, grid.basis)):
		%if node_addressing == 'indirect':
			dout->${dname} = ${get_dist('din', grid.idx_opposite[i], 'nodes[dense_gi + %d]' % rel_offset(*(-ei)))};
		%else:
			dout->${dname} = ${get_dist('din', grid.idx_opposite[i], 'gi', offset=rel_offset(*(-ei)))};
		%endif
	%endfor
</%def>

${device_func} inline void getDist(
		${nodes_array_if_required()}
		Dist *dout, ${global_ptr} ${const_ptr} float *__restrict__ din, int gi
		${dense_gi_if_required()}
		${iteration_number_if_required()}) {
	%if access_pattern == 'AB':
		%if propagate_on_read:
			${get_unpropagated_dist()}
		%else:
			${get_dist_local()}
		%endif
	%else:
		if ((iteration_number & 1) == 0) {
			${get_dist_local()}
		} else {
			${get_unpropagated_dist_from_opposite_slots()}
		}
	%endif
}

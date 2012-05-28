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

//
// Copy the idx-th distribution from din into dout.
//
${device_func} inline void getDist(Dist *dout, ${global_ptr} float *din, int idx)
{
	%for i, dname in enumerate(grid.idx_name):
		dout->${dname} = ${get_dist('din', i, 'idx')};
	%endfor
}

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

${device_func} inline unsigned int decodeNodeType(unsigned int nodetype) {
	return nodetype & ${nt_type_mask};
}

${device_func} inline unsigned int decodeNodeOrientation(unsigned int nodetype) {
	return nodetype >> ${nt_misc_shift + nt_param_shift};
}

${device_func} inline unsigned int decodeNodeParamIdx(unsigned int nodetype) {
	return (nodetype >> ${nt_misc_shift}) & ${(1 << nt_param_shift)-1};
}

%if dim == 2:
	${device_func} inline int getGlobalIdx(int gx, int gy) {
		return gx + ${arr_nx} * gy;
	}
%else:
	${device_func} inline int getGlobalIdx(int gx, int gy, int gz) {
		return gx + ${arr_nx} * gy + ${arr_nx * arr_ny} * gz;
	}
%endif


## Experimental code below
#############################################################################
<%
	def rel_offset(x, y, z=0):
		if grid.dim == 2:
			return x + y * arr_nx
		else:
			return x + arr_nx * (y + arr_ny * z)
%>

// Performs propagation when reading distributions from global memory.
// This implements the propagate-on-read scheme.
${device_func} inline void getUnpropagatedDist(Dist *dout, ${global_ptr} float *din, int idx) {
	%for i, (dname, ei) in enumerate(zip(grid.idx_name, grid.basis)):
		dout->${dname} = ${get_dist('din', i, 'idx', offset=rel_offset(*(-ei)))};
	%endfor
}


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

${device_func} inline bool isUnusedNode(unsigned int type) {
	return type == ${nt_id_unused};
}

${device_func} inline bool isFluidNode(unsigned int type) {
	return type == ${nt_id_fluid};
}

${device_func} inline bool isSlipNode(unsigned int type) {
	return type == ${nt_id_slip};
}

${device_func} inline bool isWallNode(unsigned int type) {
	return type == ${nt_id_wall};
}

${device_func} inline bool isFluidOrWallNode(unsigned int type) {
	return type <= ${nt_id_wall};
}

// This assumes we're dealing with a wall node.
${device_func} inline bool isVelocityNode(unsigned int type) {
	return type == ${nt_id_velocity};
}

${device_func} inline bool isPressureNode(unsigned int type) {
	return (type >= ${nt_id_pressure});
}

${device_func} inline bool isVelocityOrPressureNode(unsigned int type) {
	return isVelocityNode(type) || isPressureNode(type);
}

${device_func} inline bool isGhostNode(unsigned int type) {
	return (type == ${nt_id_ghost});
}

// Wet nodes are nodes that undergo a standard collision procedure.
${device_func} inline bool isWetNode(unsigned int type) {
	return (
		%if bc_wall_.wet_node:
			isFluidOrWallNode(type)
		%else:
			isFluidNode(type)
		%endif

		%if bc_velocity_.wet_node:
			|| isVelocityNode(type)
		%endif

		%if bc_pressure_.wet_node:
			|| isPressureNode(type)
		%endif
	);
}

${device_func} inline unsigned int decodeNodeType(unsigned int nodetype) {
	return nodetype & ${nt_type_mask};
}

${device_func} inline unsigned int decodeNodeOrientation(unsigned int nodetype) {
	return nodetype >> ${nt_misc_shift + nt_param_shift};
}

${device_func} inline unsigned int decodeNodeParam(unsigned int nodetype) {
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

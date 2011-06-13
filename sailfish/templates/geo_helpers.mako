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
	return type == ${geo_unused};
}

${device_func} inline bool isFluidNode(unsigned int type) {
	return type == ${geo_fluid};
}

${device_func} inline bool isSlipNode(unsigned int type) {
	return type == ${geo_slip};
}

${device_func} inline bool isWallNode(unsigned int type) {
	return type == ${geo_wall};
}

${device_func} inline bool isFluidOrWallNode(unsigned int type) {
	return type <= ${geo_wall};
}

// This assumes we're dealing with a wall node.
${device_func} inline bool isVelocityNode(unsigned int type) {
	return type == ${geo_velocity};
}

${device_func} inline bool isPressureNode(unsigned int type) {
	return (type >= ${geo_pressure});
}

${device_func} inline bool isVelocityOrPressureNode(unsigned int type) {
	return isVelocityNode(type) || isPressureNode(type);
}

${device_func} inline bool isGhostNode(unsigned int type) {
	return (type == ${geo_ghost});
}

// Wet nodes are nodes that undergo a standard collision procedure.
${device_func} inline bool isWetNode(unsigned int type) {
	return (
		%if bc_wall_.wet_nodes:
			isFluidOrWallNode(type)
		%else:
			isFluidNode(type)
		%endif

		%if bc_velocity_.wet_nodes:
			|| isVelocityNode(type)
		%endif

		%if bc_pressure_.wet_nodes:
			|| isPressureNode(type)
		%endif
	);
}

${device_func} inline unsigned int decodeNodeType(unsigned int nodetype) {
	return nodetype & ${geo_type_mask};
}

${device_func} inline unsigned int decodeNodeOrientation(unsigned int nodetype) {
	return nodetype >> ${geo_misc_shift + geo_param_shift};
}

${device_func} inline unsigned int decodeNodeParam(unsigned int nodetype) {
	return (nodetype >> ${geo_misc_shift}) & ${(1 << geo_param_shift)-1};
}

${device_func} inline unsigned int encodeBoundaryNode(unsigned int dir_mask, unsigned int obj_id) {
	return ${geo_boundary} | (obj_id << ${geo_misc_shift}) | (dir_mask << ${geo_misc_shift + geo_obj_shift});
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

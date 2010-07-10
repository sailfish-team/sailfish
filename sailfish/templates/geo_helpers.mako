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
		dout->${dname} = din[idx + DIST_SIZE*${i}];
	%endfor
}

${device_func} inline bool isUnusedNode(int type) {
	return type == ${geo_unused};
}

${device_func} inline bool isFluidNode(int type) {
	return type == ${geo_fluid};
}

${device_func} inline bool isWallNode(int type) {
	return type == ${geo_wall};
}

${device_func} inline bool isFluidOrWallNode(int type) {
	return type <= ${geo_wall};
}

// This assumes we're dealing with a wall node.
${device_func} inline bool isVelocityNode(int type) {
	return type == ${geo_velocity};
}

${device_func} inline bool isPressureNode(int type) {
	return (type >= ${geo_pressure});
}

${device_func} inline bool isVelocityOrPressureNode(int type) {
	return isVelocityNode(type) || isPressureNode(type);
}

// Wet nodes are nodes that undergo a standard collision procedure.
${device_func} inline bool isWetNode(int type) {
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

${device_func} inline int decodeNodeType(int nodetype) {
	return nodetype & ${geo_type_mask};
}

${device_func} inline int decodeNodeOrientation(int nodetype) {
	return nodetype >> ${geo_misc_shift + geo_param_shift};
}

${device_func} inline int decodeNodeParam(int nodetype) {
	return (nodetype >> ${geo_misc_shift}) & ${(1 << (geo_param_shift+1))-1};
}

${device_func} inline int encodeBoundaryNode(int dir_mask, int obj_id) {
	return ${geo_boundary} | (obj_id << ${geo_misc_shift}) | (dir_mask << ${geo_misc_shift + geo_obj_shift});
}

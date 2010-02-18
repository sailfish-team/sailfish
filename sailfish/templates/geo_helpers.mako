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
	return (type >= ${geo_bcv}) && (type < GEO_BCP);
}

${device_func} inline bool isVelocityOrPressureNode(int type) {
	return (type >= ${geo_bcv});
}

${device_func} inline bool isPressureNode(int type) {
	return (type >= ${geo_bcp});
}

${device_func} inline void decodeNodeType(int nodetype, int *orientation, int *type) {
	*orientation = nodetype & ${geo_orientation_mask};
	*type = nodetype >> ${geo_orientation_shift};
}

${device_func} inline int encodeNodeType(int orientation, int type) {
	return orientation | (type << ${geo_orientation_shift});
}

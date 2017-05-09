<%!
    from sailfish import sym
%>

<%namespace file="code_common.mako" import="dump_dists"/>
<%namespace file="kernel_common.mako" import="iteration_number_if_required,get_dist"/>

<%
	def rel_offset(x, y, z=0):
		if grid.dim == 2:
			return x + y * arr_nx
		else:
			return x + arr_nx * (y + arr_ny * z)
%>

${kernel} void HandleNTCopyNodes(
	${global_ptr} int *map,
	${global_ptr} float* dist_in
	${iteration_number_if_required()}
) {
	int gx = get_global_id(0) + ${envelope_size};
	int gy = get_global_id(1) + ${envelope_size};

	if (gx >= ${lat_nx - envelope_size} || gy >= ${lat_ny - envelope_size}) {
		return;
	}

%if access_pattern == 'AA':
	int gi_dst = getGlobalIdx(gx, gy, ${lat_nz - 2});
	int gi_src = getGlobalIdx(gx, gy, ${lat_nz - 3});
	float t;
	// Called with an updated iteration number.
	if ((iteration_number & 1) == 0) {
		%for dist_idx in sym.get_missing_dists(grid, 6):
			t = ${get_dist('dist_in', dist_idx, 'gi_src')};
			${get_dist('dist_in', dist_idx, 'gi_dst')} = t;
		%endfor
	} else {
		%for dist_idx in sym.get_missing_dists(grid, 6):
			t = ${get_dist('dist_in', grid.idx_opposite[dist_idx], 'gi_src', offset=rel_offset(*(-grid.basis[dist_idx])))};
			${get_dist('dist_in', grid.idx_opposite[dist_idx], 'gi_dst', offset=rel_offset(*(-grid.basis[dist_idx])))} = t;
		%endfor
	}
%endif
}

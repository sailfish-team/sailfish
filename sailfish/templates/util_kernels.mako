<%!
    from sailfish import sym
%>

<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>

<%def name="pbc_helper(axis, max_dim, max_dim2=None)">
	<%
		if axis == 0:
			offset = 1
		elif axis == 1:
			offset = arr_nx
		else:
			offset = arr_nx * arr_ny

		other_axes = [[1,2], [0,2], [0,1]]

		def make_cond_to_dists(axis_direction):
			direction = [0] * dim
			direction[axis] = axis_direction

			cond_to_dists = {}

			if dim == 2:
				direction[1 - axis] = 1
				corner_dists = sym.get_interblock_dists(grid, direction)
				cond_to_dists['idx1 > 1'] = corner_dists
				direction[1 - axis] = -1
				corner_dists = sym.get_interblock_dists(grid, direction)
				cond_to_dists['idx1 < {0}'.format(max_dim)] = corner_dists
			else:
				for i in (1, 0, -1):
					for j in (1, 0, -1):
						if i == 0 and j == 0:
							continue
						direction[other_axes[axis][0]] = i
						direction[other_axes[axis][1]] = j
						corner_dists = sym.get_interblock_dists(grid, direction)
						conds = []
						if i == 1:
							conds.append('idx1 > 1')
						elif i == -1:
							conds.append('idx1 < {0}'.format(max_dim))
						if j == 1:
							conds.append('idx2 > 1')
						elif j == -1:
							conds.append('idx2 < {0}'.format(max_dim2))
						cond = ' && '.join(conds)
						cond_to_dists[cond] = corner_dists

			return cond_to_dists

		cond_to_dists = make_cond_to_dists(-1)
		done = False
		done_dists = set()
	%>

	// TODO(michalj): Generalize this for grids with e_i > 1.
	// From low idx to high idx.
	%for i in sym.get_prop_dists(grid, -1, axis):
		float f${grid.idx_name[i]} = ${get_dist('dist', i, 'gi_low')};
	%endfor

	%for i in sym.get_prop_dists(grid, -1, axis):
		%if grid.basis[i].dot(grid.basis[i]) > 1:
			%for cond, dists in cond_to_dists.iteritems():
				%if i in dists:
					// Skip distributions which are not populated.
					if (${cond}) {
						${get_dist('dist', i, 'gi_high')} = f${grid.idx_name[i]};
					}
					<%
						done = True
						# Keep track of the distributions and make sure no distribution
						# appears with two different conditiosn.
						assert i not in done_dists
						done_dists.add(i)
					%>
				%endif
			%endfor

			%if not done:
				__BUG__
			%endif
		%else:
			${get_dist('dist', i, 'gi_high')} = f${grid.idx_name[i]};
		%endif
	%endfor

	// From high idx to low idx.
	%for i in sym.get_prop_dists(grid, 1, axis):
		float f${grid.idx_name[i]} = ${get_dist('dist', i, 'gi_high', offset)};
	%endfor

	<%
		cond_to_dists = make_cond_to_dists(1)
		done = False
		done_dists = set()
	%>

	%for i in sym.get_prop_dists(grid, 1, axis):
		%if grid.basis[i].dot(grid.basis[i]) > 1:
			%for cond, dists in cond_to_dists.iteritems():
				%if i in dists:
					// Skip distributions which are not populated.
					if (${cond}) {
						${get_dist('dist', i, 'gi_low', offset)} = f${grid.idx_name[i]};
					}
					<%
						done = True
						# Keep track of the distributions and make sure no distribution
						# appears with two different conditiosn.
						assert i not in done_dists
						done_dists.add(i)
					%>
				%endif
			%endfor

			%if not done:
				__BUG__
			%endif
		%else:
			${get_dist('dist', i, 'gi_low', offset)} = f${grid.idx_name[i]};
		%endif
	%endfor
</%def>

// Applies periodic boundary conditions within a single block.
//  dist: pointer to the distributions array
//  axis: along which axis the PBCs are to be applied (0:x, 1:y, 2:z)
${kernel} void ApplyPeriodicBoundaryConditions(
		${global_ptr} float *dist, int axis)
{
	int idx1 = get_global_id(0);
	int gi_low, gi_high;

	// For single block PBC, the envelope size (width of the ghost node
	// layer is always 1.
	%if dim == 2:
		if (axis == 0) {
			if (idx1 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(0, idx1);
			gi_high = getGlobalIdx(${lat_nx-2}, idx1);
			${pbc_helper(0, lat_ny-2)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx}) { return; }
			gi_low = getGlobalIdx(idx1, 0);
			gi_high = getGlobalIdx(idx1, ${lat_ny-2});
			${pbc_helper(1, lat_nx-2)}
		}
	%else:
		int idx2 = get_global_id(1);
		if (axis == 0) {
			if (idx1 >= ${lat_ny} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(0, idx1, idx2);
			gi_high = getGlobalIdx(${lat_nx-2}, idx1, idx2);
			${pbc_helper(0, lat_ny-2, lat_nz-2)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(idx1, 0, idx2);
			gi_high = getGlobalIdx(idx1, ${lat_ny-2}, idx2);
			${pbc_helper(1, lat_nx-2, lat_nz-2)}
		} else {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(idx1, idx2, 0);
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-2});
			${pbc_helper(2, lat_nx-2, lat_ny-2)}
		}
	%endif
}

%if dim == 2:
// Collects ghost node data for connections along axes other than X.
// dist: distributions array
// base_gy: where along the X axis to start collecting the data
// face: see LBBlock class constants
// buffer: buffer where the data is to be saved
${kernel} void CollectContinuousData(
		${global_ptr} float *dist, int face, int base_gx,
		int max_lx, ${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	int gi;
	float tmp;

	if (idx >= max_lx) {
		return;
	}

	switch (face) {
	%for axis in range(2, 2*dim):
		case ${axis}: {
			<%
				normal = block.face_to_normal(axis)
				dists = sym.get_interblock_dists(grid, normal)
			%>
			int dist_size = max_lx / ${len(dists)};
			int dist_num = idx / dist_size;
			int gx = idx % dist_size;

			switch (dist_num) {
				%for i, prop_dist in enumerate(dists):
				case ${i}: {
					gi = getGlobalIdx(base_gx + gx, ${lat_linear[axis]});
					tmp = ${get_dist('dist', prop_dist, 'gi')};
					break;
				}
				%endfor
			}
			buffer[idx] = tmp;
			break;
		}
	%endfor
	}
}
%else:
<%def name="_get_global_idx(axis)">
	## Y-axis
	%if axis < 4:
		gi = getGlobalIdx(base_gx + gx, ${lat_linear[axis]}, base_other + other);
	## Z-axis
	%else:
		gi = getGlobalIdx(base_gx + gx, base_other + other, ${lat_linear[axis]});
	%endif
</%def>

// The data is collected from a rectangular area of the plane corresponding to 'face'.
// The grid with which the kernel is to be called has the following dimensions:
//
//  x: # nodes along the X direction + any padding (real # nodes is identified by max_lx)
//  y: # nodes along the Y/Z direction * # of dists to transfer + any padding
//
// The data will be placed into buffer, in the following linear layout:
//
// (x0, y0, d0), (x1, y0, d0), .. (xN, y0, d0),
// (x0, y1, d0), (x1, y1, d0), .. (xN, y1, d0),
// ..
// (x0, yM, d0), (x1, yM, d0). .. (xN, yM, d0),
// (x0, y0, d1), (x1, y0, d1), .. (xN, y0, d1),
// ...
${kernel} void CollectContinuousData(
	${global_ptr} float *dist, int face, int base_gx, int base_other,
	int max_lx, int max_other, ${global_ptr} float *buffer)
{
	int gx = get_global_id(0);
	int idx = get_global_id(1);
	int gi;
	float tmp;

	if (gx >= max_lx || idx >= max_other) {
		return;
	}

	// TODO: consider intrabuffer padding to increase efficiency of writes
	switch (face) {
	%for axis in range(2, 2*dim):
		case ${axis}: {
			<%
				normal = block.face_to_normal(axis)
				dists = sym.get_interblock_dists(grid, normal)
			%>
			int dist_size = max_other / ${len(dists)};
			int dist_num = idx / dist_size;
			int other = idx % dist_size;

			switch (dist_num) {
				%for i, prop_dist in enumerate(dists):
				case ${i}: {
					${_get_global_idx(axis)};
					tmp = ${get_dist('dist', prop_dist, 'gi')};
					break;
				}
				%endfor
			}

			idx = (dist_size * max_lx * dist_num) + (other * max_lx) + gx;
			buffer[idx] = tmp;
			break;
		}
	%endfor
	}
}
%endif

%if dim == 2:
${kernel} void DistributeContinuousData(
		${global_ptr} float *dist, int face, int base_gx,
		int max_lx, ${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	int gi;

	if (idx >= max_lx) {
		return;
	}

	switch (face) {
	%for axis in range(2, 2*dim):
		case ${axis}: {
			<%
				normal = block.face_to_normal(axis)
				dists = sym.get_interblock_dists(grid, normal)
			%>
			int dist_size = max_lx / ${len(dists)};
			int dist_num = idx / dist_size;
			int gx = idx % dist_size;
			float tmp = buffer[idx];
			switch (dist_num) {
				%for i, prop_dist in enumerate(dists):
				case ${i}: {
					gi = getGlobalIdx(base_gx + gx, ${lat_linear_dist[axis]});
					${get_dist('dist', prop_dist, 'gi')} = tmp;
					break;
				}
				%endfor
			}

			break;
		}
	%endfor
	}
}
%else:
## 3D
<%def name="_get_global_dist_idx(axis)">
	## Y-axis
	%if axis < 4:
		gi = getGlobalIdx(base_gx + gx, ${lat_linear_dist[axis]}, base_other + other);
	## Z-axis
	%else:
		gi = getGlobalIdx(base_gx + gx, base_other + other, ${lat_linear_dist[axis]});
	%endif
</%def>

// Layout of the data in the buffer is the same as in the output buffer of
// CollectOrthogonalGhostData.
${kernel} void DistributeContinuousData(
		${global_ptr} float *dist, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	int gx = get_global_id(0);
	int idx = get_global_id(1);
	int gi;

	if (gx >= max_lx || idx >= max_other) {
		return;
	}

	switch (face) {
	%for axis in range(2, 2*dim):
		case ${axis}: {
			<%
				normal = block.face_to_normal(axis)
				dists = sym.get_interblock_dists(grid, normal)
			%>
			int dist_size = max_other / ${len(dists)};
			int dist_num = idx / dist_size;
			int other = idx % dist_size;
			idx = (dist_size * max_lx * dist_num) + (other * max_lx) + gx;
			float tmp = buffer[idx];

			switch (dist_num) {
				%for i, prop_dist in enumerate(dists):
				case ${i}: {
					${_get_global_dist_idx(axis)}
					${get_dist('dist', prop_dist, 'gi')} = tmp;
					break;
				}
				%endfor
			}
			break;
		}
	%endfor
	}
}
%endif

${kernel} void CollectSparseData(
		${global_ptr} int *idx_array, ${global_ptr} float *dist,
		${global_ptr} float *buffer, int max_idx)
{
	int idx = get_global_id(0);
	%if dim > 2:
		idx += get_global_size(0) * get_global_id(1);
	%endif

	if (idx >= max_idx) {
		return;
	}
	int gi = idx_array[idx];
	buffer[idx] = dist[gi];
}

${kernel} void DistributeSparseData(
		${global_ptr} int *idx_array, ${global_ptr} float *dist,
		${global_ptr} float *buffer, int max_idx)
{
	int idx = get_global_id(0);
	%if dim > 2:
		idx += get_global_size(0) * get_global_id(1);
	%endif
	if (idx >= max_idx) {
		return;
	}
	int gi = idx_array[idx];
	dist[gi] = buffer[idx];
}

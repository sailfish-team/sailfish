## Utility kernels for moving data around on the device.
## This includes:
##  - handling periodic boundary conditions via ghost nodes
##  - collecting data for transfer to other computational nodes
##  - distributing data received from other computational nodes
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

		# Maps axis number to a list of other axes.
		other_axes = [[1,2], [0,2], [0,1]]

		def handle_corners(basis, axis_target):
			"""
			:param basis: basis vector
			:param axis_target: location of the target node along the dimension
				aligned with the axis to which PBCs are applied

			:rvalue: tuple of: conditions on the index variables; target string
				to be passed to getGlobalIdx
			"""
			if basis[axis] == 0:
				return None, None

			if dim == 2:
				other = [1 - axis]
			else:
				other = other_axes[axis]

			ret_cond = []
			ret_targ = []

			# Conditions.
			conditions = []

			# Location of the target node.
			target = [''] * dim
			target[axis] = str(axis_target)

			# Corner nodes (cross three PBC boundaries).
			for i, other_ax in enumerate(other):
				if basis[other_ax] == 0 or not block_periodicity[other_ax]:
					target[other_ax] = 'idx%d' % (i + 1)
					continue

				if basis[other_ax] == 1:
					conditions.append('idx%d == %d' % ((i + 1), bnd_limits[other_ax] - 1))
					target[other_ax] = '1'
				elif basis[other_ax] == -1:
					conditions.append('idx%d == 0' % (i + 1))
					target[other_ax] = str(bnd_limits[other_ax] - 2)
				else:
					raise ValueError("Unexpected basis vector component.")

			ret_cond.append(' && '.join(conditions))
			ret_targ.append(', '.join(target))

			# Edge nodes.(cross two PBC boundaries).
			if sum((abs(i) for i in basis)) == 3:
				for i, other_ax in enumerate(other):
					id_ = 'idx%d' % (i + 1)
					id2 = 'idx%d' % (2 - i)

					target = [''] * dim
					target[axis] = str(axis_target)
					conditions = []

					if basis[other_ax] == 1:
						conditions.append('%s >= %d && %s < %d' % (id_, 2, id_, bnd_limits[other_ax] - 1))
						target[other_ax] = id_
					elif basis[other_ax] == -1:
						conditions.append('%s < %d && %s >= 1' % (id_, bnd_limits[other_ax] - 2, id_))
						target[other_ax] = id_
					else:
						raise ValueError("Unexpected basis vector component.")

					ax = other[1 - i]
					if basis[ax] == 1:
						conditions.append('%s == %d' % (id2, bnd_limits[ax] - 1))
						target[ax] = '1'
					elif basis[ax] == -1:
						conditions.append('%s == 0' % id2)
						target[ax] = str(bnd_limits[ax] - 2)

					ret_targ.append(', '.join(target))
					ret_cond.append(' && '.join(conditions))

			return ret_cond, ret_targ

		def make_cond(i):
			"""Returns a string of representing a boolean condition that has
			to be satisfied in order for the specified distribution to be
			moved to the target along a single axis.

			:param i: distribution index
			"""

			conds = []

			if dim == 2:
				if grid.basis[i][1 - axis] == 1:
					return 'idx1 > 1 && idx1 <= {0}'.format(max_dim)
				elif grid.basis[i][1 - axis] == -1:
					return 'idx1 < {0} && idx1 >= 1'.format(max_dim)
			else:
				oa1 = other_axes[axis][0]
				oa2 = other_axes[axis][1]

				if grid.basis[i][oa1] == 1:
					cond = 'idx1 > 1'
					if block_periodicity[axis] and block_periodicity[oa1]:
						cond += ' && idx1 <= {0}'.format(max_dim)
					conds.append(cond)
				elif grid.basis[i][oa1] == -1:
					cond = 'idx1 < {0}'.format(max_dim)
					if block_periodicity[axis] and block_periodicity[oa1]:
						cond += ' && idx1 >= 1'
					conds.append(cond)
				if grid.basis[i][oa2] == 1:
					cond = 'idx2 > 1'
					if block_periodicity[axis] and block_periodicity[oa2]:
						cond += ' && idx2 <= {0}'.format(max_dim2)
					conds.append(cond)
				elif grid.basis[i][oa2] == -1:
					cond = 'idx2 < {0}'.format(max_dim2)
					if block_periodicity[axis] and block_periodicity[oa2]:
						cond += ' && idx2 >= 1'
					conds.append(cond)
			return ' && '.join(conds)
	%>

	// TODO(michalj): Generalize this for grids with e_i > 1.
	// From low idx to high idx.
	%for i in sym.get_prop_dists(grid, -1, axis):
		float f${grid.idx_name[i]} = ${get_dist('dist', i, 'gi_low')};
	%endfor

	%for i in sym.get_prop_dists(grid, -1, axis):
		%if grid.basis[i].dot(grid.basis[i]) > 1:
			if (isfinite(f${grid.idx_name[i]})) {
				// Skip distributions which are not populated or cross multiple boundaries.
				if (${make_cond(i)}) {
					${get_dist('dist', i, 'gi_high')} = f${grid.idx_name[i]};
				}
				<% corner_cond, target = handle_corners(grid.basis[i], bnd_limits[axis] - 2) %>
				%if corner_cond:
					else {
						if (0) {}
						%for cond, targ in zip(corner_cond, target):
							%if cond:
								else if (${cond}) {
									int gi_high2 = getGlobalIdx(${targ});
									${get_dist('dist', i, 'gi_high2')} = f${grid.idx_name[i]};
								}
							%endif
						%endfor
					}
				%endif
			}
		%else:
			if (isfinite(f${grid.idx_name[i]})) {
				${get_dist('dist', i, 'gi_high')} = f${grid.idx_name[i]};
			}
		%endif
	%endfor

	// From high idx to low idx.
	%for i in sym.get_prop_dists(grid, 1, axis):
		float f${grid.idx_name[i]} = ${get_dist('dist', i, 'gi_high', offset)};
	%endfor

	%for i in sym.get_prop_dists(grid, 1, axis):
		%if grid.basis[i].dot(grid.basis[i]) > 1:
			if (isfinite(f${grid.idx_name[i]})) {
				// Skip distributions which are not populated or cross multiple boundaries.
				if (${make_cond(i)}) {
					${get_dist('dist', i, 'gi_low', offset)} = f${grid.idx_name[i]};
				}
				<% corner_cond, target = handle_corners(grid.basis[i], 1) %>
				%if corner_cond:
					else {
						if (0) {}
						%for cond, targ in zip(corner_cond, target):
							%if cond:
								else if (${cond}) {
									int gi_low2 = getGlobalIdx(${targ});
									${get_dist('dist', i, 'gi_low2')} = f${grid.idx_name[i]};
								}
							%endif
						%endfor
					}
				%endif
			}
		%else:
			if (isfinite(f${grid.idx_name[i]})) {
				${get_dist('dist', i, 'gi_low', offset)} = f${grid.idx_name[i]};
			}
		%endif
	%endfor
</%def>

// Applies periodic boundary conditions within a single subdomain.
//  dist: pointer to the distributions array
//  axis: along which axis the PBCs are to be applied (0:x, 1:y, 2:z)
${kernel} void ApplyPeriodicBoundaryConditions(
		${global_ptr} float *dist, int axis)
{
	int idx1 = get_global_id(0);
	int gi_low, gi_high;

	// For single block PBC, the envelope size (width of the ghost node
	// layer) is always 1.
	// TODO(michalj): Generalize this for the case when envelope_size != 1.
	%if dim == 2:
		if (axis == 0) {
			if (idx1 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(0, idx1);				// ghost node
			gi_high = getGlobalIdx(${lat_nx-2}, idx1);	// read node
			${pbc_helper(0, lat_ny-2)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx}) { return; }
			gi_low = getGlobalIdx(idx1, 0);				// ghost node
			gi_high = getGlobalIdx(idx1, ${lat_ny-2});	// real node
			${pbc_helper(1, lat_nx-2)}
		}
	%else:
		int idx2 = get_global_id(1);
		if (axis == 0) {
			if (idx1 >= ${lat_ny} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(0, idx1, idx2);				// ghost node
			gi_high = getGlobalIdx(${lat_nx-2}, idx1, idx2);	// real node
			${pbc_helper(0, lat_ny-2, lat_nz-2)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(idx1, 0, idx2);				// ghost node
			gi_high = getGlobalIdx(idx1, ${lat_ny-2}, idx2);	// real node
			${pbc_helper(1, lat_nx-2, lat_nz-2)}
		} else {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(idx1, idx2, 0);				// ghost node
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-2});	// real node
			${pbc_helper(2, lat_nx-2, lat_ny-2)}
		}
	%endif
}


<%def name="_copy_field_if_finite(src, dest)">
	if (isfinite(field[${src}])) {
		field[${dest}] = field[${src}];
	}
</%def>

// Applies periodic boundary conditions to a scalar field within a single subdomain.
//  dist: pointer to the array with the field data
//  axis: along which axis the PBCs are to be applied (0:x, 1:y, 2:z)
${kernel} void ApplyMacroPeriodicBoundaryConditions(
		${global_ptr} float *field, int axis)
{
	int idx1 = get_global_id(0);
	int gi_low, gi_high;

	// TODO(michalj): Generalize this for the case when envelope_size != 1.
	%if dim == 2:
		if (axis == 0) {
			if (idx1 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(0, idx1);					// ghost node
			gi_high = getGlobalIdx(${lat_nx-2}, idx1);		// real node
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(1, idx1);					// real node
			gi_high = getGlobalIdx(${lat_nx-1}, idx1);		// ghost node
			${_copy_field_if_finite('gi_low', 'gi_high')}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx}) { return; }
			gi_low = getGlobalIdx(idx1, 0);					// ghost node
			gi_high = getGlobalIdx(idx1, ${lat_ny-2});		// real node
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(idx1, 1);					// real node
			gi_high = getGlobalIdx(idx1, ${lat_ny-1});		// ghost node
			${_copy_field_if_finite('gi_low', 'gi_high')}
		}
	%else:
		int idx2 = get_global_id(1);
		if (axis == 0) {
			if (idx1 >= ${lat_ny} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(0, idx1, idx2);				// ghost node
			gi_high = getGlobalIdx(${lat_nx-2}, idx1, idx2);	// real node
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(1, idx1, idx2);				// real node
			gi_high = getGlobalIdx(${lat_nx-1}, idx1, idx2);	// ghost node
			${_copy_field_if_finite('gi_low', 'gi_high')}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(idx1, 0, idx2);				// ghost node
			gi_high = getGlobalIdx(idx1, ${lat_ny-2}, idx2);	// real node
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(idx1, 1, idx2);				// real node
			gi_high = getGlobalIdx(idx1, ${lat_ny-1}, idx2);	// ghost node
			${_copy_field_if_finite('gi_low', 'gi_high')}
		} else {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(idx1, idx2, 0);				// ghsot node
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-2});	// real node
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(idx1, idx2, 1);				// real node
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-1});	// ghost node
			${_copy_field_if_finite('gi_low', 'gi_high')}
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
// CollectContinuousData.
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

%if dim == 2:
${kernel} void CollectContinuousMacroData(
		${global_ptr} float *field, int base_gx, int max_lx, int gy,
		${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	if (idx >= max_lx) {
		return;
	}

	int gi = getGlobalIdx(base_gx + idx, gy);
	float tmp = field[gi];
	buffer[idx] = tmp;
}
%else:
<%def name="_get_global_macro_idx(face)">
	## Y-axis
	%if face < 4:
		gi = getGlobalIdx(base_gx + gx, ${lat_linear_macro[face]}, base_other + other);
	## Z-axis
	%else:
		gi = getGlobalIdx(base_gx + gx, base_other + other, ${lat_linear_macro[face]});
	%endif
</%def>

// The data is collected from a rectangular area of the plane corresponding to 'face'.
// The grid with which the kernel is to be called has the following dimensions:
//
//  x: # nodes along the X direction + any padding (real # nodes is identified by max_lx)
//  y: # nodes along the Y/Z direction
//
${kernel} void CollectContinuousMacroData(
		${global_ptr} float *field, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	int gx = get_global_id(0);
	int other = get_global_id(1);

	if (gx >= max_lx || other >= max_other) {
		return;
	}

	// Index in the output buffer.
	int idx = other * get_global_size(0) + gx;

	switch (face) {
	%for axis in range(2, 2 * dim):
		case ${axis}: {
			int gi;
			${_get_global_macro_idx(axis)};
			float tmp = field[gi];
			buffer[idx] = tmp;
			break;
		}
	%endfor
	}
}
%endif  ## dim == 3

%if dim == 2:
${kernel} void DistributeContinuousMacroData(
		${global_ptr} float *field, int base_gx, int max_lx, int gy,
		${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	if (idx >= max_lx) {
		return;
	}

	float tmp = buffer[idx];
	int gi = getGlobalIdx(base_gx + idx, gy);
	field[gi] = tmp;
}
%else:
<%def name="_get_global_macro_dist_idx(face)">
	## Y-axis
	%if face < 4:
		gi = getGlobalIdx(base_gx + gx, ${lat_linear[face]}, base_other + other);
	## Z-axis
	%else:
		gi = getGlobalIdx(base_gx + gx, base_other + other, ${lat_linear[face]});
	%endif
</%def>

${kernel} void DistributeContinuousMacroData(
		${global_ptr} float *field, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	int gx = get_global_id(0);
	int other = get_global_id(1);

	if (gx >= max_lx || other >= max_other) {
		return;
	}

	// Index in the input buffer.
	int idx = other * get_global_size(0) + gx;
	float tmp = buffer[idx];

	switch (face) {
	%for axis in range(2, 2 * dim):
		case ${axis}: {
			int gi;
			${_get_global_macro_dist_idx(axis)};
			field[gi] = tmp;
			break;
		}
	%endfor
	}
}
%endif  ## dim == 3

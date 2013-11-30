## Utility kernels for moving data around on the device.
## This includes:
##  - handling periodic boundary conditions via ghost nodes
##  - collecting data for transfer to other computational nodes
##  - distributing data received from other computational nodes
<%!
    from sailfish import sym
%>

<%include file="kernel_force_objects.mako"/>

<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>

##
## opposite: if True, the distributions are read and written to slots
##           opposite to where they normally should be; this is used
##           after the fully local step in the AA access pattern.
##
<%def name="pbc_helper(axis, max_dim, max_dim2=None, opposite=False)">
	<%
		if axis == 0:
			offset = 1
		elif axis == 1:
			offset = arr_nx
		else:
			offset = arr_nx * arr_ny

		if opposite:
			offset = -offset

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
					shift = 1 if not opposite else 2
					conditions.append('idx%d == %d' % ((i + 1), bnd_limits[other_ax] - shift))
					target[other_ax] = '1' if not opposite else '0'
				elif basis[other_ax] == -1:
					shift = 0 if not opposite else 1
					conditions.append('idx%d == %d' % ((i + 1), shift))
					shift = 2 if not opposite else 1
					target[other_ax] = str(bnd_limits[other_ax] - shift)
				else:
					raise ValueError("Unexpected basis vector component.")

			ret_cond.append(' && '.join(conditions))
			ret_targ.append(', '.join(target))

			# Edge nodes (cross two PBC boundaries).
			if sum((abs(i) for i in basis)) == 3:
				for i, other_ax in enumerate(other):
					id_ = 'idx%d' % (i + 1)
					id2 = 'idx%d' % (2 - i)

					target = [''] * dim
					target[axis] = str(axis_target)
					conditions = []

					# XXX: verify this code
					if basis[other_ax] == 1:
						shift = 1 if not opposite else 2
						start = 2 if not opposite else 1
						conditions.append('%s >= %d && %s < %d' % (id_, start, id_, bnd_limits[other_ax] - shift))
						target[other_ax] = id_
					elif basis[other_ax] == -1:
						shift = 2 if not opposite else 1
						conditions.append('%s < %d && %s >= 1' % (id_, bnd_limits[other_ax] - shift, id_))
						target[other_ax] = id_
					else:
						raise ValueError("Unexpected basis vector component.")

					# The other 'other' axis.
					ax = other[1 - i]
					if basis[ax] == 1:
						shift = 1 if not opposite else 2
						conditions.append('%s == %d' % (id2, bnd_limits[ax] - shift))
						target[ax] = '1' if not opposite else '0'
					elif basis[ax] == -1:
						shift = 0 if not opposite else 1
						conditions.append('%s == %d' % (id2, shift))
						shift = 2 if not opposite else 1
						target[ax] = str(bnd_limits[ax] - shift)

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

			# In the local step of the AA access pattern, all distributions are
			# unpropagated, so every fluid node has to participate in the PBC
			# (but ghost nodes do not).
			if opposite:
				if dim == 2:
					return 'idx1 >= 1 && idx1 <= {0}'.format(max_dim)
				else:
					return 'idx1 >= 1 && idx2 >= 1 && idx1 <= {0} && idx2 <= {1}'.format(
						max_dim, max_dim2)
			# If the distributions are already propagated, then certain locations
			# could not have been propagated to. For instance, fNW for a node at
			# y == 1 should not take part in the PBC copy.
			else:
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

	%if node_addressing == 'indirect':
		const int gi_low_dense = gi_low;
		const int gi_high_dense = gi_high;
		gi_low = nodes[gi_low];
		gi_high = nodes[gi_high];
		if (gi_low != INVALID_NODE)
	%endif

	{
	// TODO(michalj): Generalize this for grids with e_i > 1.
	// Load distributions to be propagated from low idx to high idx.
	%for i in sym.get_prop_dists(grid, -1, axis):
		<% j = grid.idx_opposite[i] if opposite else i %>
		const float f${grid.idx_name[i]} = ${get_dist('dist', j, 'gi_low')};
	%endfor

	%for i in sym.get_prop_dists(grid, -1, axis):
		<% j = grid.idx_opposite[i] if opposite else i %>
		%if grid.basis[i].dot(grid.basis[i]) > 1:
			if (isfinite(f${grid.idx_name[i]})) {
				// Skip distributions which are not populated or cross multiple boundaries.
				if (${make_cond(i)}) {
					${get_dist('dist', j, 'gi_high')} = f${grid.idx_name[i]};
				}
				<%
					axis_target = bnd_limits[axis] - 2 if not opposite else bnd_limits[axis] - 1
					corner_cond, target = handle_corners(grid.basis[i], axis_target)
				%>
				%if corner_cond:
					%if not opposite:
						else
					%endif
					{
						if (0) {}
						%for cond, targ in zip(corner_cond, target):
							%if cond:
								else if (${cond}) {
									int gi_high2 = getGlobalIdx(${targ});
									%if node_addressing == 'indirect':
										gi_high2 = nodes[gi_high2];
										if (gi_high2 != INVALID_NODE)
									%endif
									{
										${get_dist('dist', j, 'gi_high2')} = f${grid.idx_name[i]};
									}
								}
							%endif
						%endfor
					}
				%endif
			}
		%else:
			if (gi_high != INVALID_NODE && isfinite(f${grid.idx_name[i]})) {
				${get_dist('dist', j, 'gi_high')} = f${grid.idx_name[i]};
			}
		%endif
	%endfor
	}  // low to high


	%if node_addressing == 'indirect':
		<% _offset = 0 %>
		gi_high = nodes[gi_high_dense + ${offset}];
		gi_low = nodes[gi_low_dense + ${offset}];
		if (gi_high != INVALID_NODE)
	%else:
		<% _offset = offset %>
	%endif
	{
	// Load distributrions to be propagated from high idx to low idx.
	%for i in sym.get_prop_dists(grid, 1, axis):
		<% j = grid.idx_opposite[i] if opposite else i %>
		const float f${grid.idx_name[i]} = ${get_dist('dist', j, 'gi_high', _offset)};
	%endfor

	%for i in sym.get_prop_dists(grid, 1, axis):
		<% j = grid.idx_opposite[i] if opposite else i %>
		%if grid.basis[i].dot(grid.basis[i]) > 1:
			if (isfinite(f${grid.idx_name[i]})) {
				// Skip distributions which are not populated or cross multiple boundaries.
				if (${make_cond(i)} ${' && gi_low != INVALID_NODE' if node_addressing == 'indirect' else ''}) {
					${get_dist('dist', j, 'gi_low', _offset)} = f${grid.idx_name[i]};
				}
				<%
					axis_target = 1 if not opposite else 0
					corner_cond, target = handle_corners(grid.basis[i], axis_target)
				%>
				%if corner_cond:
					${'else' if not opposite else ''}
					{
						if (0) {}
						%for cond, targ in zip(corner_cond, target):
							%if cond:
								else if (${cond}) {
									int gi_low2 = getGlobalIdx(${targ});
									%if node_addressing == 'indirect':
										gi_low2 = nodes[gi_low2];
										if (gi_low2 != INVALID_NODE)
									%endif
									{ ${get_dist('dist', j, 'gi_low2')} = f${grid.idx_name[i]}; }
								}
							%endif
						%endfor
					}
				%endif
			}
		%else:
			if (isfinite(f${grid.idx_name[i]}) && gi_low != INVALID_NODE) {
				${get_dist('dist', j, 'gi_low', _offset)} = f${grid.idx_name[i]};
			}
		%endif
	%endfor
	}  // high to low
</%def>

// Applies periodic boundary conditions within a single subdomain.
//  dist: pointer to the distributions array
//  axis: along which axis the PBCs are to be applied (0:x, 1:y, 2:z)
${kernel} void ApplyPeriodicBoundaryConditions(
	${nodes_array_if_required()}
	${global_ptr} float *dist, int axis)
{
	const int idx1 = get_global_id(0);
	int gi_low, gi_high;

	// For single block PBC, the envelope size (width of the ghost node
	// layer) is always 1.
	// TODO(michalj): Generalize this for the case when envelope_size != 1.
	%if dim == 2:
		if (axis == 0) {
			if (idx1 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(0, idx1);				// ghost node
			gi_high = getGlobalIdx(${lat_nx-2}, idx1);	// real node
			${pbc_helper(0, lat_ny-2)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx}) { return; }
			gi_low = getGlobalIdx(idx1, 0);				// ghost node
			gi_high = getGlobalIdx(idx1, ${lat_ny-2});	// real node
			${pbc_helper(1, lat_nx-2)}
		}
	%else:
		const int idx2 = get_global_id(1);
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

// Like ApplyPeriodicBoundaryConditions above, but works after the fully local
// step of the AA access pattern. The differences are:
//  - distributions opposite to normal ones are copied
//  - data is copied from real nodes to ghost nodes
${kernel} void ApplyPeriodicBoundaryConditionsWithSwap(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int axis)
{
	const int idx1 = get_global_id(0);
	int gi_low, gi_high;

	// For single block PBC, the envelope size (width of the ghost node
	// layer) is always 1.
	// TODO(michalj): Generalize this for the case when envelope_size != 1.
	%if dim == 2:
		if (axis == 0) {
			if (idx1 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(1, idx1);				// real node
			gi_high = getGlobalIdx(${lat_nx-1}, idx1);	// ghost node
			${pbc_helper(0, lat_ny-2, opposite=True)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx}) { return; }
			gi_low = getGlobalIdx(idx1, 1);				// real node
			gi_high = getGlobalIdx(idx1, ${lat_ny-1});	// ghost node
			${pbc_helper(1, lat_nx-2, opposite=True)}
		}
	%else:
		const int idx2 = get_global_id(1);
		if (axis == 0) {
			if (idx1 >= ${lat_ny-1} || idx2 >= ${lat_nz-1} || idx1 < 1 || idx2 < 1) { return; }
			gi_low = getGlobalIdx(1, idx1, idx2);				// real node
			gi_high = getGlobalIdx(${lat_nx-1}, idx1, idx2);	// ghost node
			${pbc_helper(0, lat_ny-2, lat_nz-2, opposite=True)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx-1} || idx2 >= ${lat_nz-1} || idx1 < 1 || idx2 < 1) { return; }
			gi_low = getGlobalIdx(idx1, 1, idx2);				// real node
			gi_high = getGlobalIdx(idx1, ${lat_ny-1}, idx2);	// ghost node
			${pbc_helper(1, lat_nx-2, lat_nz-2, opposite=True)}
		} else {
			if (idx1 >= ${lat_nx-1} || idx2 >= ${lat_ny-1} || idx1 < 1 || idx2 < 1) { return; }
			gi_low = getGlobalIdx(idx1, idx2, 1);				// real node
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-1});	// ghost node
			${pbc_helper(2, lat_nx-2, lat_ny-2, opposite=True)}
		}
	%endif
}


<%def name="_copy_field_if_finite(src, dest)">
	%if node_addressing == 'indirect':
		<% h = '%s != INVALID_NODE && %s != INVALID_NODE && ' % (src, dest) %>
	%else:
		<% h = '' %>
	%endif


	if (${h} isfinite(field[${src}])) {
		field[${dest}] = field[${src}];
	}
</%def>

<%def name="_indirect_index_pbc()">
	%if node_addressing == 'indirect':
		gi_low = nodes[gi_low];
		gi_high = nodes[gi_high];
	%endif
</%def>

// Applies periodic boundary conditions to a scalar field within a single subdomain.
//  dist: pointer to the array with the field data
//  axis: along which axis the PBCs are to be applied (0:x, 1:y, 2:z)
${kernel} void ApplyMacroPeriodicBoundaryConditions(
		${nodes_array_if_required()}
		${global_ptr} float *field, int axis)
{
	const int idx1 = get_global_id(0);
	int gi_low, gi_high;

	// TODO(michalj): Generalize this for the case when envelope_size != 1.
	%if dim == 2:
		if (axis == 0) {
			if (idx1 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(0, idx1);					// ghost node
			gi_high = getGlobalIdx(${lat_nx-2}, idx1);		// real node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(1, idx1);					// real node
			gi_high = getGlobalIdx(${lat_nx-1}, idx1);		// ghost node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_low', 'gi_high')}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx}) { return; }
			gi_low = getGlobalIdx(idx1, 0);					// ghost node
			gi_high = getGlobalIdx(idx1, ${lat_ny-2});		// real node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(idx1, 1);					// real node
			gi_high = getGlobalIdx(idx1, ${lat_ny-1});		// ghost node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_low', 'gi_high')}
		}
	%else:
		const int idx2 = get_global_id(1);
		if (axis == 0) {
			if (idx1 >= ${lat_ny} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(0, idx1, idx2);				// ghost node
			gi_high = getGlobalIdx(${lat_nx-2}, idx1, idx2);	// real node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(1, idx1, idx2);				// real node
			gi_high = getGlobalIdx(${lat_nx-1}, idx1, idx2);	// ghost node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_low', 'gi_high')}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(idx1, 0, idx2);				// ghost node
			gi_high = getGlobalIdx(idx1, ${lat_ny-2}, idx2);	// real node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(idx1, 1, idx2);				// real node
			gi_high = getGlobalIdx(idx1, ${lat_ny-1}, idx2);	// ghost node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_low', 'gi_high')}
		} else {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(idx1, idx2, 0);				// ghost node
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-2});	// real node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_high', 'gi_low')}
			gi_low = getGlobalIdx(idx1, idx2, 1);				// real node
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-1});	// ghost node
			${_indirect_index_pbc()}
			${_copy_field_if_finite('gi_low', 'gi_high')}
		}
	%endif
}

%if dim == 2:
<%def name="collect_continuous_data_body_2d(opposite=False)">
	const int idx = get_global_id(0);
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
				if opposite:
					gy = lat_linear_macro[axis]
				else:
					gy = lat_linear[axis]
			%>
			const int dist_size = max_lx / ${len(dists)};
			const int dist_num = idx / dist_size;
			const int gx = idx % dist_size;
			int gi = getGlobalIdx(base_gx + gx, ${gy});
			${indirect_index(orig=None, position_warning=False)}

			switch (dist_num) {
				%for i, prop_dist in enumerate(dists):
				case ${i}: {
					%if opposite:
						tmp = ${get_dist('dist', grid.idx_opposite[prop_dist], 'gi')};
					%else:
						tmp = ${get_dist('dist', prop_dist, 'gi')};
					%endif
					break;
				}
				%endfor
			}
			buffer[idx] = tmp;
			break;
		}
	%endfor
	}
</%def>

// Collects ghost node data for connections along axes other than X.
// dist: distributions array
// base_gy: where along the X axis to start collecting the data
// face: see LBBlock class constants
// buffer: buffer where the data is to be saved
${kernel} void CollectContinuousData(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx,
		int max_lx, ${global_ptr} float *buffer)
{
	${collect_continuous_data_body_2d(False)}
}

// As above, used for the AA access pattern.
${kernel} void CollectContinuousDataWithSwap(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx,
		int max_lx, ${global_ptr} float *buffer)
{
	${collect_continuous_data_body_2d(True)}
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
	${indirect_index(orig=None, position_warning=False)}
</%def>

## In the case of in-place propagation, the data is read from the same place as
## in the case of macroscopic variables.
<%def name="_get_global_idx_opp(axis)">
	## Y-axis
	%if axis < 4:
		gi = getGlobalIdx(base_gx + gx, ${lat_linear_macro[axis]}, base_other + other);
	## Z-axis
	%else:
		gi = getGlobalIdx(base_gx + gx, base_other + other, ${lat_linear_macro[axis]});
	%endif
	${indirect_index(orig=None, position_warning=False)}
</%def>

<%def name="collect_continuous_data_body_3d(opposite=False)">
	const int gx = get_global_id(0);
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
					%if opposite:
						${_get_global_idx_opp(axis)};
						tmp = ${get_dist('dist', grid.idx_opposite[prop_dist], 'gi')};
					%else:
						${_get_global_idx(axis)};
						tmp = ${get_dist('dist', prop_dist, 'gi')};
					%endif
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
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	${collect_continuous_data_body_3d(False)};
}

${kernel} void CollectContinuousDataWithSwap(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	${collect_continuous_data_body_3d(True)};
}
%endif

%if dim == 2:
<%def name="distribute_continuous_data_body_2d(opposite)">
	const int idx = get_global_id(0);

	if (idx >= max_lx) {
		return;
	}

	switch (face) {
	%for axis in range(2, 2*dim):
		case ${axis}: {
			<%
				normal = block.face_to_normal(axis)
				dists = sym.get_interblock_dists(grid, normal)
				if opposite:
					gy = lat_linear_with_swap[axis]
				else:
					gy = lat_linear_dist[axis]
			%>
			const int dist_size = max_lx / ${len(dists)};
			const int dist_num = idx / dist_size;
			const int gx = idx % dist_size;
			const float tmp = buffer[idx];
			int gi = getGlobalIdx(base_gx + gx, ${gy});
			${indirect_index(orig=None, position_warning=False)}
			switch (dist_num) {
				%for i, prop_dist in enumerate(dists):
				case ${i}: {
					%if opposite:
						${get_dist('dist', grid.idx_opposite[prop_dist], 'gi')} = tmp;
					%else:
						${get_dist('dist', prop_dist, 'gi')} = tmp;
					%endif
					break;
				}
				%endfor
			}

			break;
		}
	%endfor
	}
</%def>

${kernel} void DistributeContinuousData(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx,
		int max_lx, ${global_ptr} float *buffer)
{
	${distribute_continuous_data_body_2d(False)}
}

${kernel} void DistributeContinuousDataWithSwap(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx,
		int max_lx, ${global_ptr} float *buffer)
{
	${distribute_continuous_data_body_2d(True)}
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
	${indirect_index(orig=None, position_warning=False)}
</%def>

<%def name="_get_global_dist_idx_opp(axis)">
	## Y-axis
	%if axis < 4:
		gi = getGlobalIdx(base_gx + gx, ${lat_linear_with_swap[axis]}, base_other + other);
	## Z-axis
	%else:
		gi = getGlobalIdx(base_gx + gx, base_other + other, ${lat_linear_with_swap[axis]});
	%endif
	${indirect_index(orig=None, position_warning=False)}
</%def>

<%def name="distribute_continuous_data_body_3d(opposite)">
	const int gx = get_global_id(0);
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
			const int dist_size = max_other / ${len(dists)};
			const int dist_num = idx / dist_size;
			const int other = idx % dist_size;
			idx = (dist_size * max_lx * dist_num) + (other * max_lx) + gx;
			const float tmp = buffer[idx];

			switch (dist_num) {
				%for i, prop_dist in enumerate(dists):
				case ${i}: {
					%if opposite:
						${_get_global_dist_idx_opp(axis)}
						${get_dist('dist', grid.idx_opposite[prop_dist], 'gi')} = tmp;
					%else:
						${_get_global_dist_idx(axis)}
						${get_dist('dist', prop_dist, 'gi')} = tmp;
					%endif
					break;
				}
				%endfor
			}
			break;
		}
	%endfor
	}
</%def>

// Layout of the data in the buffer is the same as in the output buffer of
// CollectContinuousData.
${kernel} void DistributeContinuousData(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	${distribute_continuous_data_body_3d(False)}
}

${kernel} void DistributeContinuousDataWithSwap(
		${nodes_array_if_required()}
		${global_ptr} float *dist, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	${distribute_continuous_data_body_3d(True)}
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
	if (gi == INVALID_NODE) return;
	if (gi >= DIST_SIZE * ${grid.Q} || gi < 0) {
		printf("invalid node index detected in sparse coll %d (%d, %d)\n", gi, get_global_id(0), get_global_id(1));
		return;
	}
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
	if (gi == INVALID_NODE) return;
	if (gi >= DIST_SIZE * ${grid.Q} || gi < 0) {
		printf("invalid node index detected in sparse dist %d (%d, %d)\n", gi, get_global_id(0), get_global_id(1));
		return;
	}

	dist[gi] = buffer[idx];
}

%if dim == 2:
${kernel} void CollectContinuousMacroData(
		${nodes_array_if_required()}
		${global_ptr} float *field, int base_gx, int max_lx, int gy,
		${global_ptr} float *buffer)
{
	const int idx = get_global_id(0);
	if (idx >= max_lx) {
		return;
	}

	int gi = getGlobalIdx(base_gx + idx, gy);
	${indirect_index(orig=None, position_warning=False)}
	buffer[idx] = field[gi];
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
	${indirect_index(orig=None, position_warning=False)}
</%def>

// The data is collected from a rectangular area of the plane corresponding to 'face'.
// The grid with which the kernel is to be called has the following dimensions:
//
//  x: # nodes along the X direction + any padding (real # nodes is identified by max_lx)
//  y: # nodes along the Y/Z direction
//
${kernel} void CollectContinuousMacroData(
		${nodes_array_if_required()}
		${global_ptr} float *field, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	const int gx = get_global_id(0);
	const int other = get_global_id(1);

	if (gx >= max_lx || other >= max_other) {
		return;
	}

	// Index in the output buffer.
	const int idx = other * get_global_size(0) + gx;

	switch (face) {
	%for axis in range(2, 2 * dim):
		case ${axis}: {
			int gi;
			${_get_global_macro_idx(axis)};
			buffer[idx] = field[gi];
			break;
		}
	%endfor
	}
}
%endif  ## dim == 3

%if dim == 2:
${kernel} void DistributeContinuousMacroData(
		${nodes_array_if_required()}
		${global_ptr} float *field, int base_gx, int max_lx, int gy,
		${global_ptr} float *buffer)
{
	const int idx = get_global_id(0);
	if (idx >= max_lx) {
		return;
	}

	int gi = getGlobalIdx(base_gx + idx, gy);
	${indirect_index(orig=None, position_warning=False)}
	field[gi] = buffer[idx];
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
	${indirect_index(orig=None, position_warning=False)}
</%def>

${kernel} void DistributeContinuousMacroData(
		${nodes_array_if_required()}
		${global_ptr} float *field, int face, int base_gx, int base_other,
		int max_lx, int max_other, ${global_ptr} float *buffer)
{
	const int gx = get_global_id(0);
	const int other = get_global_id(1);

	if (gx >= max_lx || other >= max_other) {
		return;
	}

	// Index in the input buffer.
	const int idx = other * get_global_size(0) + gx;
	const float tmp = buffer[idx];

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

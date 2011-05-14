<%!
    from sailfish import sym
%>

%if 'gravity' in context.keys():
	${const_var} float gravity = ${gravity}f;
%endif

<%def name="bgk_args_decl()">
	float rho, float *iv0
	%if simtype == 'shan-chen':
		, float *ea0
	%endif
</%def>

<%def name="bgk_args()">
	g0m0, v
	%if simtype == 'shan-chen':
		, sca0
	%endif
</%def>

%if subgrid != 'les-smagorinsky':
	${const_var} float tau0 = ${tau}f;		// relaxation time
%endif
${const_var} float visc = ${visc}f;		// viscosity

<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
${kernel_common.nonlocal_fields_decl()}
${kernel_common.body(bgk_args_decl)}

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>
<%namespace file="propagation.mako" import="*"/>

<%include file="tracers.mako"/>

<%def name="init_dist_with_eq()">
	%for local_var in bgk_equilibrium_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
	%endfor

	%for i, (feq, idx) in enumerate(bgk_equilibrium[0]):
		${get_odist('dist1_in', i)} = ${cex(feq, vectors=True)};
	%endfor
</%def>

%if dim == 2:
${kernel} void SetLocalVelocity(
	${global_ptr} float *dist1_in,
	${global_ptr} float *irho,
	${kernel_args_1st_moment('ov')}
	int x, int y, float vx, float vy)
{
	int gx = x + get_global_id(0) - get_local_size(1) / 2;
	int gy = y + get_global_id(1) - get_local_size(1) / 2;

	${wrap_coords()}

	int gi = gx + ${arr_nx}*gy;
	float rho = irho[gi];
	float v0[${dim}];

	v0[0] = vx;
	v0[1] = vy;

	${init_dist_with_eq()}

	ovx[gi] = vx;
	ovy[gi] = vy;
}
%endif

// A kernel to set the node distributions using the equilibrium distributions
// and the macroscopic fields.
${kernel} void SetInitialConditions(
	${global_ptr} float *dist1_in,
	${kernel_args_1st_moment('iv')}
	${global_ptr} float *irho)
{
	${local_indices()}

	// Cache macroscopic fields in local variables.
	float rho = irho[gi];
	float v0[${dim}];

	v0[0] = ivx[gi];
	v0[1] = ivy[gi];
	%if dim == 3:
		v0[2] = ivz[gi];
	%endif

	${init_dist_with_eq()}
}

${kernel} void PrepareMacroFields(
	${global_ptr} int *map,
	${global_ptr} float *dist1_in,
	${global_ptr} float *orho)
{
	${local_indices()}

	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type) || isGhostNode(type))
		return;

	int orientation = decodeNodeOrientation(ncode);

	Dist fi;
	float out;

	getDist(&fi, dist1_in, gi);
	get0thMoment(&fi, type, orientation, &out);
	orho[gi] = out;
}

${kernel} void CollideAndPropagate(
	${global_ptr} int *map,
	${global_ptr} float *dist_in,
	${global_ptr} float *dist_out,
	${global_ptr} float *orho,
	${kernel_args_1st_moment('ov')}
	int save_macro
%if simtype == 'shan-chen':
	,${global_ptr} float *gg0m0
%endif
	)
{
	${local_indices()}

	// Shared variables for in-block propagation
	%for i in sym.get_prop_dists(grid, 1):
		${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
	%endfor
	%for i in sym.get_prop_dists(grid, 1):
		#define prop_${grid.idx_name[grid.idx_opposite[i]]} prop_${grid.idx_name[i]}
	%endfor

	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type) || isGhostNode(type))
		return;

	int orientation = decodeNodeOrientation(ncode);

	// Cache the distributions in local variables
	Dist d0;
	getDist(&d0, dist_in, gi);

	%if simtype == 'shan-chen':
		${sc_calculate_accel()}
	%endif

	// Macroscopic quantities for the current cell
	float g0m0, v[${dim}];

	%if simtype == 'shan-chen':
		${sc_macro_fields()}
	%else:
		getMacro(&d0, ncode, type, orientation, &g0m0, v);
	%endif

	precollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v);
	${relaxate(bgk_args)}
	postcollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v, gi, dist_out);

	// only save the macroscopic quantities if requested to do so
	if (save_macro == 1) {
		orho[gi] = g0m0;
		ovx[gi] = v[0];
		ovy[gi] = v[1];
		%if dim == 3:
			ovz[gi] = v[2];
		%endif
	}

	${propagate('dist_out', 'd0')}
}

<%def name="pbc_helper(axis)">
	<%
		if axis == 0:
			offset = 1
		elif axis == 1:
			offset = arr_nx
		else:
			offset = arr_nx + arr_ny
	%>

	// TODO(michalj): Generalize this for grids with e_i > 1.
	// From low idx to high idx.
	%for i in sym.get_prop_dists(grid, -1, axis):
		float f${grid.idx_name[i]} = ${get_dist('dist', i, 'gi_low')};
	%endfor

	%for i in sym.get_prop_dists(grid, -1, axis):
		${get_dist('dist', i, 'gi_high')} = f${grid.idx_name[i]};
	%endfor

	// From high idx to low idx.
	%for i in sym.get_prop_dists(grid, 1, axis):
		float f${grid.idx_name[i]} = ${get_dist('dist', i, 'gi_high', offset)};
	%endfor

	%for i in sym.get_prop_dists(grid, 1, axis):
		${get_dist('dist', i, 'gi_low', offset)} = f${grid.idx_name[i]};
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
	// layer is alawys 1.
	%if dim == 2:
		if (axis == 0) {
			if (idx1 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(0, idx1);
			gi_high = getGlobalIdx(${lat_nx-2}, idx1);
			${pbc_helper(0)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx}) { return; }
			gi_low = getGlobalIdx(idx1, 0);
			gi_high = getGlobalIdx(idx1, ${lat_ny-2});
			${pbc_helper(1)}
		}
	%else:
		int idx2 = get_global_id(1);
		if (axis == 0) {
			if (idx1 >= ${lat_ny} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(0, idx1, idx2);
			gi_high = getGlobalIdx(${lat_nx-2}, idx1, idx2);
			${pbc_helper(0)}
		} else if (axis == 1) {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_nz}) { return; }
			gi_low = getGlobalIdx(idx1, 0, idx2);
			gi_high = getGlobalIdx(idx1, ${lat_ny-2}, idx2);
			${pbc_helper(1)}
		} else {
			if (idx1 >= ${lat_nx} || idx2 >= ${lat_ny}) { return; }
			gi_low = getGlobalIdx(idx1, idx2, 0);
			gi_high = getGlobalIdx(idx1, idx2, ${lat_nz-2});
			${pbc_helper(2)}
		}
	%endif
}

## TODO(michalj): Extend this for 3D.
// Collects ghost node data for connections along axes other than X.
// dist: distributions array
// base_gy: where along the X axis to start collecting the data
// axis_dir: see LBBlock class constants
// buffer: buffer where the data is to be saved
${kernel} void CollectOrthogonalGhostData(
		${global_ptr} float *dist, int base_gx,
		int axis_dir, int max_idx, ${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	int gi;
	float tmp;

	if (idx >= max_idx) {
		return;
	}

	switch (axis_dir) {
	%for axis in range(2, 2*dim):
		case ${axis}: {
			<%
				prop_dists = sym.get_prop_dists(grid,
						block.axis_dir_to_dir(axis),
						block.axis_dir_to_axis(axis))
			%>
			int dist_size = get_global_size(0) / ${len(prop_dists)};
			int dist_num = idx / dist_size;
			int gx = idx % dist_size;
			switch (dist_num) {
				%for prop_dist in prop_dists:
				case ${prop_dist}: {
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

${kernel} void DistributeOrthogonalGhostData(
		${global_ptr} float *dist, int base_gx,
		int axis_dir, int max_idx, ${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	int gi;
	float tmp = buffer[idx];

	if (idx >= max_idx) {
		return;
	}

	switch (axis_dir) {
	%for axis in range(2, 2*dim):
		case ${axis}: {
			<%
				prop_dists = sym.get_prop_dists(grid,
						block.axis_dir_to_dir(axis),
						block.axis_dir_to_axis(axis))
			%>
			int dist_size = get_global_size(0) / ${len(prop_dists)};
			int dist_num = idx / dist_size;
			int gx = idx % dist_size;
			switch (dist_num) {
				%for prop_dist in prop_dists:
				case ${prop_dist}: {
					gi = getGlobalIdx(base_gx + gx, ${lat_linear[axis]});
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

${kernel} void CollectXGhostData(
		${global_ptr} int *idx_array, ${global_ptr} float *dist,
		${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	if (idx > ${distrib_collect_x_size-1}) {
		return;
	}
	int gi = idx_array[idx];
	buffer[idx] = dist[gi];
}

${kernel} void DistributeXGhostData(
		${global_ptr} int *idx_array, ${global_ptr} float *dist,
		${global_ptr} float *buffer)
{
	int idx = get_global_id(0);
	if (idx > ${distrib_collect_x_size-1}) {
		return;
	}
	int gi = idx_array[idx];
	dist[gi] = buffer[idx];
}

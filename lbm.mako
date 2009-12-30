<%!
    import sym
%>

#define BLOCK_SIZE ${block_size}
#define DIST_SIZE ${dist_size}
#define GEO_FLUID ${geo_fluid}
#define GEO_BCV ${geo_bcv}
#define GEO_BCP ${geo_bcp}

#define DT 1.0f

${const_var} float tau = ${tau}f;		// relaxation time
${const_var} float visc = ${visc}f;		// viscosity
${const_var} float geo_params[${num_params+1}] = {
% for param in geo_params:
	${param}f,
% endfor
0};		// geometry parameters

<%include file="opencl_compat.mako"/>
<%include file="geo_helpers.mako"/>

<%def name="barrier()">
	%if backend == 'cuda':
		__syncthreads();
	%else:
		barrier(CLK_LOCAL_MEM_FENCE);
	%endif
</%def>

<%def name="zouhe_bb(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_bb(grid, orientation):
			${sym.use_pointers(str(arg))} = ${sym.use_pointers(str(val))};
		%endfor
		break;
</%def>

<%def name="zouhe_fixup(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_fixup(grid, orientation):
			${str(arg)} = ${str(val)};
		%endfor
		break;
</%def>


<%def name="zouhe_velocity(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_velocity(grid, orientation):
			${sym.use_pointers(str(arg))} = ${sym.use_pointers(str(val))};
		%endfor
		break;
</%def>

<%def name="get_boundary_velocity(node_type, mx, my, mz, rho=0, moments=False)">
	int idx = (${node_type} - GEO_BCV) * ${dim};
	%if moments:
		${mx} = geo_params[idx] * ${rho};
		${my} = geo_params[idx+1] * ${rho};
		%if dim == 3:
			${mz} = geo_params[idx+2] * ${rho};
		%endif
	%else:
		${mx} = geo_params[idx];
		${my} = geo_params[idx+1];
		%if dim == 3:
			${mz} = geo_params[idx+2];
		%endif
	%endif
</%def>

<%def name="get_boundary_pressure(node_type, rho)">
	int idx = (GEO_BCP-GEO_BCV) * ${dim} + (${node_type} - GEO_BCP);
	${rho} = geo_params[idx] * 3.0f;
</%def>

<%def name="get_boundary_params(node_type, mx, my, mz, rho, moments=False)">
	if (${node_type} >= GEO_BCV) {
		// Velocity boundary condition.
		if (${node_type} < GEO_BCP) {
			${get_boundary_velocity(node_type, mx, my, mz, rho, moments)}
		// Pressure boundary condition.
		} else {
			${get_boundary_pressure(node_type, rho)}
		}
	}
</%def>

<%def name="fill_missing_distributions()">
	switch (orientation) {
		%for i in range(0, grid.Q-1):
			case ${i}: {
				%for lvalue, rvalue in sym.fill_missing_dists(grid, 'fi', missing_dir=i):
					${lvalue} = ${rvalue};
				%endfor
				break;
			}
		%endfor
	}
</%def>

<%def name="external_force(node_type, vx, vy, vz, rho=0, momentum=False)">
	%if ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0:
		if (!isWallNode(${node_type})) {
			%if momentum:
				%if ext_accel_x != 0.0:
					${vx} += ${rho} * ${'%.20ff' % (0.5 * ext_accel_x)};
				%endif
				%if ext_accel_y != 0.0:
					${vy} += ${rho} * ${'%.20ff' % (0.5 * ext_accel_y)};
				%endif
				%if dim == 3 and ext_accel_z != 0.0:
					${vz} += ${rho} * ${'%.20ff' % (0.5 * ext_accel_z)};
				%endif
			%else:
				%if ext_accel_x != 0.0:
					${vx} += ${'%.20ff' % (0.5 * ext_accel_x)};
				%endif
				%if ext_accel_y != 0.0:
					${vy} += ${'%.20ff' % (0.5 * ext_accel_y)};
				%endif
				%if dim == 3 and ext_accel_z != 0.0:
					${vz} += ${'%.20ff' % (0.5 * ext_accel_z)};
				%endif
			%endif
		}
	%endif
</%def>

${device_func} inline void bounce_back(Dist *fi)
{
	float t;

	%for i in sym.bb_swap_pairs(grid):
		t = fi->${grid.idx_name[i]};
		fi->${grid.idx_name[i]} = fi->${grid.idx_name[grid.idx_opposite[i]]};
		fi->${grid.idx_name[grid.idx_opposite[i]]} = t;
	%endfor
}

${device_func} inline void compute_macro_quant(Dist *fi, float *rho, float *v)
{
	*rho = ${sym.ex_rho(grid, 'fi')};
	%for d in range(0, grid.dim):
		v[${d}] = ${str(sym.ex_velocity(grid, 'fi', d, '*rho')).replace('/*', '/ *')};
	%endfor
}

%if bc_wall == 'zouhe' or bc_velocity == 'zouhe' or bc_pressure == 'zouhe':
${device_func} void zouhe_bb(Dist *fi, int orientation, float *rho, float *v)
{
	// Bounce-back of the non-equilibrium parts.
	switch (orientation) {
		%for i in range(0, len(grid.basis)-1):
			${zouhe_bb(i)}
		%endfor
		case ${geo_dir_other}:
			bounce_back(fi);
			return;
	}

	float nvx, nvy;
	%if dim == 3:
		float nvz;
	%endif

	// Compute new macroscopic variables.
	nvx = ${str(sym.ex_velocity(grid, 'fi', 0, 'nrho', momentum=True)).replace('/*', '/ *')};
	nvy = ${str(sym.ex_velocity(grid, 'fi', 1, 'nrho', momentum=True)).replace('/*', '/ *')};
	%if dim == 3:
		nvz = ${str(sym.ex_velocity(grid, 'fi', 2, 'nrho', momentum=True)).replace('/*', '/ *')};
	%endif

	// Compute momentum difference.
	nvx = *rho * v[0] - nvx;
	nvy = *rho * v[1] - nvy;
	%if dim == 3:
		nvz = *rho * v[2] - nvz;
	%endif

	switch (orientation) {
		%for i in range(0, len(grid.basis)-1):
			${zouhe_fixup(i)}
		%endfor
	}
}
%endif

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// the node class node_type.
//
${device_func} inline void getMacro(Dist *fi, int node_type, int orientation, float *rho, float *v)
{
	#define vx v[0]
	#define vy v[1]
	#define vz v[2]

	if (isFluidOrWallNode(node_type) || orientation == ${geo_dir_other}) {
		compute_macro_quant(fi, rho, v);
	} else if (isVelocityNode(node_type)) {
		// We're dealing with a boundary node, for which some of the distributions
		// might be meaningless.  Fill them with the values of the opposite
		// distributions.
		%if bc_velocity != 'fullbb' and bc_velocity != None:
			${fill_missing_distributions()}
			*rho = ${sym.ex_rho(grid, 'fi')};
			${get_boundary_velocity('node_type', 'v[0]', 'v[1]', 'v[2]')}

			switch (orientation) {
				%for i in range(0, grid.Q-1):
					case ${i}:
						*rho = ${sym.ex_rho(grid, 'fi', missing_dir=i, rho='*rho')};
						break;
				%endfor
			}
		%else:
			compute_macro_quant(fi, rho, v);
		%endif
	} else if (isPressureNode(node_type)) {
		%if bc_pressure != 'fullbb' and bc_pressure != None:
			${fill_missing_distributions()}
			*rho = ${sym.ex_rho(grid, 'fi')};
			float par_rho;
			${get_boundary_pressure('node_type', 'par_rho')}

			switch (orientation) {
				%for i in range(0, grid.Q-1):
					case ${i}: {
						%for d in range(0, grid.dim):
							v[${d}] = ${str(sym.ex_velocity(grid, 'fi', d, '*rho', missing_dir=i, par_rho='par_rho')).replace('/*', '/ *')};
						%endfor
						break;
					 }
				%endfor
			}

			*rho = par_rho;
		%else:
			compute_macro_quant(fi, rho, v);
		%endif
	}

	#undef vx
	#undef vy
	#undef vz

	%if bc_wall == 'zouhe':
		if (isWallNode(node_type)) {
			v[0] = 0.0f;
			v[1] = 0.0f;
			%if dim == 3:
				v[2] = 0.0f;
			%endif
			zouhe_bb(fi, orientation, rho, v);
		}
	%endif

	%if bc_velocity == 'zouhe':
		if (isVelocityNode(node_type)) {
			zouhe_bb(fi, orientation, rho, v);
		}
	%endif

	%if bc_pressure == 'zouhe':
		if (isPressureNode(node_type)) {
			zouhe_bb(fi, orientation, rho, v);
		}
	%endif

	${external_force('node_type', 'v[0]', 'v[1]', 'v[2]')}
}

${device_func} inline void boundaryConditions(Dist *fi, int node_type, int orientation, float *rho, float *v)
{
	%if bc_wall == 'fullbb':
		if (isWallNode(node_type)) {
			bounce_back(fi);
		}
	%endif

	#define vx v[0]
	#define vy v[1]
	#define vz v[2]

	%if bc_velocity == 'fullbb':
		if (isVelocityNode(node_type)) {
			bounce_back(fi);
			${get_boundary_velocity('node_type', 'v[0]', 'v[1]', 'v[2]')}
			%for i, ve in enumerate(grid.basis):
				// * *rho for compressible
				fi->${grid.idx_name[i]} += 1.0f * ${sym.make_float(2.0 * grid.weights[i] * grid.v.dot(ve) / grid.cssq)};
			%endfor
			*rho = ${sym.ex_rho(grid, 'fi')};
		}
	%endif

	%if bc_velocity == 'equilibrium':
		if (isVelocityNode(node_type)) {
			%for feq, idx in sym.bgk_equilibrium(grid):
				fi->${idx} = ${feq};
			%endfor
		}
	%endif

	%if bc_pressure == 'equilibrium':
		if (isPressureNode(node_type)) {
			%for feq, idx in sym.bgk_equilibrium(grid):
				fi->${idx} = ${feq};
			%endfor
		}
	%endif

	#undef vx
	#undef vy
	#undef vz
}

//
// A kernel to update the position of tracer particles.
//
// Each thread updates the position of a single particle using Euler's algorithm.
//
${kernel} void LBMUpdateTracerParticles(${global_ptr} float *dist, ${global_ptr} int *map,
		${global_ptr} float *x, ${global_ptr} float *y \
%if dim == 3:
	, ${global_ptr} float *z \
%endif
		)
{
	float rho, v[${dim}];

	int gi = get_global_id(0);
	float cx = x[gi];
	float cy = y[gi];

	int ix = (int)(cx);
	int iy = (int)(cy);

	%if dim == 3:
		float cz = z[gi];
		int iz = (int)(cz);

		if (iz < 0)
			iz  = 0;

		if (iz > ${lat_d-1})
			iz = ${lat_d-1};
	%endif

	// Sanity checks.
	if (iy < 0)
		iy = 0;

	if (ix < 0)
		ix = 0;

	if (ix > ${lat_w-1})
		ix = ${lat_w-1};

	if (iy > ${lat_h-1})
		iy = ${lat_h-1};

	%if dim == 2:
		int idx = ix + ${lat_w}*iy;
	%else:
		int idx = ix + ${lat_w}*iy + ${lat_w*lat_h}*iz;
	%endif

	Dist fc;

## HACK: If a call to getDist() is made below, the overall performance of the simulation
## will be decreased by a factor of 2, regardless of whether this kernel is even executed.
## This might be caused by the NVIDIA OpenCL compiler not inlining the getDist function.
## To avoid the performance loss, we temporarily inline getDist manually.
	// getDist(&fc, dist, idx);

	%for i, dname in enumerate(grid.idx_name):
		fc.${dname} = dist[idx + DIST_SIZE*${i}];
	%endfor

	int type, orientation;
	decodeNodeType(map[idx], &orientation, &type);
	getMacro(&fc, type, orientation, &rho, v);

	cx = cx + v[0] * DT;
	cy = cy + v[1] * DT;
	%if dim == 3:
		cz = cz + v[2] * DT;
	%endif

	// Periodic boundary conditions.
	if (cx > ${lat_w})
		cx = 0.0f;

	if (cy > ${lat_h})
		cy = 0.0f;

	if (cx < 0.0f)
		cx = (float)${lat_w};

	if (cy < 0.0f)
		cy = (float)${lat_h};

	%if dim == 3:
		if (cz > ${lat_d})
			cz = 0.0f;

		if (cz < 0.0f)
			cz = (float)(${lat_d});

		z[gi] = cz;
	%endif
	x[gi] = cx;
	y[gi] = cy;
}

% if model == 'mrt':
//
// Relaxation in moment space.
//
${device_func} void MS_relaxate(Dist *fi, int node_type)
{
	DistM fm, feq;

	%for mrt, val in sym.bgk_to_mrt(grid, 'fi', 'fm'):
		${mrt} = ${val};
	%endfor

	${external_force('node_type', 'fm.mx', 'fm.my', 'fm.mz', 'fm.rho', momentum=True)}

	#define mx fm.mx
	#define my fm.my
	#define mz fm.mz
	#define rho fm.rho

	// Calculate equilibrium distributions in moment space.
	%for i, eq in enumerate(grid.mrt_equilibrium):
		%if eq != 0:
			feq.${grid.mrt_names[i]} = ${eq};
		%endif
	%endfor

	// Relexate the non-conserved moments,
	%if bc_velocity == 'equilibrium':
		if (isVelocityNode(node_type)) {
			%for i, coll in enumerate(grid.mrt_collision):
				%if coll != 0:
					fm.${grid.mrt_names[i]} = feq.${grid.mrt_names[i]};
				%endif
			%endfor
		}
	%endif

	%if bc_pressure == 'equilibrium':
		if (isPressureNode(node_type)) {
			%for i, coll in enumerate(grid.mrt_collision):
				%if coll != 0:
					fm.${grid.mrt_names[i]} = feq.${grid.mrt_names[i]};
				%endif
			%endfor
		}
	%endif

	%for i, name in enumerate(grid.mrt_names):
		%if grid.mrt_collision[i] != 0:
			fm.${name} -= ${sym.make_float(grid.mrt_collision[i])} * (fm.${name} - feq.${name});
		%endif
	%endfor

	#undef mx
	#undef my
	#undef mz
	#undef rho

	${external_force('node_type', 'fm.mx', 'fm.my', 'fm.mz', 'fm.rho', momentum=True)}

	%for bgk, val in sym.mrt_to_bgk(grid, 'fi', 'fm'):
		${bgk} = ${val};
	%endfor
}
% endif	## model == mrt

% if model == 'bgk':
//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
${device_func} void BGK_relaxate(float rho, float *v, Dist *fi, int node_type)
{
	Dist feq;

	#define vx v[0]
	#define vy v[1]
	#define vz v[2]

	%for feq, idx in sym.bgk_equilibrium(grid):
		feq.${idx} = ${feq};
	%endfor

	%for idx in grid.idx_name:
		fi->${idx} += (feq.${idx} - fi->${idx}) / tau;
	%endfor

	%if ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0:
		if (!isWallNode(node_type))
		{
			// External acceleration.
			#define eax ${'%.20ff' % ext_accel_x}
			#define eay ${'%.20ff' % ext_accel_y}
			#define eaz ${'%.20ff' % ext_accel_z}
			float pref = ${sym.bgk_external_force_pref()};

			%for val, idx in sym.bgk_external_force(grid):
				fi->${idx} += ${val};
			%endfor
		}
	%endif

	#undef vx
	#undef vy
	#undef vz
}
%endif

<%def name="_relaxate()">
	%if model == 'bgk':
		BGK_relaxate(rho, v, &fi, type);
	%else:
		MS_relaxate(&fi, type);
	%endif
</%def>

## TODO: This could be optimized.
<%def name="relaxate()">
	if (isFluidNode(type)) {
		${_relaxate()}
	}
	%if bc_wall_.wet_nodes:
		else if (isWallNode(type)) {
			${_relaxate()}
		}
	%endif
	%if bc_velocity_.wet_nodes:
		else if (isVelocityNode(type)) {
			${_relaxate()}
		}
	%endif
	%if bc_pressure_.wet_nodes:
		else if (isPressureNode(type)) {
			${_relaxate()}
		}
	%endif
</%def>

<%def name="prop_bnd(effective_dir, i, di, local, offset)">
## Generate the propagation code for a specific base direction.
##
## This is a generic function which should work for any dimensionality and grid
## type.
##
## Args:
##   effective_dir: X propagation direction (1 for East, -1 for West)
##   offset: target offset in the distribution array
##   i: index of the base vector along which to propagate
##   di: dimension index

	## This is the final dimension, generate the actual propagation code.
	%if di == dim:
		%if dim == 2:
			${set_odist(i, effective_dir, grid.basis[i][1], 0, offset, local)}
		%else:
			${set_odist(i, effective_dir, grid.basis[i][1], grid.basis[i][2], offset, local)}
		%endif
	## Make a recursive call to prop_bnd to process the remaining dimensions.
	## The recursive calls are done to generate checks for out-of-domain
	## propagation.
	%else:
		## Make sure we're not propagating outside of the simulation domain.
		%if grid.basis[i][di] > 0:
			if (${loc_names[di]} < ${bnd_limits[di]-1}) { \
		%elif grid.basis[i][di] < 0:
			if (${loc_names[di]} > 0) { \
		%endif
			## Recursive call for the next dimension.
			${prop_bnd(effective_dir, i, di+1, local, offset)}
		%if grid.basis[i][di] != 0:
			} \
		%endif

		## In case we are about to propagate outside of the simulation domain,
		## check for periodic boundary conditions for the current dimension.
		## If they are enabled, update the offset by a value precomputed in
		## pbc_offsets and proceed to the following dimension.
		%if periodicity[di] and grid.basis[i][di] != 0:
			else {
				${prop_bnd(effective_dir, i, di+1, local, offset+pbc_offsets[di][int(grid.basis[i][di])])}
			}
		%endif
	%endif
</%def>

## Propagate eastwards or westwards knowing that there is an east/westward
## node layer to propagate to.
<%def name="prop_block_bnd(dir, dist_source, offset=0)">
## Generate the propagation code for all directions with a X component.  The X component
## is special as shared-memory propogation is done in the X direction.
##
## Args:
##   dir: X propagation direction (1 for East, -1 for West, 0 for orthogonal to X axis)
##
	%for i in sym.get_prop_dists(grid, dir):
		%if dist_source == 'prop_local':
			${prop_bnd(0, i, 1, True, offset)}
		%else:
			${prop_bnd(dir, i, 1, False, offset)}
		%endif
	%endfor
</%def>

<%def name="set_odist(idir, xoff, yoff, zoff, offset, local)">
<%
	def rel_offset(x, y, z):
		if grid.dim == 2:
			return x + y * lat_w
		else:
			return x + lat_w * (y + lat_h*z)
%>
	%if local:
		dist_out[gi + ${dist_size*idir + rel_offset(xoff, yoff, zoff) + offset}] = prop_${grid.idx_name[idir]}[lx];
	%else:
		dist_out[gi + ${dist_size*idir + rel_offset(xoff, yoff, zoff) + offset}] = fi.${grid.idx_name[idir]};
	%endif
</%def>

${kernel} void LBMCollideAndPropagate(${global_ptr} int *map, ${global_ptr} float *dist_in,
		${global_ptr} float *dist_out, ${global_ptr} float *orho, ${global_ptr} float *ovx,
		${global_ptr} float *ovy, \
%if dim == 3:
		${global_ptr} float *ovz, \
%endif
		int save_macro)
{
	int lx = get_local_id(0);	// ID inside the current block
	%if dim == 2:
		int gx = get_global_id(0);
		int gy = get_group_id(1);
		int gi = gx + ${lat_w}*gy;
	%else:
		// This is a workaround for the limitations of current CUDA devices.
		// We would like the grid to be 3 dimensional, but only 2 dimensions
		// are supported.  We thus encode the first two dimensions (x, y) of
		// the simulation grid into the x dimension of the CUDA/OpenCL grid
		// as:
		//   x_dev = y * num_blocks + x.
		//
		// This works fine, as x is relatively small, since:
		//   x = x_sim / block_size.
		int gx = get_global_id(0) % ${lat_w};
		int gy = get_global_id(0) / ${lat_w};
		int gz = get_global_id(1);
		int gi = gx + gy*${lat_w} + ${lat_w*lat_h}*gz;
	%endif

	// shared variables for in-block propagation
	%for i in sym.get_prop_dists(grid, 1):
		${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
	%endfor
	%for i in sym.get_prop_dists(grid, -1):
		${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
	%endfor

	// cache the distributions in local variables
	Dist fi;
	getDist(&fi, dist_in, gi);

	int type, orientation;
	decodeNodeType(map[gi], &orientation, &type);

	// macroscopic quantities for the current cell
	float rho, v[${dim}];

	getMacro(&fi, type, orientation, &rho, v);
	boundaryConditions(&fi, type, orientation, &rho, v);
	${barrier()}

	// only save the macroscopic quantities if requested to do so
	if (save_macro == 1) {
		orho[gi] = rho;
		ovx[gi] = v[0];
		ovy[gi] = v[1];
		%if dim == 3:
			ovz[gi] = v[2];
		%endif
	}

	${relaxate()}

	// update the 0-th direction distribution
	dist_out[gi] = fi.fC;

	// E propagation in shared memory
	if (lx < ${block_size-1}) {
		%for i in sym.get_prop_dists(grid, 1):
			prop_${grid.idx_name[i]}[lx+1] = fi.${grid.idx_name[i]};
		%endfor
	// E propagation in global memory (at right block boundary)
	} else if (gx < ${lat_w-1}) {
		${prop_block_bnd(1, 'prop_global')}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(1, 'prop_global', pbc_offsets[0][1])}
	}
	%endif

	// W propagation in shared memory
	if (lx > 0) {
		%for i in sym.get_prop_dists(grid, -1):
			prop_${grid.idx_name[i]}[lx-1] = fi.${grid.idx_name[i]};
		%endfor
	// W propagation in global memory (at left block boundary)
	} else if (gx > 0) {
		${prop_block_bnd(-1, 'prop_global')}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(-1, 'prop_global', pbc_offsets[0][-1])}
	}
	%endif

	${barrier()}

	// Save locally propagated distributions into global memory.
	// The leftmost thread is not updated in this block.
	if (lx > 0) {
		${prop_block_bnd(1, 'prop_local')}
	}

	// Propagation in directions orthogonal to the X axis (global memory)
	${prop_block_bnd(0, 'prop_global')}

	// The rightmost thread is not updated in this block.
	if (lx < ${block_size-1}) {
		${prop_block_bnd(-1, 'prop_local')}
	}
}


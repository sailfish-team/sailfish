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

typedef struct Dist {
	float \
	%for i, dname in enumerate(sym.GRID.idx_name):
${dname}\
		%if i < len(sym.GRID.idx_name)-1:
, \
		%endif
	%endfor
	;
} Dist;

%if model == 'mrt':
// Distribution in momentum space.
typedef struct DistM {
	float \
	%for i, dname in enumerate(sym.GRID.mrt_names):
${dname}\
		%if i < len(sym.GRID.mrt_names)-1:
, \
		%endif
	%endfor
	;
} DistM;
%endif

<%include file="opencl_compat.mako"/>

//
// Copy the idx-th distribution from din into dout.
//
${device_func} inline void getDist(Dist *dout, ${global_ptr} float *din, int idx)
{
	%for i, dname in enumerate(sym.GRID.idx_name):
		dout->${dname} = din[idx + DIST_SIZE*${i}];
	%endfor
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

<%def name="zouhe_bb(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_bb(orientation):
			${sym.use_pointers(str(arg))} = ${sym.use_pointers(str(val))};
		%endfor
		break;
</%def>

<%def name="zouhe_fixup(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_fixup(orientation):
			${str(arg)} = ${str(val)};
		%endfor
		break;
</%def>


<%def name="zouhe_velocity(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_velocity(orientation):
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

	%for i in sym.bb_swap_pairs():
		t = fi->${sym.GRID.idx_name[i]};
		fi->${sym.GRID.idx_name[i]} = fi->${sym.GRID.idx_name[sym.GRID.idx_opposite[i]]};
		fi->${sym.GRID.idx_name[sym.GRID.idx_opposite[i]]} = t;
	%endfor
}

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
		*rho = ${sym.ex_rho('fi')};
		v[0] = ${str(sym.ex_velocity('fi', 0, '*rho')).replace('/*', '/ *')};
		v[1] = ${str(sym.ex_velocity('fi', 1, '*rho')).replace('/*', '/ *')};
		%if dim == 3:
			v[2] = ${str(sym.ex_velocity('fi', 2, '*rho')).replace('/*', '/ *')};
		%endif
	} else {
		// We're dealing with a boundary node, for which some of the distributions
		// might be meaningless.
		if (isVelocityNode(node_type)) {
			${get_boundary_velocity('node_type', 'v[0]', 'v[1]', 'v[2]')}
			switch (orientation) {
				%for i in range(0, len(sym.GRID.basis)-1):
					case ${i}:
					{
						*rho = ${sym.ex_rho('fi', missing_dir=i)};
						break;
					}
				%endfor
			}
		} else {
			${get_boundary_pressure('node_type', '*rho')}
			switch (orientation) {
				%for i in range(0, len(sym.GRID.basis)-1):
					case ${i}:
					{
						v[0] = ${str(sym.ex_velocity('fi', 0, '*rho', missing_dir=i)).replace('/*', '/ *')};
						v[1] = ${str(sym.ex_velocity('fi', 1, '*rho', missing_dir=i)).replace('/*', '/ *')};
						%if dim == 3:
							v[2] = ${str(sym.ex_velocity('fi', 2, '*rho', missing_dir=i)).replace('/*', '/ *')};
						%endif
						break;
					 }
				%endfor
			}
		}
	}

	#undef vx
	#undef vy
	#undef vz

	%if boundary_type == 'zouhe':
		if (isWallNode(node_type)) {
			v[0] = 0.0f;
			v[1] = 0.0f;
			%if dim == 3:
				v[2] = 0.0f;
			%endif
		}

		// Bounce-back of the non-equilibrium parts.
		switch (orientation) {
			${zouhe_bb(geo_wall_n)}
			${zouhe_bb(geo_wall_s)}
			${zouhe_bb(geo_wall_e)}
			${zouhe_bb(geo_wall_w)}
			${zouhe_bb(geo_wall_ne)}
			${zouhe_bb(geo_wall_nw)}
			${zouhe_bb(geo_wall_se)}
			${zouhe_bb(geo_wall_sw)}
			case ${geo_dir_other}:
				bounce_back(fi);
				return;
		}

		float nvx, nvy;
		%if dim == 3:
			float nvz;
		%endif

		// Compute new macroscopic variables.
		nvx = ${str(sym.ex_velocity('fi', 0, 'nrho', momentum=True)).replace('/*', '/ *')};
		nvy = ${str(sym.ex_velocity('fi', 1, 'nrho', momentum=True)).replace('/*', '/ *')};
		%if dim == 3:
			nvz = ${str(sym.ex_velocity('fi', 2, 'nrho', momentum=True)).replace('/*', '/ *')};
		%endif

		// Compute momentum difference.
		nvx = *rho * v[0] - nvx;
		nvy = *rho * v[1] - nvy;
		%if dim == 3:
			nvz = *rho * v[2] - nvz;
		%endif

		switch (orientation) {
			${zouhe_fixup(geo_wall_n)}
			${zouhe_fixup(geo_wall_s)}
			${zouhe_fixup(geo_wall_e)}
			${zouhe_fixup(geo_wall_w)}
			${zouhe_fixup(geo_wall_ne)}
			${zouhe_fixup(geo_wall_nw)}
			${zouhe_fixup(geo_wall_se)}
			${zouhe_fixup(geo_wall_sw)}
		}

	%endif

	${external_force('node_type', 'v[0]', 'v[1]', 'v[2]')}
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

	%for i, dname in enumerate(sym.GRID.idx_name):
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

	%for mrt, val in sym.bgk_to_mrt('fi', 'fm'):
		${mrt} = ${val};
	%endfor

	${get_boundary_params('node_type', 'fm.mx', 'fm.my', 'fm.mz', 'fm.rho', True)}
	${external_force('node_type', 'fm.mx', 'fm.my', 'fm.mz', 'fm.rho', momentum=True)}

	#define mx fm.mx
	#define my fm.my
	#define mz fm.mz
	#define rho fm.rho

	// Calculate equilibrium distributions in moment space.
	%for i, eq in enumerate(sym.GRID.mrt_equilibrium):
		%if eq != 0:
			feq.${sym.GRID.mrt_names[i]} = ${eq};
		%endif
	%endfor

	// Relexate the non-conserved moments,
	%if boundary_type == 'fullbb' or boundary_type == 'halfbb':
		if (isVelocityOrPressureNode(node_type)) {
			%for i, coll in enumerate(sym.GRID.mrt_collision):
				%if coll != 0:
					fm.${sym.GRID.mrt_names[i]} = feq.${sym.GRID.mrt_names[i]};
				%endif
			%endfor
		} else
	%endif
	{
		%for i, name in enumerate(sym.GRID.mrt_names):
			%if sym.GRID.mrt_collision[i] != 0:
				fm.${name} -= ${sym.make_float(sym.GRID.mrt_collision[i])} * (fm.${name} - feq.${name});
			%endif
		%endfor
	}

	#undef mx
	#undef my
	#undef mz
	#undef rho

	${external_force('node_type', 'fm.mx', 'fm.my', 'fm.mz', 'fm.rho', momentum=True)}

	%for bgk, val in sym.mrt_to_bgk('fi', 'fm'):
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

	%for feq, idx in sym.bgk_equilibrium():
		feq.${idx} = ${feq};
	%endfor

	%for idx in sym.GRID.idx_name:
		fi->${idx} += (feq.${idx} - fi->${idx}) / tau;
	%endfor

	%if boundary_type == 'fullbb' or boundary_type == 'halfbb':
		// XXX: For some weird reason, putting the above relaxation code inside
		// the else clause below breaks when double precision is used with CUDA 2.3.
		// To avoid the problem, run it unconditionally above.
		if (isVelocityOrPressureNode(node_type)) {
			%for idx in sym.GRID.idx_name:
				fi->${idx} = feq.${idx};
			%endfor
		}
	%endif

	%if ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0:
		%if boundary_type == 'fullbb':
			if (!isWallNode(node_type))
		%endif
		{
			// External acceleration.
			#define eax ${'%.20ff' % ext_accel_x}
			#define eay ${'%.20ff' % ext_accel_y}
			#define eaz ${'%.20ff' % ext_accel_z}
			float pref = ${sym.bgk_external_force_pref()};

			%for val, idx in sym.bgk_external_force():
				fi->${idx} += ${val};
			%endfor
		}
	%endif

	#undef vx
	#undef vy
	#undef vz
}
%endif

<%def name="relaxate()">
	% if model == 'bgk':
		BGK_relaxate(rho, v, &fi, type);
	% else:
		MS_relaxate(&fi, type);
	% endif
</%def>

/*
FIXME: Temporarily disable this until it is converted into a grid-independent form.

${device_func} inline void half_bb(Dist *fi, const int node_type)
{
	// TODO: add support for corners
	switch (node_type) {
	case GEO_WALL_E:
		fi->fNE = fi->fSW;
		fi->fSE = fi->fNW;
		fi->fE = fi->fW;
		break;

	case GEO_WALL_W:
		fi->fNW = fi->fSE;
		fi->fSW = fi->fNE;
		fi->fW = fi->fE;
		break;

	case GEO_WALL_S:
		fi->fSE = fi->fNW;
		fi->fSW = fi->fNE;
		fi->fS = fi->fN;
		break;

	case GEO_WALL_N:
		fi->fNE = fi->fSW;
		fi->fNW = fi->fSE;
		fi->fN = fi->fS;
		break;
	}
}
*/

<%def name="prop_bnd(dir, effective_dir, i, di, dname, dist_source, offset)">
	%if di == dim:
		%if dim == 2:
			set_odist(gi+rel(${effective_dir},${sym.GRID.basis[i][1]},0)+${offset}, ${dname}, ${dist_source}(${dname}));
		%else:
			set_odist(gi+rel(${effective_dir},${sym.GRID.basis[i][1]},${sym.GRID.basis[i][2]})+${offset}, ${dname}, ${dist_source}(${dname}));
		%endif
	%else:
		%if sym.GRID.basis[i][di] > 0:
			if (${loc_names[di]} < ${bnd_limits[di]-1}) { \
		%elif sym.GRID.basis[i][di] < 0:
			if (${loc_names[di]} > 0) { \
		%endif
			${prop_bnd(dir, effective_dir, i, di+1, dname, dist_source, offset)}
		%if sym.GRID.basis[i][di] != 0:
			} \
		%endif
		%if periodicity[di] and sym.GRID.basis[i][di] != 0:
			else {
				${prop_bnd(dir, effective_dir, i, di+1, dname, dist_source, offset+pbc_offsets[di][int(sym.GRID.basis[i][di])])}
			}
		%endif
	%endif
</%def>

## Propagate eastwards or westwards knowing that there is an east/westward
## node layer to propagate to.
<%def name="prop_block_bnd(dir, effective_dir, res, dist_source, offset=0)">
	%for i, dname in sym.get_prop_dists(dir, res):
		${prop_bnd(dir, effective_dir, i, 1, dname, dist_source, offset)}
	%endfor
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
	%for i, dir in sym.get_prop_dists(1):
		${shared_var} float prop_${dir}[BLOCK_SIZE];
	%endfor
	%for i ,dir in sym.get_prop_dists(-1):
		${shared_var} float prop_${dir}[BLOCK_SIZE];
	%endfor

	// cache the distributions in local variables
	Dist fi;
	getDist(&fi, dist_in, gi);

	int type, orientation;
	decodeNodeType(map[gi], &orientation, &type);

	% if boundary_type == 'fullbb':
		if (isWallNode(type)) {
			bounce_back(&fi);
		}
	% elif boundary_type == 'halfbb':
		if (isWallNode(type)) {
			half_bb(&fi, type);
		}
	% endif

	// macroscopic quantities for the current cell
	float rho, v[${dim}];

	// In the MRT model, there is no need to calculate the macroscopic
	// variables unless we want to save them as output.
	%if model == 'mrt':
		if (save_macro == 1)
	%endif
	getMacro(&fi, type, orientation, &rho, v);

	// only save the macroscopic quantities if requested to do so
	if (save_macro == 1) {
		orho[gi] = rho;
		ovx[gi] = v[0];
		ovy[gi] = v[1];
		%if dim == 3:
			ovz[gi] = v[2];
		%endif
	}

	% if boundary_type == 'fullbb':
		if (!isWallNode(type)) {
			${relaxate()}
		}
	% else:
		${relaxate()}
	% endif

	%for i, dname in enumerate(sym.GRID.idx_name):
		#define dir_${dname} ${i}
	%endfor

	#define dir_idx(idx) dir_##idx
	#define set_odist(idx, dir, val) dist_out[DIST_SIZE*dir_idx(dir) + idx] = val

	#define prop_global(dir) fi.dir
	#define prop_local(dir) prop_##dir[lx]

	%if dim == 2:
		#define rel(x,y,z) ((x) + ${lat_w}*(y))
	%else:
		#define rel(x,y,z) ((x) + ${lat_w}*(y) + ${lat_w*lat_h}*(z))
	%endif

	// update the 0-th direction distribution
	set_odist(gi, fC, fi.fC);

	// E propagation in shared memory
	if (lx < ${block_size-1}) {
		%for i, dir in sym.get_prop_dists(1):
			prop_${dir}[lx+1] = fi.${dir};
		%endfor
	// E propagation in global memory (at right block boundary)
	} else if (gx < ${lat_w-1}) {
		${prop_block_bnd(1, 1, 1, 'prop_global')}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(1, 1, 1, 'prop_global', pbc_offsets[0][1])}
	}
	%endif

	// W propagation in shared memory
	if (lx > 0) {
		%for i, dir in sym.get_prop_dists(-1):
			prop_${dir}[lx-1] = fi.${dir};
		%endfor
	// W propagation in global memory (at left block boundary)
	} else if (gx > 0) {
		${prop_block_bnd(-1, -1, 1, 'prop_global')}
	}
	%if periodic_x:
	// periodic boundary conditions in the X direction
	else {
		${prop_block_bnd(-1, -1, 1, 'prop_global', pbc_offsets[0][-1])}
	}
	%endif

% if backend == 'cuda':
	__syncthreads();
% else:
	barrier(CLK_LOCAL_MEM_FENCE);
% endif

	// Save locally propagated distributions into global memory.
	// The leftmost thread is not updated in this block
	if (lx > 0) {
		${prop_block_bnd(1, 0, 1, 'prop_local')}
	}

	// N, S propagation (global memory)
	${prop_block_bnd(1, 0, 0, 'prop_global')}

	// the rightmost thread is not updated in this block
	if (lx < ${block_size-1}) {
		${prop_block_bnd(-1, 0, 1, 'prop_local')}
	}
}


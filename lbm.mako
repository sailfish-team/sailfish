<%!
    import sym
%>

#define BLOCK_SIZE ${block_size}
#define DIST_SIZE ${dist_size}
#define GEO_FLUID ${geo_fluid}
#define GEO_WALL_E ${geo_wall_e}
#define GEO_WALL_W ${geo_wall_w}
#define GEO_WALL_S ${geo_wall_s}
#define GEO_WALL_N ${geo_wall_n}
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
	%for i, dname in enumerate(sym.idx_name):
${dname}\
		%if i < len(sym.idx_name)-1:
, \
		%endif
	%endfor
	;
} Dist;

// Distribution in momentum space.
typedef struct DistM {
	float rho, en, ens, mx, ex, my, ey, sd, sod;
} DistM;

<%include file="opencl_compat.mako"/>

//
// Copy the idx-th distribution from din into dout.
//
${device_func} inline void getDist(Dist *dout, ${global_ptr} float *din, int idx)
{
	%for i, dname in enumerate(sym.idx_name):
		dout->${dname} = din[idx + DIST_SIZE*${i}];
	%endfor
}

${device_func} inline bool isWallNode(int type) {
	return type == ${geo_wall};
}

// This assumes we're dealing with a wall node.
${device_func} inline bool isVelocityNode(int type) {
	return (type >= ${geo_bcv}) && (type < GEO_BCP);
}

${device_func} inline void decodeNodeType(int nodetype, int *orientation, int *type) {
	*orientation = nodetype & ${geo_orientation_mask};
	*type = nodetype >> ${geo_orientation_shift};
}

<%def name="zouhe_velocity(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_velocity(orientation):
			${sym.use_pointers(str(arg))} = ${sym.use_pointers(str(val))};
		%endfor
		break;
</%def>

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// the node class node_type.
//
${device_func} inline void getMacro(Dist *fi, int node_type, int orientation, float *rho, float *vx, float *vy)
{
	% if boundary_type == 'zouhe':
		if (isWallNode(node_type) || isVelocityNode(node_type)) {
			if (node_type > ${geo_wall}) {
				int idx = (node_type - GEO_BCV) * 2;
				*vx = geo_params[idx];
				*vy = geo_params[idx+1];
			} else {
				*vx = 0.0f;
				*vy = 0.0f;
			}

			switch (orientation) {
			${zouhe_velocity(geo_wall_n)}
			${zouhe_velocity(geo_wall_s)}
			${zouhe_velocity(geo_wall_e)}
			${zouhe_velocity(geo_wall_w)}

			case ${geo_wall_ne}:
				*rho = (2.0f * (fi->fW + fi->fS + fi->fSW) + fi->fC + fi->fNW + fi->fSE) / (1.0f - 11.0f/12.0f*(*vx + *vy));
				fi->fNE = fi->fSW + 1.0f/12.0f * *rho * (*vx + *vy);
				fi->fE = *rho * (11.0f/12.0f * *vx - 1.0f/12.0f * *vy) - fi->fSE + fi->fW + fi->fNW;
				fi->fN = *rho * (-1.0f/12.0f * *vx + 11.0f/12.0f * *vy) -fi->fNW + fi->fS + fi->fSE;
				break;

			case ${geo_wall_se}:
				*rho = (2.0f * (fi->fN + fi->fW + fi->fNW) + fi->fC + fi->fSW + fi->fNE) / (1.0f - 11.0f/12.0f*(*vx - *vy));
				fi->fSE = fi->fNW + 1.0f/12.0f * *rho * (*vx - *vy);
				fi->fE = *rho * (11.0f/12.0f * *vx + 1.0f/12.0f * *vy) - fi->fNE + fi->fW + fi->fSW;
				fi->fS = *rho * (-1.0f/12.0f * *vx - 11.0f/12.0f * *vy) - fi->fSW + fi->fN + fi->fNE;
				break;

			case ${geo_wall_nw}:
				*rho = (2.0f * (fi->fE + fi->fS + fi->fSE) + fi->fC + fi->fSW + fi->fNE) / (1.0f - 11.0f/12.0f*(-*vx + *vy));
				fi->fNW = fi->fSE + 1.0f/12.0f * *rho * (-*vx + *vy);
				fi->fW = *rho * (-11.0f/12.0f * *vx - 1.0f/12.0f * *vy) - fi->fSW + fi->fE + fi->fNE;
				fi->fN = *rho * (1.0f/12.0f * *vx + 11.0f/12.0f * *vy) - fi->fNE + fi->fS + fi->fSW;
				break;

			case ${geo_wall_sw}:
				*rho = (2.0f * (fi->fE + fi->fN + fi->fNE) + fi->fC + fi->fSE + fi->fNW) / (1.0f + 11.0f/12.0f*(*vx + *vy));
				fi->fSW = fi->fNE + 1.0f/12.0f * *rho * -(*vx + *vy);
				fi->fW = *rho * (-11.0f/12.0f * *vx + 1.0f/12.0f * *vy) - fi->fNW + fi->fE + fi->fSE;
				fi->fS = *rho * (1.0f/12.0f * *vx - 11.0f/12.0f * *vy) - fi->fSE + fi->fN + fi->fNW;
				break;
			}

			return;
		}
	% endif

	*rho = ${sym.ex_rho('fi')};

	% if boundary_type != 'fullbb' and boundary_type != 'halfbb':
	if (node_type >= GEO_BCV) {
		// Velocity boundary condition.
		if (node_type < GEO_BCP) {
			int idx = (node_type - GEO_BCV) * 2;
			*vx = geo_params[idx];
			*vy = geo_params[idx+1];
			return;
		// Pressure boundary condition.
		} else {
			// c_s^2 = 1/3, P/c_s^2 = rho
			int idx = (GEO_BCP-GEO_BCV) * 2 + (node_type - GEO_BCP);
			*rho = geo_params[idx] * 3.0f;
		}
	}
	% endif

	*vx = ${str(sym.ex_velocity('fi', 0, '*rho')).replace('/*', '/ *')};
	*vy = ${str(sym.ex_velocity('fi', 1, '*rho')).replace('/*', '/ *')};

	if (!isWallNode(node_type)) {
		*vx += ${'%.20f' % (0.5 * ext_accel_x)};
		*vy += ${'%.20f' % (0.5 * ext_accel_y)};
	}
}

//
// A kernel to update the position of tracer particles.
//
// Each thread updates the position of a single particle using Euler's algorithm.
//
${kernel} void LBMUpdateTracerParticles(${global_ptr} float *dist, ${global_ptr} int *map,
		${global_ptr} float *x, ${global_ptr} float *y)
{
	float rho, vx, vy;

	int gi = get_global_id(0);
	float cx = x[gi];
	float cy = y[gi];

	int ix = (int)(cx);
	int iy = (int)(cy);

	// Sanity checks.
	if (iy < 0)
		iy = 0;

	if (ix < 0)
		ix = 0;

	if (ix > ${lat_w-1})
		ix = ${lat_w-1};

	if (iy > ${lat_h-1})
		iy = ${lat_h-1};

	int idx = ix + ${lat_w}*iy;

	Dist fc;

## HACK: If a call to getDist() is made below, the overall performance of the simulation
## will be decreased by a factor of 2, regardless of whether this kernel is even executed.
## This might be caused by the NVIDIA OpenCL compiler not inlining the getDist function.
## To avoid the performance loss, we temporarily inline getDist manually.
	// getDist(&fc, dist, idx);

	%for i, dname in enumerate(sym.idx_name):
		fc.${dname} = dist[idx + DIST_SIZE*${i}];
	%endfor

	int type, orientation;
	decodeNodeType(map[idx], &orientation, &type);
	getMacro(&fc, type, orientation, &rho, &vx, &vy);

	cx = cx + vx * DT;
	cy = cy + vy * DT;

	// Periodic boundary conditions.
	if (cx > ${lat_w})
		cx = 0.0f;

	if (cy > ${lat_h})
		cy = 0.0f;

	if (cx < 0.0f)
		cx = (float)${lat_w};

	if (cy < 0.0f)
		cy = (float)${lat_h};

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

	fm.rho = 1.0f*fi->fC + 1.0f*fi->fE + 1.0f*fi->fN + 1.0f*fi->fW + 1.0f*fi->fS + 1.0f*fi->fNE + 1.0f*fi->fNW + 1.0f*fi->fSW + 1.0f*fi->fSE;
	fm.en = -4.0f*fi->fC - 1.0f*fi->fE - 1.0f*fi->fN - 1.0f*fi->fW - 1.0f*fi->fS + 2.0f*fi->fNE + 2.0f*fi->fNW + 2.0f*fi->fSW + 2.0f*fi->fSE;
	fm.ens = 4.0f*fi->fC - 2.0f*fi->fE - 2.0f*fi->fN - 2.0f*fi->fW - 2.0f*fi->fS + 1.0f*fi->fNE + 1.0f*fi->fNW + 1.0f*fi->fSW + 1.0f*fi->fSE;
	fm.mx =  0.0f*fi->fC + 1.0f*fi->fE + 0.0f*fi->fN - 1.0f*fi->fW + 0.0f*fi->fS + 1.0f*fi->fNE - 1.0f*fi->fNW - 1.0f*fi->fSW + 1.0f*fi->fSE;
	fm.ex =  0.0f*fi->fC - 2.0f*fi->fE + 0.0f*fi->fN + 2.0f*fi->fW + 0.0f*fi->fS + 1.0f*fi->fNE - 1.0f*fi->fNW - 1.0f*fi->fSW + 1.0f*fi->fSE;
	fm.my =  0.0f*fi->fC + 0.0f*fi->fE + 1.0f*fi->fN + 0.0f*fi->fW - 1.0f*fi->fS + 1.0f*fi->fNE + 1.0f*fi->fNW - 1.0f*fi->fSW - 1.0f*fi->fSE;
	fm.ey =  0.0f*fi->fC + 0.0f*fi->fE - 2.0f*fi->fN + 0.0f*fi->fW + 2.0f*fi->fS + 1.0f*fi->fNE + 1.0f*fi->fNW - 1.0f*fi->fSW - 1.0f*fi->fSE;
	fm.sd =  0.0f*fi->fC + 1.0f*fi->fE - 1.0f*fi->fN + 1.0f*fi->fW - 1.0f*fi->fS + 0.0f*fi->fNE + 0.0f*fi->fNW + 0.0f*fi->fSW - 0.0f*fi->fSE;
	fm.sod = 0.0f*fi->fC + 0.0f*fi->fE + 0.0f*fi->fN + 0.0f*fi->fW + 0.0f*fi->fS + 1.0f*fi->fNE - 1.0f*fi->fNW + 1.0f*fi->fSW - 1.0f*fi->fSE;

	if (node_type >= GEO_BCV) {
		// Velocity boundary condition.
		if (node_type < GEO_BCP) {
			int idx = (node_type - GEO_BCV) * 2;
			fm.mx = geo_params[idx];
			fm.my = geo_params[idx+1];
		// Pressure boundary condition.
		} else {
			int idx = (GEO_BCP-GEO_BCV) * 2 + (node_type - GEO_BCP);
			fm.rho = geo_params[idx] * 3.0f;
		}
	}

	float h = fm.mx*fm.mx + fm.my*fm.my;
	feq.en  = -2.0f*fm.rho + 3.0f*h;
	feq.ens = fm.rho - 3.0f*h;
	feq.ex  = -fm.mx;
	feq.ey  = -fm.my;
	feq.sd  = (fm.mx*fm.mx - fm.my*fm.my);
	feq.sod = (fm.mx*fm.my);

	float tau7 = 4.0f / (12.0f*visc + 2.0f);
	float tau4 = 3.0f*(2.0f - tau7) / (3.0f - tau7);
	float tau8 = 1.0f/((2.0f/tau7 - 1.0f)*0.5f + 0.5f);

	if (node_type == GEO_FLUID || isWallNode(node_type)) {
		fm.en  -= 1.63f * (fm.en - feq.en);
		fm.ens -= 1.14f * (fm.ens - feq.ens);
		fm.ex  -= tau4 * (fm.ex - feq.ex);
		fm.ey  -= 1.92f * (fm.ey - feq.ey);
		fm.sd  -= tau7 * (fm.sd - feq.sd);
		fm.sod -= tau8 * (fm.sod - feq.sod);
	} else {
		fm.en  = feq.en;
		fm.ens = feq.ens;
		fm.ex  = feq.ex;
		fm.ey  = feq.ey;
		fm.sd  = feq.sd;
		fm.sod = feq.sod;
	}

	fi->fC  = (1.0f/9.0f)*fm.rho - (1.0f/9.0f)*fm.en + (1.0f/9.0f)*fm.ens;
	fi->fE  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens + (1.0f/6.0f)*fm.mx - (1.0f/6.0f)*fm.ex + 0.25f*fm.sd;
	fi->fN  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens + (1.0f/6.0f)*fm.my - (1.0f/6.0f)*fm.ey - 0.25f*fm.sd;
	fi->fW  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens - (1.0f/6.0f)*fm.mx + (1.0f/6.0f)*fm.ex + 0.25f*fm.sd;
	fi->fS  = (1.0f/9.0f)*fm.rho - (1.0f/36.0f)*fm.en - (1.0f/18.0f)*fm.ens - (1.0f/6.0f)*fm.my + (1.0f/6.0f)*fm.ey - 0.25f*fm.sd;
	fi->fNE = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 +(1.0f/6.0f)*fm.mx + (1.0f/12.0f)*fm.ex + (1.0f/6.0f)*fm.my + (1.0f/12.0f)*fm.ey + 0.25f*fm.sod;
	fi->fNW = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 -(1.0f/6.0f)*fm.mx - (1.0f/12.0f)*fm.ex + (1.0f/6.0f)*fm.my + (1.0f/12.0f)*fm.ey - 0.25f*fm.sod;
	fi->fSW = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 -(1.0f/6.0f)*fm.mx - (1.0f/12.0f)*fm.ex - (1.0f/6.0f)*fm.my - (1.0f/12.0f)*fm.ey + 0.25f*fm.sod;
	fi->fSE = (1.0f/9.0f)*fm.rho + (1.0f/18.0f)*fm.en + (1.0f/36.0f)*fm.ens +
			 +(1.0f/6.0f)*fm.mx + (1.0f/12.0f)*fm.ex - (1.0f/6.0f)*fm.my - (1.0f/12.0f)*fm.ey - 0.25f*fm.sod;
}
% endif

% if model == 'bgk':
//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
${device_func} void BGK_relaxate(float rho, float vx, float vy, Dist *fi, int node_type)
{
	Dist feq;

	%for feq, idx in sym.bgk_equilibrium():
		feq.${idx} = ${feq};
	%endfor

	%for idx in sym.idx_name:
		fi->${idx} += (feq.${idx} - fi->${idx}) / tau;
	%endfor

	%if ext_accel_x != 0.0 or ext_accel_y != 0.0:
		%if boundary_type == 'fullbb':
			if (!isWallNode(node_type))
		%endif
		{
			// External acceleration.
			#define eax ${'%.20ff' % ext_accel_x}
			#define eay ${'%.20ff' % ext_accel_y}
			float pref = ${sym.bgk_external_force_pref()};

			%for val, idx in sym.bgk_external_force():
				fi->${idx} += ${val};
			%endfor
		}
%endif
}
%endif

<%def name="relaxate()">
	% if model == 'bgk':
		BGK_relaxate(rho, vx, vy, &fi, type);
	% else:
		MS_relaxate(&fi, type, type);
	% endif
</%def>

${device_func} inline void bounce_back(Dist *fi)
{
	float t;

	%for i in sym.bb_swap_pairs():
		t = fi->${sym.idx_name[i]};
		fi->${sym.idx_name[i]} = fi->${sym.idx_name[sym.idx_opposite[i]]};
		fi->${sym.idx_name[sym.idx_opposite[i]]} = t;
	%endfor
}

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

${kernel} void LBMCollideAndPropagate(${global_ptr} int *map, ${global_ptr} float *dist_in,
		${global_ptr} float *dist_out, ${global_ptr} float *orho, ${global_ptr} float *ovx,
		${global_ptr} float *ovy, int save_macro)
{
	int tix = get_local_id(0);
	int ti = get_global_id(0);
	int gi = ti + ${lat_w}*get_group_id(1);

	// shared variables for in-block propagation
	${shared_var} float fo_E[BLOCK_SIZE];
	${shared_var} float fo_W[BLOCK_SIZE];
	${shared_var} float fo_SE[BLOCK_SIZE];
	${shared_var} float fo_SW[BLOCK_SIZE];
	${shared_var} float fo_NE[BLOCK_SIZE];
	${shared_var} float fo_NW[BLOCK_SIZE];

	// cache the distribution in local variables
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
	float rho, vx, vy;
	getMacro(&fi, type, orientation, &rho, &vx, &vy);

	// only save the macroscopic quantities if requested to do so
	if (save_macro == 1) {
		orho[gi] = rho;
		ovx[gi] = vx;
		ovy[gi] = vy;
	}

	% if boundary_type == 'fullbb':
		if (!isWallNode(type)) {
			${relaxate()}
		}
	% else:
		${relaxate()}
	% endif

	%for i, dname in enumerate(sym.idx_name):
		#define dir_${dname} ${i}
	%endfor

	#define dir_idx(idx) dir_##idx
	#define set_odist(idx, dir, val) dist_out[DIST_SIZE*dir_idx(dir) + idx] = val
	#define rel(x,y) ((x) + ${lat_w}*(y))

	// update the 0-th direction distribution
	set_odist(gi, fC, fi.fC);

	// E propagation in shared memory
	if (tix < get_local_size(0)-1) {
		fo_E[tix+1] = fi.fE;
		fo_NE[tix+1] = fi.fNE;
		fo_SE[tix+1] = fi.fSE;
	// E propagation in global memory (at right block boundary)
	} else if (ti < ${lat_w-1}) {
		set_odist(gi+rel(1,0), fE, fi.fE);
		if (get_group_id(1) > 0)	set_odist(gi+rel(1,-1), fSE, fi.fSE);
		%if periodic_y:
			else					set_odist(ti+${lat_w*(lat_h-1)+1}, fSE, fi.fSE);
		%endif
		if (get_group_id(1) < ${lat_h-1})	set_odist(gi+rel(1,1), fNE, fi.fNE);
		%if periodic_y:
			else					set_odist(ti+1, fNE, fi.fNE);
		%endif
	}
	%if periodic_x:
	else {
		set_odist(gi+rel(${-lat_w+1}, 0), fE, fi.fE);
		if (get_group_id(1) > 0)			set_odist(gi+rel(${-lat_w+1},-1), fSE, fi.fSE);
		%if periodic_y:
			else							set_odist(rel(0, ${lat_h-1}), fSE, fi.fSE);
		%endif
		if (get_group_id(1) < ${lat_h-1})	set_odist(gi+rel(${-lat_w+1},1), fNE, fi.fNE);
		%if periodic_y:
			else							set_odist(rel(0, 0), fNE, fi.fNE);
		%endif
	}
	%endif

	// W propagation in shared memory
	if (tix > 0) {
		fo_W[tix-1] = fi.fW;
		fo_NW[tix-1] = fi.fNW;
		fo_SW[tix-1] = fi.fSW;
	// W propagation in global memory (at left block boundary)
	} else if (ti > 0) {
		set_odist(gi+rel(-1,0), fW, fi.fW);
		if (get_group_id(1)	> 0)			set_odist(gi+rel(-1,-1), fSW, fi.fSW);
		%if periodic_y:
			else							set_odist(ti+${lat_w*(lat_h-1)-1}, fSW, fi.fSW);
		%endif
		if (get_group_id(1) < ${lat_h-1})	set_odist(gi+rel(-1,1), fNW, fi.fNW);
		%if periodic_y:
			else							set_odist(ti-1, fNW, fi.fNW);
		%endif
	}
	%if periodic_x:
	else {
		set_odist(gi+rel(${lat_w-1},0), fW, fi.fW);
		if (get_group_id(1) > 0)			set_odist(gi+rel(${lat_w-1},-1), fSW, fi.fSW);
		%if periodic_y:
			else							set_odist(${lat_h*lat_w-1}, fSW, fi.fSW);
		%endif
		if (get_group_id(1) < ${lat_h-1})	set_odist(gi+rel(${lat_w-1},1), fNW, fi.fNW);
		%if periodic_y:
			else							set_odist(${lat_w-1}, fNW, fi.fNW);
		%endif
	}
	%endif

% if backend == 'cuda':
	__syncthreads();
% else:
	barrier(CLK_LOCAL_MEM_FENCE);
% endif

	// the leftmost thread is not updated in this block
	if (tix > 0) {
		set_odist(gi, fE, fo_E[tix]);
		if (get_group_id(1) > 0)			set_odist(gi-${lat_w}, fSE, fo_SE[tix]);
		%if periodic_y:
			else							set_odist(gi+${lat_w*(lat_h-1)}, fSE, fo_SE[tix]);
		%endif
		if (get_group_id(1) < ${lat_h-1})	set_odist(gi+${lat_w}, fNE, fo_NE[tix]);
		%if periodic_y:
			else							set_odist(ti, fNE, fo_NE[tix]);
		%endif
	}

	// N + S propagation (global memory)
	if (get_group_id(1) > 0)			set_odist(gi-${lat_w}, fS, fi.fS);
	%if periodic_y:
		else							set_odist(ti+${lat_w*(lat_h-1)}, fS, fi.fS);
	%endif
	if (get_group_id(1) < ${lat_h-1})	set_odist(gi+${lat_w}, fN, fi.fN);
	%if periodic_y:
		else							set_odist(ti, fN, fi.fN);
	%endif

	// the rightmost thread is not updated in this block
	if (tix < get_local_size(0)-1) {
		set_odist(gi, fW, fo_W[tix]);
		if (get_group_id(1) > 0)			set_odist(gi-${lat_w}, fSW, fo_SW[tix]);
		%if periodic_y:
			else							set_odist(gi+${lat_w*(lat_h-1)}, fSW, fo_SW[tix]);
		%endif
		if (get_group_id(1) < ${lat_h-1})	set_odist(gi+${lat_w}, fNW, fo_NW[tix]);
		%if periodic_y:
			else							set_odist(ti, fNW, fo_NW[tix]);
		%endif
	}
}


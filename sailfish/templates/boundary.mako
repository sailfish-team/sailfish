<%!
    from sailfish import sym
%>

<%namespace file="code_common.mako" import="*"/>
<%namespace file="propagation.mako" import="rel_offset,get_odist"/>
<%namespace file="utils.mako" import="get_field_off,nonlocal_fld,fld_args"/>

<%def name="noneq_bb(orientation)">
	case ${orientation}:
		%for arg, val in sym.noneq_bb(grid, orientation):
			${cex(arg, pointers=True)} = ${cex(val, pointers=True)};
		%endfor
		break;
</%def>

<%def name="zouhe_fixup(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_fixup(grid, orientation):
			${str(arg)} = ${cex(val)};
		%endfor
		break;
</%def>

<%def name="get_boundary_velocity(node_param, mx, my, mz, rho=0, moments=False)">
	%if moments:
		${mx} = geo_params[${node_param} * ${dim}] * ${rho};
		${my} = geo_params[${node_param} * ${dim} + 1] * ${rho};
		%if dim == 3:
			${mz} = geo_params[${node_param} * ${dim} + 2] * ${rho};
		%endif
	%else:
		${mx} = geo_params[${node_param} * ${dim}];
		${my} = geo_params[${node_param} * ${dim} + 1];
		%if dim == 3:
			${mz} = geo_params[${node_param} * ${dim} + 2];
		%endif
	%endif
</%def>

<%def name="get_boundary_pressure(node_param, rho)">
	${rho} = geo_params[${geo_num_velocities * dim} + ${node_param}] * 3.0f;
</%def>

<%def name="fill_missing_distributions()">
	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			case ${i}: {
				%for lvalue, rvalue in sym.fill_missing_dists(grid, 'fi', missing_dir=i):
					${lvalue.var} = ${rvalue};
				%endfor
				break;
			}
		%endfor
	}
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

<%def name="sc_ppot_lin(comp)">
	${nonlocal_fld(comp)}
</%def>

<%def name="sc_ppot_exp(comp)">
	1.0f - exp(-${nonlocal_fld(comp)})
</%def>

<%def name="sc_ppot(comp)">
	<%
		local.template.get_def(sc_pseudopotential).render_context(context, comp)
	%>
</%def>

<%def name="sc_calculate_accel()">
##
## Declare and evaluate the Shan-Chen accelerations.
##
	%for x in set(sum(force_couplings.keys(), ())):
		float sca${x}[${dim}];
	%endfor

	if (!isWallNode(type)) {
		%for dists, coupling_const in force_couplings.iteritems():

			// Interaction between two components.
			%if dists[0] != dists[1]:
				%if dim == 2:
					shan_chen_accel(gi, gg${dists[0]}m0, gg${dists[1]}m0, ${dists[0]}, ${dists[1]},
						${coupling_const}, sca${dists[0]}, sca${dists[1]}, gx, gy);
				%else:
					shan_chen_accel(gi, gg${dists[0]}m0, gg${dists[1]}m0, ${dists[0]}, ${dists[1]},
						${coupling_const}, sca${dists[0]}, sca${dists[1]}, gx, gy, gz);
				%endif
			// Self-interaction of a singel component.
			%else:
				%if dim == 2:
					shan_chen_accel_self(gi, gg${dists[0]}m0, ${dists[0]}, ${coupling_const}, sca${dists[0]}, gx, gy);
				%else:
					shan_chen_accel_self(gi, gg${dists[0]}m0, ${dists[0]}, ${coupling_const}, sca${dists[0]}, gx, gy, gz);
				%endif
			%endif
		%endfor
	}
</%def>

<%def name="sc_macro_fields()">
##
## Calculate the density and velocity for the Shan-Chen coupled fields.
##
	float total_dens;

	%for i, x in enumerate(set(sum(force_couplings.keys(), ()))):
		get0thMoment(&d${x}, type, orientation, &g${x}m0);
		compute_1st_moment(&d${x}, v, ${i}, 1.0f/tau${x});
	%endfor

	total_dens = 0.0f;
	%for x in set(sum(force_couplings.keys(), ())):
		total_dens += g${x}m0 / tau${x};
	%endfor

	%for i in range(0, dim):
		%for x in set(sum(force_couplings.keys(), ())):
			sca${x}[${i}] /= g${x}m0;
		%endfor
		v[${i}] /= total_dens;
	%endfor
</%def>

%if simtype == 'shan-chen':

${device_func} inline float sc_ppot(${global_ptr} float *fx, int fi1, int idx, int nx, int ny
%if dim == 3:
, int nz
%endif
)
{
	if (0) { ; }

	%for fld_id in image_fields:
		else if (fi1 == ${fld_id}) {
			return ${sc_ppot(fld_id)};
		}
	%endfor
	else {
		return ${sc_ppot('x')};
	}

	return 0.0f;
}


${device_func} inline void shan_chen_accel_self(int i, ${global_ptr} float *f1, int fi1, float cc, float *a1, int x, int y
%if dim == 3:
	, int z
%endif
)
{
	float t1;

	%for i in range(0, dim):
		a1[${i}] = 0.0f;
	%endfor

	int off, nx, ny;
	%if dim == 3:
		int nz;
	%endif

	%for i, ve in enumerate(grid.basis):
		%if dim == 3:
			${get_field_off(ve[0], ve[1], ve[2])};
		%else:
			${get_field_off(ve[0], ve[1], 0)};
		%endif

		t1 = sc_ppot(f1, fi1, i + off, ${fld_args()});

		%if ve[0] != 0:
			a1[0] += t1 * ${ve[0] * grid.weights[i]};
		%endif
		%if ve[1] != 0:
			a1[1] += t1 * ${ve[1] * grid.weights[i]};
		%endif
		%if dim == 3 and ve[2] != 0:
			a1[2] += t1 * ${ve[2] * grid.weights[i]};
		%endif
	%endfor

	off = 0;
	nx = x;
	ny = y;
	%if dim == 3:
		nz = z;
	%endif

	t1 = sc_ppot(f1, fi1, i + off, ${fld_args()});

	%for i in range(0, dim):
		a1[${i}] *= t1 * cc;
	%endfor
}

${device_func} inline void shan_chen_accel(int i, ${global_ptr} float *f1, ${global_ptr} float *f2,
int fi1, int fi2, float cc, float *a1, float *a2, int x, int y
%if dim == 3:
	, int z
%endif
)
{
	float t1, t2;

	%for i in range(0, dim):
		a1[${i}] = 0.0f;
		a2[${i}] = 0.0f;
	%endfor

	int off, nx, ny;
	%if dim == 3:
		int nz;
	%endif

	%for i, ve in enumerate(grid.basis):
		%if dim == 3:
			${get_field_off(ve[0], ve[1], ve[2])};
		%else:
			${get_field_off(ve[0], ve[1], 0)};
		%endif

		t1 = sc_ppot(f1, fi1, i + off, ${fld_args()});
		t2 = sc_ppot(f2, fi2, i + off, ${fld_args()});

		%if ve[0] != 0:
			a1[0] += t2 * ${ve[0] * grid.weights[i]};
			a2[0] += t1 * ${ve[0] * grid.weights[i]};
		%endif
		%if ve[1] != 0:
			a1[1] += t2 * ${ve[1] * grid.weights[i]};
			a2[1] += t1 * ${ve[1] * grid.weights[i]};
		%endif
		%if dim == 3 and ve[2] != 0:
			a1[2] += t2 * ${ve[2] * grid.weights[i]};
			a2[2] += t1 * ${ve[2] * grid.weights[i]};
		%endif
	%endfor

	off = 0;
	nx = x;
	ny = y;
	%if dim == 3:
		nz = z;
	%endif


	t1 = sc_ppot(f1, fi1, i + off, ${fld_args()});
	t2 = sc_ppot(f2, fi2, i + off, ${fld_args()});

	%for i in range(0, dim):
		a1[${i}] *= t1 * cc;
		a2[${i}] *= t2 * cc;
	%endfor
}
%endif

// Compute the 0th moment of the distributions, i.e. density.
${device_func} inline void compute_0th_moment(Dist *fi, float *out)
{
	*out = ${sym.ex_rho(grid, 'fi', incompressible)};
}

// Compute the 1st moments of the distributions, i.e. momentum.
${device_func} inline void compute_1st_moment(Dist *fi, float *out, int add, float factor)
{
	if (add) {
		%for d in range(0, grid.dim):
			out[${d}] += factor * (${cex(sym.ex_velocity(grid, 'fi', d, momentum=True), pointers=True)});
		%endfor
	} else {
		%for d in range(0, grid.dim):
			out[${d}] = factor * (${cex(sym.ex_velocity(grid, 'fi', d, momentum=True), pointers=True)});
		%endfor
	}
}

// Compute the 2nd moments of the distributions.  Order of components is:
// 2D: xx, xy, yy
// 3D: xx, xy, xz, yy, yz, zz
${device_func} inline void compute_2nd_moment(Dist *fi, float *out)
{
	%for i, (a, b) in enumerate([(x,y) for x in range(0, dim) for y in range(x, dim)]):
		out[${i}] = ${cex(sym.ex_flux(grid, 'fi', a, b), pointers=True)};
	%endfor
}

// Compute the 1st moments of the distributions and divide it by the 0-th moment
// i.e. compute velocity.
${device_func} inline void compute_1st_div_0th(Dist *fi, float *out, float zero)
{
	%for d in range(0, grid.dim):
		out[${d}] = ${cex(sym.ex_velocity(grid, 'fi', d), pointers=True, rho='zero')};
	%endfor
}

${device_func} inline void compute_macro_quant(Dist *fi, float *rho, float *v)
{
	compute_0th_moment(fi, rho);
	compute_1st_div_0th(fi, v, *rho);
}

%if bc_wall == 'zouhe' or bc_velocity == 'zouhe' or bc_pressure == 'zouhe':
${device_func} void zouhe_bb(Dist *fi, int orientation, float *rho, float *v0)
{
	// Bounce-back of the non-equilibrium parts.
	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			${noneq_bb(i)}
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
	nvx = ${cex(sym.ex_velocity(grid, 'fi', 0, momentum=True))};
	nvy = ${cex(sym.ex_velocity(grid, 'fi', 1, momentum=True))};
	%if dim == 3:
		nvz = ${cex(sym.ex_velocity(grid, 'fi', 2, momentum=True))};
	%endif

	// Compute momentum difference.
	nvx = *rho * v0[0] - nvx;
	nvy = *rho * v0[1] - nvy;
	%if dim == 3:
		nvz = *rho * v0[2] - nvz;
	%endif

	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			${zouhe_fixup(i)}
		%endfor
	}
}
%endif

## TODO integrate it via mako with the function below

${device_func} inline void get0thMoment(Dist *fi, int node_type, int orientation, float *out)
{
	compute_0th_moment(fi, out);
}

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// the node class node_type.
//
${device_func} inline void getMacro(Dist *fi, int ncode, int node_type, int orientation, float *rho, float *v0)
{
	if (isFluidOrWallNode(node_type) || isSlipNode(node_type) || orientation == ${geo_dir_other}) {
		compute_macro_quant(fi, rho, v0);
		if (isWallNode(node_type)) {
			%if bc_wall_.location == 0.0 and bc_wall_.wet_nodes:
				v0[0] = 0.0f;
				v0[1] = 0.0f;
				%if dim == 3:
					v0[2] = 0.0f;
				%endif
			%endif

			%if bc_wall == 'zouhe':
				zouhe_bb(fi, orientation, rho, v0);
			%endif
		}
	} else if (isVelocityNode(node_type)) {
		%if bc_velocity == 'zouhe':
			int node_param = decodeNodeParam(ncode);
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			${get_boundary_velocity('node_param', 'v0[0]', 'v0[1]', 'v0[2]')}
			zouhe_bb(fi, orientation, rho, v0);
		// We're dealing with a boundary node, for which some of the distributions
		// might be meaningless.  Fill them with the values of the opposite
		// distributions.
		%elif bc_velocity == 'equilibrium':
			int node_param = decodeNodeParam(ncode);
			${fill_missing_distributions()}
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			${get_boundary_velocity('node_param', 'v0[0]', 'v0[1]', 'v0[2]')}

			switch (orientation) {
				%for i in range(1, grid.dim*2+1):
					case ${i}:
						*rho = ${cex(sym.ex_rho(grid, 'fi', incompressible, missing_dir=i), pointers=True)};
						break;
				%endfor
			}
		%else:
			compute_macro_quant(fi, rho, v0);
		%endif
	} else if (isPressureNode(node_type)) {
		%if bc_pressure == 'zouhe' or bc_pressure == 'equilibrium':
			int node_param = decodeNodeParam(ncode);
			${fill_missing_distributions()}
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			float par_rho;
			${get_boundary_pressure('node_param', 'par_rho')}

			switch (orientation) {
				%for i in range(1, grid.dim*2+1):
					case ${i}: {
						%for d in range(0, grid.dim):
							v0[${d}] = ${cex(sym.ex_velocity(grid, 'fi', d, missing_dir=i, par_rho='par_rho'), pointers=True)};
						%endfor
						break;
					 }
				%endfor
			}

			%if bc_pressure == 'zouhe':
				zouhe_bb(fi, orientation, &par_rho, v0);
				compute_macro_quant(fi, rho, v0);
			%endif
			*rho = par_rho;
		%else:
			compute_macro_quant(fi, rho, v0);
		%endif
	}
}

// TODO: Check whether it is more efficient to actually recompute
// node_type and orientation instead of passing them as variables.
${device_func} inline void postcollisionBoundaryConditions(Dist *fi, int ncode, int node_type, int orientation, float *rho, float *v0, int gi, ${global_ptr} float *dist_out)
{
	%if bc_wall == 'halfbb':
		if (isWallNode(node_type)) {
			switch (orientation) {
			%for i in range(1, grid.dim*2+1):
				case ${i}: {
					%for lvalue, rvalue in sym.fill_missing_dists(grid, 'fi', missing_dir=i):
						${get_odist('dist_out', lvalue.idx)} = ${rvalue};
					%endfor
					break;
				}
			%endfor
			}
		}
	%endif
}

${device_func} inline void precollisionBoundaryConditions(Dist *fi, int ncode, int node_type, int orientation, float *rho, float *v0)
{
	%if bc_wall == 'fullbb':
		if (isWallNode(node_type)) {
			bounce_back(fi);
		}
	%endif

	%if bc_velocity == 'fullbb':
		if (isVelocityNode(node_type)) {
			bounce_back(fi);
			int node_param = decodeNodeParam(ncode);
			${get_boundary_velocity('node_param', 'v0[0]', 'v0[1]', 'v0[2]')}
			%for i, ve in enumerate(grid.basis):
				fi->${grid.idx_name[i]} += ${cex(
					sim.S.rho0 * 2 * grid.weights[i] * grid.v.dot(ve) / grid.cssq, pointers=True)};
			%endfor
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
		}
	%endif

	%if bc_velocity == 'equilibrium' or bc_pressure == 'equilibrium':
		%for local_var in bgk_equilibrium_vars:
			float ${cex(local_var.lhs)} = ${cex(local_var.rhs)};
		%endfor
	%endif

	%if bc_velocity == 'equilibrium':
		if (isVelocityNode(node_type)) {
			%for eq in bgk_equilibrium:
				%for feq, idx in eq:
					fi->${idx} = ${cex(feq, pointers=True)};
				%endfor
			%endfor
		}
	%endif

	%if bc_pressure == 'equilibrium':
		if (isPressureNode(node_type)) {
			%for eq in bgk_equilibrium:
				%for feq, idx in eq:
					fi->${idx} = ${cex(feq, pointers=True)};
				%endfor
			%endfor
		}
	%endif

	%if bc_slip == 'slipbb':
		if (isSlipNode(node_type)) {
			float t;
			switch (orientation) {
			%for i in range(1, grid.dim*2+1):
				case ${i}: {
					%for j, k in sym.slip_bb_swap_pairs(grid, i):
						t = fi->${grid.idx_name[j]};
						fi->${grid.idx_name[j]} = fi->${grid.idx_name[k]};
						fi->${grid.idx_name[k]} = t;
					%endfor
					break;
				}
			%endfor
			}
		}
	%endif
}


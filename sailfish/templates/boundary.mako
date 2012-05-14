<%!
    from sailfish import sym
    import sailfish.node_type as nt
%>

<%namespace file="code_common.mako" import="*"/>
<%namespace file="propagation.mako" import="rel_offset,get_odist"/>

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

${device_func} inline void node_param_get_vector(const int idx, float *out) {
	out[0] = node_params[idx];
	out[1] = node_params[idx + 1];
	%if dim == 3:
		out[2] = node_params[idx + 2];
	%endif
}

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

%if nt.NTZouHeVelocity in node_types or nt.NTZouHeDensity in node_types:
${device_func} void zouhe_bb(Dist *fi, int orientation, float *rho, float *v0)
{
	// Bounce-back of the non-equilibrium parts.
	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			${noneq_bb(i)}
		%endfor
		case ${nt_dir_other}:
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
%endif  ## ZouHe

## TODO integrate it via mako with the function below

${device_func} inline void get0thMoment(Dist *fi, int node_type, int orientation, float *out)
{
	compute_0th_moment(fi, out);
}

// Common code for the equilibrium and Zou-He density boundary conditions.
<%def name="_macro_density_bc_common()">
	int node_param_idx = decodeNodeParamIdx(ncode);
	${fill_missing_distributions()}
	*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
	float par_rho = node_params[node_param_idx];

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
</%def>

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// the node class node_type.
//
${device_func} inline void getMacro(
		Dist *fi, int ncode, int node_type, int orientation, float *rho,
		float *v0)
{
	if (NTUsesStandardMacro(node_type) || orientation == ${nt_dir_other}) {
		compute_macro_quant(fi, rho, v0);
		%if nt.NTHalfBBWall in node_types:
			if (isNTHalfBBWall(node_type)) {
				v0[0] = 0.0f;
				v0[1] = 0.0f;
				%if dim == 3:
					v0[2] = 0.0f;
				%endif
			}
		%endif
	}
	%if nt.NTEquilibriumVelocity in node_types:
		else if (isNTEquilibriumVelocity(node_type)) {
			int node_param_idx = decodeNodeParamIdx(ncode);
			// We're dealing with a boundary node, for which some of the distributions
			// might be meaningless.  Fill them with the values of the opposite
			// distributions.
			${fill_missing_distributions()}
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			node_param_get_vector(node_param_idx, v0);

			switch (orientation) {
				%for i in range(1, grid.dim*2+1):
					case ${i}:
						*rho = ${cex(sym.ex_rho(grid, 'fi', incompressible, missing_dir=i), pointers=True)};
						break;
				%endfor
			}
		}
	%endif
	%if nt.NTZouHeVelocity in node_types:
		else if (isNTZouHeVelocity(node_type)) {
			int node_param_idx = decodeNodeParamIdx(ncode);
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			node_param_get_vector(node_param_idx, v0);
			zouhe_bb(fi, orientation, rho, v0);
		}
	%endif
	%if nt.NTZouHeDensity in node_types:
		else if (isNTZouHeDensity(node_type)) {
			${_macro_density_bc_common()}
			zouhe_bb(fi, orientation, &par_rho, v0);
			compute_macro_quant(fi, rho, v0);
			*rho = par_rho;
		}
	%endif
	%if nt.NTEquilibriumDensity in node_types:
		else if (isNTEquilibriumDensity(node_type)) {
			${_macro_density_bc_common()}
			*rho = par_rho;
		}
	%endif
}

// TODO: Check whether it is more efficient to actually recompute
// node_type and orientation instead of passing them as variables.
${device_func} inline void postcollisionBoundaryConditions(
		Dist *fi, int ncode, int node_type, int orientation,
		float *rho, float *v0, int gi, ${global_ptr} float *dist_out)
{
	%if nt.NTHalfBBWall in node_types:
		if (isNTHalfBBWall(node_type)) {
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
	%if nt.NTFullBBWall in node_types:
		if (isNTFullBBWall(node_type)) {
			bounce_back(fi);
		}
	%endif

	%if (nt.NTEquilibriumVelocity in node_types) or (nt.NTEquilibriumDensity in node_types):
		## Additional variables required for the evaluation of the
		## equilibrium distribution function.
		%for local_var in bgk_equilibrium_vars:
			float ${cex(local_var.lhs)} = ${cex(local_var.rhs)};
		%endfor
		if (is_NTEquilibriumNode(node_type)) {
			%for eq in bgk_equilibrium:
				%for feq, idx in eq:
					fi->${idx} = ${cex(feq, pointers=True)};
				%endfor
			%endfor
		}
	%endif
	%if nt.NTSlip in node_types:
		%if grid.dim == 3 and grid.Q == 13:
			__SLIP_BOUNDARY_CONDITION_UNSUPPORTED_IN_D3Q13__
		%endif
		if (isNTSlip(node_type)) {
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

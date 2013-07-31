<%!
    from sailfish import sym, sym_equilibrium
    import sailfish.node_type as nt
%>

<%namespace file="code_common.mako" import="*"/>
<%namespace file="propagation.mako" import="rel_offset,get_odist"/>
<%namespace file="utils.mako" import="*"/>
<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>

<%def name="misc_bc_args_decl()">
	${cond(misc_bc_vars, ', ')} ${', '.join('float* %s' % x for x in misc_bc_vars)}
</%def>

<%def name="misc_bc_args()">
	${cond(misc_bc_vars, ', ')} ${', '.join(misc_bc_vars)}
</%def>

<%def name="declare_misc_bc_vars()">
	%if nt.NTWallTMS in node_types:
		// Target macroscopic values for TMS nodes.
		float tg_rho[1];
		float tg_v[${dim}];
	%endif
</%def>

%if time_dependence:
	${device_func} inline float get_time_from_iteration(unsigned int iteration) {
		return iteration * ${cex(dt_per_lattice_time_unit)};
	}
%endif

%for i, expressions in symbol_idx_map.iteritems():
	${device_func} inline void time_dep_param_${i}(float *out ${dynamic_val_args_decl()}) {
		%if time_dependence:
			float phys_time = get_time_from_iteration(iteration_number);
		%endif
		%for j, expr in enumerate(expressions):
			out[${j}] = ${cex(expr)};
		%endfor
	}
%endfor

// Returns a node parameter which is a vector (in 'out').
${device_func} inline void node_param_get_vector(const int idx, float *out
		${dynamic_val_args_decl()}) {
	%if (time_dependence or space_dependence) and symbol_idx_map:
		if (idx >= ${non_symbolic_idxs}) {
			switch (idx) {
				%for key, val in symbol_idx_map.iteritems():
					%if len(val) == dim:
						case ${key}:
							time_dep_param_${key}(out ${dynamic_val_args()});
							return;
					%endif
				%endfor
				default:
					die();
			}
		}
	%endif
	out[0] = node_params[idx];
	out[1] = node_params[idx + 1];
	%if dim == 3:
		out[2] = node_params[idx + 2];
	%endif
}

// Returns a node parameter which is a scalar.
${device_func} inline float node_param_get_scalar(const int idx ${dynamic_val_args_decl()}) {
	%if (time_dependence or space_dependence) and symbol_idx_map:
		if (idx >= ${non_symbolic_idxs}) {
			switch (idx) {
				%for key, val in symbol_idx_map.iteritems():
					%if len(val) == 1:
						case ${key}: {
							float out;
							time_dep_param_${key}(&out ${dynamic_val_args()});
							return out;
						}
					%endif
				%endfor
				default:
					die();
			}
		}
	%endif
	return node_params[idx];
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

// Add comments for the Guo density implementation.
<%def name="guo_density_node_index_shift_intro()">
	%if nt.NTGuoDensity in node_types:
		int orig_gi = gi;
		if (isNTGuoDensity(type)) {
			switch (orientation) {
				%for dir_ in grid.dir2vecidx.keys():
					case (${dir_}): {
						## TODO: add a function to calculate the local indices from gi
						%if dim == 2:
							gi += ${rel_offset(*(list(grid.dir_to_vec(dir_)) + [0]))};
							gx += ${grid.dir_to_vec(dir_)[0]};
							gy += ${grid.dir_to_vec(dir_)[1]};
						%else:
							gi += ${rel_offset(*(grid.dir_to_vec(dir_)))};
							gx += ${grid.dir_to_vec(dir_)[0]};
							gy += ${grid.dir_to_vec(dir_)[1]};
							gz += ${grid.dir_to_vec(dir_)[2]};
						%endif
						break;
					}
				%endfor
			}
		}
	%endif
</%def>

<%def name="guo_density_restore_index()">
	%if nt.NTGuoDensity in node_types:
		if (isNTGuoDensity(type)) {
			gi = orig_gi;
		}
	%endif
</%def>

<%def name="guo_density_node_index_shift_final()">
	%if nt.NTGuoDensity in node_types:
		if (isNTGuoDensity(type)) {
			switch (orientation) {
				%for dir_ in grid.dir2vecidx.keys():
					case (${dir_}): {
						## TODO: add a function to calculate the local indices from gi
						gx -= ${grid.dir_to_vec(dir_)[0]};
						gy -= ${grid.dir_to_vec(dir_)[1]};
						%if dim == 3:
							gz -= ${grid.dir_to_vec(dir_)[2]};
						%endif
						break;
					}
				%endfor
			}
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
			out[${d}] += factor * (${cex(sym.ex_velocity(grid, 'fi', d, config, momentum=True), pointers=True)});
		%endfor
	} else {
		%for d in range(0, grid.dim):
			out[${d}] = factor * (${cex(sym.ex_velocity(grid, 'fi', d, config, momentum=True), pointers=True)});
		%endfor
	}
}

// Compute the 2nd moments of the distributions.  Order of components is:
// 2D: xx, xy, yy
// 3D: xx, xy, xz, yy, yz, zz
${device_func} inline void compute_2nd_moment(Dist *fi, float *out)
{
	%for i, (a, b) in enumerate([(x,y) for x in range(0, dim) for y in range(x, dim)]):
		out[${i}] = ${cex(sym.ex_flux(grid, 'fi', a, b, config), pointers=True)};
	%endfor
}

// Computes the 2nd moment of the non-equilibrium distribution function
// given the full distribution fuction 'fi'.
${device_func} inline void compute_noneq_2nd_moment(Dist* fi, const float rho, float *v0, float *out)
{
	%for i, (a, b) in enumerate([(x,y) for x in range(0, dim) for y in range(x, dim)]):
		out[${i}] = ${cex(sym.ex_flux(grid, 'fi', a, b, config), pointers=True)} -
					${cex(sym.ex_eq_flux(grid, a, b))};
	%endfor
}

// Compute the 1st moments of the distributions and divide it by the 0-th moment
// i.e. compute velocity.
${device_func} inline void compute_1st_div_0th(Dist *fi, float *out, float zero)
{
	%for d in range(0, grid.dim):
		out[${d}] = ${cex(sym.ex_velocity(grid, 'fi', d, config), pointers=True, rho='zero')};
	%endfor
}

${device_func} inline void compute_macro_quant(Dist *fi, float *rho, float *v)
{
	compute_0th_moment(fi, rho);
	compute_1st_div_0th(fi, v, *rho);
}

%if nt.NTZouHeVelocity in node_types or nt.NTZouHeDensity in node_types or nt.NTRegularizedVelocity in node_types:
<%def name="do_noneq_bb(orientation)">
	case ${orientation}:
		<%
			import copy
			tmp_config = copy.copy(config)

			# When this function is called, rho is the node parameter, and therefore
			# the full density, not the density delta. We turn off the round-off
			# minimization locally so that the standard form of equilibrium is used.
			tmp_config.minimize_roundoff = False
		%>
		%for arg, val in sym.noneq_bb(grid, orientation, equilibria[0](grid, tmp_config).expression):
			${cex(arg, pointers=True)} = ${cex(val, pointers=True)};
		%endfor
		break;
</%def>

<%def name="noneq_bb()">
	// Bounce-back of the non-equilibrium parts.
	switch (orientation) {
		%for i in range(1, grid.dim * 2 + 1):
			${do_noneq_bb(i)}
		%endfor
		case ${nt_dir_other}:
			bounce_back(fi);
			return;
	}
</%def>
%endif

%if nt.NTZouHeVelocity in node_types or nt.NTZouHeDensity in node_types:
<%def name="zouhe_fixup(orientation)">
	case ${orientation}:
		%for arg, val in sym.zouhe_fixup(grid, orientation):
			${str(arg)} = ${cex(val, vectors=False)};
		%endfor
		break;
</%def>

${device_func} void zouhe_bb(Dist *fi, int orientation, float *rho, float *v0)
{
	${noneq_bb()}

	float nvx, nvy;
	%if dim == 3:
		float nvz;
	%endif

	// Compute new macroscopic variables.
	nvx = ${cex(sym.ex_velocity(grid, 'fi', 0, config, momentum=True))};
	nvy = ${cex(sym.ex_velocity(grid, 'fi', 1, config, momentum=True))};
	%if dim == 3:
		nvz = ${cex(sym.ex_velocity(grid, 'fi', 2, config, momentum=True))};
	%endif

	// Compute momentum difference. rho here needs to be the full density.
	nvx = *rho * v0[0] - nvx;
	nvy = *rho * v0[1] - nvy;
	%if dim == 3:
		nvz = *rho * v0[2] - nvz;
	%endif

	// Redistribute excess momentum.
	switch (orientation) {
		%for i in range(1, grid.dim * 2 + 1):
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
	float par_rho = node_param_get_scalar(node_param_idx ${dynamic_val_args()});

	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			case ${i}: {
				%for d in range(0, grid.dim):
					v0[${d}] = ${cex(sym.ex_velocity(grid, 'fi', d, config, missing_dir=i, par_rho='par_rho'), pointers=True)};
				%endfor
				break;
			 }
		%endfor
	}
</%def>

<%def name="_macro_velocity_bc_common()">
	int node_param_idx = decodeNodeParamIdx(ncode);
	// We're dealing with a boundary node, for which some of the distributions
	// might be meaningless.  Fill them with the values of the opposite
	// distributions.
	${fill_missing_distributions()}
	*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
	node_param_get_vector(node_param_idx, v0 ${dynamic_val_args()});

	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			case ${i}:
				*rho = ${cex(sym.ex_rho(grid, 'fi', incompressible, missing_dir=i), pointers=True)};
				break;
		%endfor
	}
</%def>

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// the node class node_type.
//
${device_func} inline void getMacro(
		Dist *fi, int ncode, int node_type, int orientation, float *rho,
		float *v0 ${dynamic_val_args_decl()})
{
	if (NTUsesStandardMacro(node_type) || orientation == ${nt_dir_other}) {
		compute_macro_quant(fi, rho, v0);
	}
	%if nt.NTEquilibriumVelocity in node_types:
		else if (isNTEquilibriumVelocity(node_type)) {
			${_macro_velocity_bc_common()}
		}
	%endif
	%if nt.NTZouHeVelocity in node_types:
		else if (isNTZouHeVelocity(node_type)) {
			${_macro_velocity_bc_common()}
		}
	%endif
	%if nt.NTRegularizedVelocity in node_types:
		else if (isNTRegularizedVelocity(node_type)) {
			${_macro_velocity_bc_common()}
		}
	%endif
	%if nt.NTZouHeDensity in node_types:
		else if (isNTZouHeDensity(node_type)) {
			${_macro_density_bc_common()}
			zouhe_bb(fi, orientation, &par_rho, v0);
			compute_macro_quant(fi, rho, v0);
			*rho = par_rho ${'-1.0f' if config.minimize_roundoff else ''};
		}
	%endif
	%if nt.NTEquilibriumDensity in node_types:
		else if (isNTEquilibriumDensity(node_type)) {
			${_macro_density_bc_common()}
			*rho = par_rho ${'-1.0f' if config.minimize_roundoff else ''};
		}
	%endif
	%if nt.NTRegularizedDensity in node_types:
		else if (isNTRegularizedDensity(node_type)) {
			${_macro_density_bc_common()}
			*rho = par_rho ${'-1.0f' if config.minimize_roundoff else ''};
		}
	%endif
}


// Uses extrapolation/other schemes to compute missing distributions for some implementations
// of boundary condtitions.
${device_func} inline void fixMissingDistributions(
		Dist *fi, ${global_ptr} float *dist_in, int ncode, int node_type, int orientation, int gi,
		${kernel_args_1st_moment('iv')}
		${global_ptr} float *gg0m0
		${misc_bc_args_decl()}
		${scratch_space_if_required()}) {
	if (0) {}
	## These boundary conditions are non-local and can thus be implemented in
	## this way only with the AB access pattern. In the AA access pattern,
	## their implementation requires a separate kernel call.
	%if access_pattern == 'AB':
		%if nt.NTCopy in node_types:
			else if (isNTCopy(node_type)) {
				switch (orientation) {
				%for o in range(1, grid.dim*2+1):
					case ${o}: {
						%for dist_idx in sym.get_missing_dists(grid, o):
							fi->${grid.idx_name[dist_idx]} = ${get_odist('dist_in', dist_idx, *grid.dir_to_vec(o))};
						%endfor
						break;
					}
				%endfor
				}
			}
		%endif

		%if nt.NTYuOutflow in node_types:
			else if (isNTYuOutflow(node_type)) {
				switch (orientation) {
				%for o in range(1, grid.dim*2+1):
					case ${o}: {
						%for dist_idx in sym.get_missing_dists(grid, o):
							fi->${grid.idx_name[dist_idx]} =
								2.0f * ${get_odist('dist_in', dist_idx, *grid.dir_to_vec(o))} -
								${get_odist('dist_in', dist_idx, *(2 * grid.dir_to_vec(o)))};
						%endfor
						break;
					}
				%endfor
				}
			}
		%endif

		%if nt.NTGradFreeflow in node_types:
			else if (isNTGradFreeflow(node_type)) {
				// Load values for velocity and density from the previous step.
				float rho = gg0m0[gi];
				float vx = ivx[gi];
				float vy = ivy[gi];
				${'float vz = ivz[gi];' if dim == 3 else ''}
				float flux[${flux_components}];
				int scratch_id = decodeNodeScratchId(ncode);
				loadNodeScratchSpace(scratch_id, node_type, node_scratch_space, flux);
				%for idx, grad_approx in zip(grid.idx_name, sym.grad_approx(grid)):
					// Fill undefined distributions from the Grad aproximation.
					if (!isfinite(fi->${idx})) {
						fi->${idx} = ${cex(grad_approx, vectors=False)};
					}
				%endfor
			}
		%endif
	%endif

	%if nt.NTWallTMS in node_types:
		// Replaces the missing distributions using the bounce-back rule.
		// No density/momentum correction happens here.
		else if (isNTWallTMS(node_type)) {
			compute_macro_quant(fi, tg_rho, tg_v);

			<% eq = sym_equilibrium.get_equilibrium(config, equilibria, grids, 0) %>
			%for local_var in eq.local_vars:
				const float ${cex(local_var.lhs)} =
					${cex(local_var.rhs, rho='*tg_rho', vel='tg_v')};
			%endfor
			// Replace missing distributions with equilibrium ones
			// calculated for the target macroscopic variables.
			switch (orientation) {
				%for i in range(1, grid.dim*2+1):
					case ${i}: {
						%for lvalue, rvalue in sym.fill_missing_dists(grid, 'fi', missing_dir=i):
							${lvalue.var} = ${cex(eq.expression[lvalue.idx], rho='*tg_rho', vel='tg_v')};
						%endfor
						break;
					}
				%endfor
			}
		}
	%endif
}

// TODO: Check whether it is more efficient to actually recompute
// node_type and orientation instead of passing them as variables.
${device_func} inline void postcollisionBoundaryConditions(
		Dist *fi, int ncode, int node_type, int orientation,
		float *rho, float *v0, int gi, ${global_ptr} float *dist_out
		${iteration_number_if_required()}
		${misc_bc_args_decl()}
		${scratch_space_if_required()})
{
	if (0) {}

	%if nt.NTHalfBBWall in node_types:
		else if (isNTHalfBBWall(node_type)) {
			%for i, (name, opp_idx) in enumerate(zip(grid.idx_name[1:], grid.idx_opposite[1:])):
				## Don't generate code for cases that never happen.
				%if unused_tag_bits & (1 << i) == 0:
					// ${name} points to a missing node
					if (orientation & ${1 << i} == 0) {
						%if access_pattern == 'AB':
							${get_odist('dist_out', opp_idx)} = fi->${name};
						%else:
							if (iteration_number & 1) {
								${get_odist('dist_out', opp_idx)} = fi->${name};
							} else {
								${get_odist('dist_out', i + 1,
											*grid.basis[i + 1])} = fi->${name};
							}
						%endif
					}
				%endif
			%endfor

##			switch (orientation) {
##			%for i in range(1, grid.dim * 2 + 1):
##				case ${i}: {
##					%for lvalue, rvalue in sym.fill_missing_dists(grid, 'fi', missing_dir=i):
##						%if access_pattern == 'AB':
##							${get_odist('dist_out', lvalue.idx)} = ${rvalue};  // ${lvalue.var}
##						%else:
##							if (iteration_number & 1) {
##								${get_odist('dist_out', lvalue.idx)} = ${rvalue};  // ${lvalue.var}
##							} else {
##								${get_odist('dist_out', grid.idx_opposite[lvalue.idx],
##											*grid.basis[grid.idx_opposite[lvalue.idx]])} = ${rvalue};  // ${lvalue.var}
##							}
##						%endif
##					%endfor
##					break;
##				}
##			%endfor
##			}
		}
	%endif

	%if nt.NTGradFreeflow in node_types:
		// Store the flux tensor so that it can be used to compute the Grad approximation
		// in the next iteration.
		else if (isNTGradFreeflow(node_type)) {
			int scratch_id = decodeNodeScratchId(ncode);
			float flux[${flux_components}];
			compute_2nd_moment(fi, flux);
			storeNodeScratchSpace(scratch_id, node_type, flux, node_scratch_space);
		}
	%endif

	%if nt.NTWallTMS in node_types:
		// Adds the (f^eq(TG) - f^eq(inst)) part of the distribution.
		else if (isNTWallTMS(node_type)) {
			{
				<% eq = sym_equilibrium.get_equilibrium(config, equilibria, grids, 0) %>
				%for local_var in eq.local_vars:
					const float ${cex(local_var.lhs)} =
						${cex(local_var.rhs, rho='*tg_rho', vel='tg_v', pointers=True)};
				%endfor

				%for val, idx in zip(eq.expression, grid.idx_name):
					fi->${idx} += ${cex(val, rho='*tg_rho', vel='tg_v', pointers=True)};
				%endfor
			}
			{
				<% eq = sym_equilibrium.get_equilibrium(config, equilibria, grids, 0) %>
				%for local_var in eq.local_vars:
					const float ${cex(local_var.lhs)} = ${cex(local_var.rhs, pointers=True)};
				%endfor

				%for val, idx in zip(eq.expression, grid.idx_name):
					fi->${idx} -= ${cex(val, pointers=True)};
				%endfor
			}

			// Write new values back to memory like for the half-way bounce-back.
			switch (orientation) {
			%for i in range(1, grid.dim * 2 + 1):
				case ${i}: {
					%for lvalue, rvalue in sym.fill_missing_dists(grid, 'fi', missing_dir=i):
						%if access_pattern == 'AB':
							${get_odist('dist_out', lvalue.idx)} = ${rvalue};  // ${lvalue.var}
						%else:
							if (iteration_number & 1) {
								${get_odist('dist_out', lvalue.idx)} = ${rvalue};  // ${lvalue.var}
							} else {
								${get_odist('dist_out', grid.idx_opposite[lvalue.idx],
											*grid.basis[grid.idx_opposite[lvalue.idx]])} = ${rvalue};  // ${lvalue.var}
							}
						%endif
					%endfor
					break;
				}
			%endfor
			}
		}
	%endif
}


${device_func} inline void precollisionBoundaryConditions(Dist *fi, int ncode,
		int node_type, int orientation, float *rho, float *v0
		${', ' + global_ptr + 'float *dist_out, int gi' if access_pattern == 'AA' and nt.NTDoNothing in node_types else ''}
		${iteration_number_if_required()})
{
	if (0) {}

	%if nt.NTFullBBWall in node_types:
		else if (isNTFullBBWall(node_type)) {
			bounce_back(fi);
		}
	%endif

	%if (nt.NTEquilibriumVelocity in node_types) or (nt.NTEquilibriumDensity in node_types):
		## Additional variables required for the evaluation of the
		## equilibrium distribution function.
		else if (is_NTEquilibriumNode(node_type)) {
			<% eq = equilibria[0](grid, config) %>
			%for local_var in eq.local_vars:
				float ${cex(local_var.lhs)} = ${cex(local_var.rhs)};
			%endfor
			%for feq, idx in zip(eq.expression, grid.idx_name):
				fi->${idx} = ${cex(feq, pointers=True)};
			%endfor
		}
	%endif

	%if nt.NTZouHeVelocity in node_types:
		else if (isNTZouHeVelocity(node_type)) {
			zouhe_bb(fi, orientation, rho, v0);
		}
	%endif

	%if nt.NTRegularizedVelocity in node_types or nt.NTRegularizedDensity in node_types:
		else if (0 ${'|| isNTRegularizedVelocity(node_type)' if nt.NTRegularizedVelocity in node_types else ''}
				   ${'|| isNTRegularizedDensity(node_type)' if nt.NTRegularizedDensity in node_types else ''}) {
			${noneq_bb()}
			float flux[${flux_components}];
			compute_noneq_2nd_moment(fi, *rho, v0, flux);

			<%
				eq = equilibria[0](grid, config)
				reg_diff = sym.reglb_flux_tensor(grid)
			%>
			%for local_var in eq.local_vars:
				float ${cex(local_var.lhs)} = ${cex(local_var.rhs)};
			%endfor
			%for feq, idx, reg in zip(eq.expression, grid.idx_name, reg_diff):
				fi->${idx} = max(1e-7f, ${cex(feq, pointers=True)} + ${cex(reg, pointers=True)});
			%endfor
		}
	%endif

	%if nt.NTSlip in node_types:
		%if grid.dim == 3 and grid.Q == 13:
			__SLIP_BOUNDARY_CONDITION_UNSUPPORTED_IN_D3Q13__
		%endif
		else if (isNTSlip(node_type)) {
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

	%if access_pattern == 'AA' and nt.NTDoNothing in node_types:
		// Only need to do special processing for the propagate in-place step.
		// For this step, we store the missing distributions in the ghost nodes
		// adjacent to the do-nothing node in such a way that in the next iteration
		// they will be retrieved using the standard procedure.
		else if (isNTDoNothing(node_type)) {
			switch (orientation) {
				%for o in range(1, grid.dim*2+1):
					case ${o}: {
						%for dist_idx in sym.get_missing_dists(grid, o):
							if (iteration_number & 1) {
								${get_odist('dist_out', dist_idx)} = fi->${grid.idx_name[dist_idx]};
							} else {
								${get_odist('dist_out', grid.idx_opposite[dist_idx],
											*grid.basis[grid.idx_opposite[dist_idx]])} = fi->${grid.idx_name[dist_idx]};
							}
						%endfor
						break;
					}
				%endfor
			}
		}
	%endif
}

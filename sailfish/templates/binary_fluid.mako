<%!
    from sailfish import sym
    import sailfish.node_type as nt
    import sympy
%>
<%
	# Necessary to support force reassignment in the free energy MRT model.
    phi_needs_rho = (force_for_eq is not UNDEFINED and force_for_eq.get(1) == 0)
%>

<%def name="bgk_args_decl_sc(grid_idx)">
	%if grid_idx == 0:
		float rho, float *iv0, float *ea${grid_idx}
	%else:
		float phi, float *iv0, float *ea${grid_idx}
	%endif
</%def>

<%def name="bgk_args_decl_fe(grid_idx)">
	%if grid_idx == 0:
		float rho, float phi, float lap1, float *iv0, float *grad1
	%else:
		${'float rho, ' if phi_needs_rho else ''} float phi, float lap1, float *iv0
	%endif
</%def>

<%def name="bgk_args_fe(grid_idx)">
	%if grid_idx == 0:
		g0m0, g1m0, lap1, v, grad1
	%else:
		${'g0m0, ' if phi_needs_rho else ''} g1m0, lap1, v
	%endif
</%def>

%if simtype == 'shan-chen':
	## In the free-energy model, the relaxation time is a local quantity.
	${const_var} float tau0 = ${tau}f;		// relaxation time
	// Relaxation time for the 2nd fluid component.
%else:
	// Relaxation time for the order parameter field.
%endif
${const_var} float tau1 = ${tau_phi}f;

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
%if simtype == 'shan-chen':
	<%namespace file="shan_chen.mako" import="*" name="shan_chen"/>
	${kernel_common.body(bgk_args_decl_sc)}
	${shan_chen.body()}
%elif simtype == 'free-energy':
	${kernel_common.body(bgk_args_decl_fe)}
%endif
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>
<%namespace file="propagation.mako" import="*"/>
<%namespace file="utils.mako" import="*"/>

%if simtype == 'free-energy':
<%include file="finite_difference_optimized.mako"/>
%endif

<%def name="init_dist_with_eq()">
	%if simtype == 'free-energy':
		float lap1, grad1[${dim}];
		%if dim == 2:
			laplacian_and_grad(iphi, -1, gi, &lap1, grad1, gx, gy);
		%else:
			laplacian_and_grad(iphi, -1, gi, &lap1, grad1, gx, gy, gz);
		%endif
	%endif

	%for eq, dist_name in zip([f(g, config) for f, g in zip(equilibria, grids)], ['dist1_in', 'dist2_in']):
		%for local_var in eq.local_vars:
			float ${cex(local_var.lhs)} = ${cex(local_var.rhs)};
		%endfor

		%for i, feq in enumerate(eq.expression):
			${get_odist(dist_name, i)} = ${cex(feq)};
		%endfor
	%endfor
</%def>

// A kernel to set the node distributions using the equilibrium distributions
// and the macroscopic fields.
${kernel} void SetInitialConditions(
	${global_ptr} float *dist1_in,
	${global_ptr} float *dist2_in,
	${kernel_args_1st_moment('iv')}
	${global_ptr} ${const_ptr} float *__restrict__ irho,
	${global_ptr} ${const_ptr} float *__restrict__ iphi)
{
	${local_indices()}

	// Cache macroscopic fields in local variables.
	float rho = irho[gi];
	float phi = iphi[gi];
	float v0[${dim}];

	v0[0] = ivx[gi];
	v0[1] = ivy[gi];
	%if dim == 3:
		v0[2] = ivz[gi];
	%endif

	${init_dist_with_eq()}
}

%if simtype == 'free-energy':
${kernel} void FreeEnergyPrepareMacroFields(
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist1_in,
	${global_ptr} ${const_ptr} float *__restrict__ dist2_in,
	${global_ptr} float *__restrict__ orho,
	${global_ptr} float *__restrict__ ophi,
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${load_node_type()}

	// Do not not update the fields for pressure nodes, where by definition
	// they are constant.
	%for node_type in (nt.NTGuoDensity, nt.NTEquilibriumDensity, nt.NTZouHeDensity):
		%if node_type in node_types:
			if (is${node_type.__name__}(type)) { return; }
		%endif
	%endfor

	Dist fi;
	float out;

	if (isWetNode(type)) {
		getDist(&fi, dist2_in, gi ${iteration_number_arg_if_required()});
		get0thMoment(&fi, type, orientation, &out);
		ophi[gi] = out;
	}

	## Assume neutral wetting for all walls by adjusting the phase gradient
	## near the wall.
	##
	## This wetting boundary condition implementation is as in option 2 in
	## Halim Kusumaatmaja's PhD thesis, p.18.

	## Symbols used on the schematics below:
	##
	## W: wall node (current node, pointed to by 'gi')
	## F: fluid node
	## |: actual location of the wall
	## .: space between fluid nodes
	## x: node from which data is read
	## y: node to which data is being written
	##
	## The schematics assume a bc_wall_grad_order of 2.
	%if nt.NTFullBBWall in node_types:
		int helper_idx = gi;
		## Full BB: F . F | W
		##          x ----> y
		if (isNTFullBBWall(type)) {
			switch (orientation) {
				%for dir in grid.dir2vecidx.keys():
					case ${dir}: {  // ${grid.dir_to_vec(dir)}
						%if dim == 3:
							helper_idx += ${rel_offset(*(bc_wall_grad_order*grid.dir_to_vec(dir)))};
						%else:
							## rel_offset() needs a 3-vector, so make the z-coordinate 0
							helper_idx += ${rel_offset(*(list(bc_wall_grad_order*grid.dir_to_vec(dir)) + [0]))};
						%endif
						break;
					}
				%endfor
			}
			getDist(&fi, dist2_in, helper_idx ${iteration_number_arg_if_required()});
			get0thMoment(&fi, type, orientation, &out);
			ophi[gi] = out - (${bc_wall_grad_order*bc_wall_grad_phase});
		}
	%endif  ## NTFullBBWall
	%if nt.NTHalfBBWall in node_types:
		%if bc_wall_grad_order != 1:
			__ONLY_FIRST_ORDER_GRADIENTS_ARE_SUPPORTED_FOR_HALF_BB_WETTING_WALLS__
		%endif
		int helper_idx = gi;

		## Half-way  BB: F . W | U
		##               x ----> y
		if (isNTHalfBBWall(type)) {
			switch (orientation) {
				%for dir in grid.dir2vecidx.keys():
					case ${dir}: {  // ${grid.dir_to_vec(dir)}
						%if dim == 3:
							helper_idx -= ${rel_offset(*(grid.dir_to_vec(dir)))};
						%else:
							helper_idx -= ${rel_offset(*(list(grid.dir_to_vec(dir)) + [0]))};
						%endif
						break;
					}
				%endfor
			}

			ophi[helper_idx] = out - (${bc_wall_grad_order*bc_wall_grad_phase});
		}
	%endif
}

${kernel} void FreeEnergyCollideAndPropagateFluid(
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist1_in,
	${global_ptr} float *__restrict__ dist1_out,
	${global_ptr} float *__restrict__ gg0m0,
	${global_ptr} float *__restrict__ gg1m0,
	${kernel_args_1st_moment('ov')}
	${global_ptr} float *__restrict__ gg1laplacian,
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${shared_mem_propagation_vars()}
	${load_node_type()}
	${guo_density_node_index_shift_intro()}

	float lap1, grad1[${dim}];
	if (isWetNode(type)) {
		laplacian_and_grad(gg1m0, 1, gi, &lap1, grad1, gx, gy ${', gz' if dim == 3 else ''});
	}

	// Macroscopic quantities for the current cell.
	float g0m0, v[${dim}], g1m0;

	// Cache the distributions in local variables.
	Dist d0;
	getDist(&d0, dist1_in, gi ${iteration_number_arg_if_required()});
	g1m0 = gg1m0[gi];
	${guo_density_restore_index()}

	getMacro(&d0, ncode, type, orientation, &g0m0, v ${dynamic_val_call_args()});

	// Save laplacian and velocity to global memory so that they can be reused
	// in the relaxation of the order parameter field.
	ovx[gi] = v[0];
	ovy[gi] = v[1];
	${'ovz[gi] = v[2]' if dim == 3 else ''};
	gg1laplacian[gi] = lap1;

	%if phi_needs_rho:
		gg0m0[gi] = g0m0;
	%endif

	precollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v
								   ${', dist_out1, gi' if access_pattern == 'AA' and nt.NTDoNothing in node_types else ''}
								   ${iteration_number_arg_if_required()});

	${relaxate(bgk_args_fe, 0)}
	postcollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v, gi, dist1_out
									${iteration_number_arg_if_required()});
	${guo_density_node_index_shift_final()}
	${check_invalid_values()}
	${save_macro_fields(velocity=False)}
	${propagate('dist1_out', 'd0')}
}

${kernel} void FreeEnergyCollideAndPropagateOrderParam(
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist1_in,
	${global_ptr} float *__restrict__ dist1_out,
	${global_ptr + const_ptr + ' float *__restrict__ gg0m0,' if phi_needs_rho else ''}
	${global_ptr} ${const_ptr} float *__restrict__ gg1m0,
	${kernel_args_1st_moment('ov')}
	${global_ptr} ${const_ptr} float *__restrict__ gg1laplacian,
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${shared_mem_propagation_vars()}
	${load_node_type()}
	${guo_density_node_index_shift_intro()}
	// Cache the distributions in local variables.
	Dist d0;
	getDist(&d0, dist1_in, gi ${iteration_number_arg_if_required()});
	${guo_density_restore_index()}
	float lap1 = gg1laplacian[gi];
	float g1m0, v[${dim}];
	${'float g0m0 = gg0m0[gi];' if phi_needs_rho else ''}

	v[0] = ovx[gi];
	v[1] = ovy[gi];
	${'v[2] = ovz[gi]' if dim == 3 else ''};
	g1m0 = gg1m0[gi];

	precollisionBoundaryConditions(&d0, ncode, type, orientation, &g1m0, v
								   ${', dist1_out, gi' if access_pattern == 'AA' and nt.NTDoNothing in node_types else ''}
								   ${iteration_number_arg_if_required()});
	${relaxate(bgk_args_fe, 1)}
	postcollisionBoundaryConditions(&d0, ncode, type, orientation, &g1m0, v, gi, dist1_out
									${iteration_number_arg_if_required()});
	${guo_density_node_index_shift_final()}
	${check_invalid_values()}
	${propagate('dist1_out', 'd0')}
}

%endif  ## free-energy

%if simtype == 'shan-chen':
<%include file="binary_shan_chen.mako"/>
%endif  ## shan-chen

<%include file="util_kernels.mako"/>

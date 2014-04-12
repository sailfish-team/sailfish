<%!
    from sailfish import sym
    import sailfish.node_type as nt
    import sympy
%>
<%
	# Necessary to support force reassignment in the free energy MRT model.
	phi_needs_rho = (force_for_eq is not UNDEFINED and force_for_eq.get(1) == 0)
%>

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="utils.mako" import="*"/>
<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>
<%namespace file="propagation.mako" import="*"/>

<%def name="bgk_args_fe(grid_idx)">
	%if grid_idx == 0:
		g0m0, g1m0, lap1, v, grad1
	%else:
		${'g0m0, ' if phi_needs_rho else ''} g1m0, lap1, v
	%endif
</%def>

${kernel} void FreeEnergyPrepareMacroFields(
	${nodes_array_if_required()}
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
	${indirect_index()}
	${load_node_type()}

	if (isPropagationOnly(type)) {
		return;
	}

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
		getDist(
			${nodes_array_arg_if_required()}
			&fi, dist2_in, gi
			${dense_gi_arg_if_required()}
			${iteration_number_arg_if_required()});
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
		int helper_idx = ${'dense_gi' if node_addressing == 'indirect' else 'gi'};
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

			%if node_addressing == 'indirect':
				int dense_helper_idx = helper_idx;
				helper_idx = nodes[helper_idx];
			%endif
			getDist(
				${nodes_array_arg_if_required()}
				&fi, dist2_in, helper_idx
				${', dense_helper_idx' if node_addressing == 'indirect' else ''}
				${iteration_number_arg_if_required()});
			get0thMoment(&fi, type, orientation, &out);
			ophi[gi] = out - (${bc_wall_grad_order*bc_wall_grad_phase});
		}
	%endif  ## NTFullBBWall
	%if nt.NTHalfBBWall in node_types:
		%if bc_wall_grad_order != 1:
			#error Only first order gradients are supported for half-way BB wetting walls.
		%endif
		int helper_idx = ${'dense_gi' if node_addressing == 'indirect' else 'gi'};

		## Half-way  BB: F . W | U
		##                   x -> y
		if (isNTHalfBBWall(type)) {
			%if use_link_tags:
				#error Link tagging it not supported for binary fluid models. Use --nouse_link_tags.
			%endif

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

			%if node_addressing == 'indirect':
				helper_idx = nodes[helper_idx];
				if (helper_idx == INVALID_NODE) {
					printf("Invalid node detected at dense_gi = %d\n", dense_gi);
					die();
				}
			%endif
			ophi[helper_idx] = out - (${bc_wall_grad_order*bc_wall_grad_phase});
		}
	%endif
}

${kernel} void FreeEnergyCollideAndPropagateFluid(
	${nodes_array_if_required()}
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist1_in,
	${global_ptr} float *__restrict__ dist_out,
	${global_ptr} float *__restrict__ gg0m0,
	${global_ptr} float *__restrict__ gg1m0,
	${kernel_args_1st_moment('ov')}
	${global_ptr} float *__restrict__ gg1laplacian,
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${indirect_index()}
	${shared_mem_propagation_vars()}
	${load_node_type()}

	Dist d0;
	if (!isPropagationOnly(type)) {
		${guo_density_node_index_shift_intro()}

		float lap1, grad1[${dim}];
		if (isWetNode(type)) {
			laplacian_and_grad(gg1m0, 1, gi, &lap1, grad1, gx, gy ${', gz' if dim == 3 else ''});
		}

		// Macroscopic quantities for the current cell.
		float g0m0, v[${dim}], g1m0;

		// Cache the distributions in local variables.
		getDist(
			${nodes_array_arg_if_required()}
			&d0, dist1_in, gi
			${dense_gi_arg_if_required()}
			${iteration_number_arg_if_required()});
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
									   ${precollision_arguments()}
									   ${iteration_number_arg_if_required()});

		${relaxate(bgk_args_fe, 0)}
		postcollisionBoundaryConditions(&d0, ncode, type, orientation, &g0m0, v, gi, dist_out
										${iteration_number_arg_if_required()});
		${guo_density_node_index_shift_final()}
		${check_invalid_values(0)}
		${save_macro_fields(velocity=False)}
	}  // propagation only
	${propagate('dist_out', 'd0')}
}

${kernel} void FreeEnergyCollideAndPropagateOrderParam(
	${nodes_array_if_required()}
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist1_in,
	${global_ptr} float *__restrict__ dist_out,
	${global_ptr + const_ptr + ' float *__restrict__ gg0m0,' if phi_needs_rho else ''}
	${global_ptr} ${const_ptr} float *__restrict__ gg1m0,
	${kernel_args_1st_moment('ov')}
	${global_ptr} ${const_ptr} float *__restrict__ gg1laplacian,
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${indirect_index()}
	${shared_mem_propagation_vars()}
	${load_node_type()}

	Dist d1;
	if (!isPropagationOnly(type)) {
		${guo_density_node_index_shift_intro()}
		// Cache the distributions in local variables.
		getDist(
			${nodes_array_arg_if_required()}
			&d1, dist1_in, gi
			${dense_gi_arg_if_required()}
			${iteration_number_arg_if_required()});
		${guo_density_restore_index()}
		float lap1 = gg1laplacian[gi];
		float g1m0, v[${dim}];
		${'float g0m0 = gg0m0[gi];' if phi_needs_rho else ''}

		v[0] = ovx[gi];
		v[1] = ovy[gi];
		${'v[2] = ovz[gi]' if dim == 3 else ''};
		g1m0 = gg1m0[gi];

		precollisionBoundaryConditions(&d1, ncode, type, orientation, &g1m0, v
									   ${precollision_arguments()}
									   ${iteration_number_arg_if_required()});
		${relaxate(bgk_args_fe, 1)}
		postcollisionBoundaryConditions(&d1, ncode, type, orientation, &g1m0, v, gi, dist_out
										${iteration_number_arg_if_required()});
		${guo_density_node_index_shift_final()}
		${check_invalid_values(1)}
	}  // propagation only
	${propagate('dist_out', 'd1')}
}

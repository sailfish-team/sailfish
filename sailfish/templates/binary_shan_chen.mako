<%!
    from sailfish import sym
    import sailfish.node_type as nt
    import sympy
%>

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="utils.mako" import="*"/>
<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>
<%namespace file="propagation.mako" import="*"/>
<%namespace file="shan_chen.mako" import="*" name="shan_chen"/>

<%def name="bgk_args_sc(grid_idx)">
	g${grid_idx}m0, v, sca0
</%def>

${kernel} void ShanChenPrepareMacroFields(
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist1_in,
	${global_ptr} ${const_ptr} float *__restrict__ dist2_in,
	${global_ptr} float *__restrict__ orho0,
	${global_ptr} float *__restrict__ orho1,
	${kernel_args_1st_moment('ov')}
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${load_node_type()}

	// Do not update the macroscopic fields for nodes which do not
	// represent any fluid.
	if (!isWetNode(type)) {
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
	float rho0, rho1;
	float v[${dim}];

	// Velocity requires input from both lattices, so we calculate it here so
	// that it can just be read out in the collision kernel.
	v[0] = 0.0f;
	v[1] = 0.0f;
	${'v[2] = 0.0f;' if dim == 3 else ''}

	getDist(&fi, dist1_in, gi ${iteration_number_arg_if_required()});
	get0thMoment(&fi, type, orientation, &rho0);
	compute_1st_moment(&fi, v, 0, tau0_inv);
	orho0[gi] = rho0;

	// TODO: try moving this line earlier and see what the performance impact is.
	getDist(&fi, dist2_in, gi ${iteration_number_arg_if_required()});
	get0thMoment(&fi, type, orientation, &rho1);
	compute_1st_moment(&fi, v, 1, tau1_inv);
	orho1[gi] = rho1;

	// Velocity becomes a weighted average of the values for invidual components.
	const float total_rho = tau0_inv * rho0 + tau1_inv * rho1;
	%for i in range(0, dim):
		v[${i}] /= total_rho;
	%endfor

	ovx[gi] = v[0];
	ovy[gi] = v[1];
	${'ovz[gi] = v[2];' if dim == 3 else ''}
}

%for grid_idx in range(0, 2):
${kernel} void ShanChenCollideAndPropagate${grid_idx}(
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist1_in,
	${global_ptr} float *__restrict__ dist1_out,
	${global_ptr} ${const_ptr} float *__restrict__ gg0m0,
	${global_ptr} ${const_ptr} float *__restrict__ gg1m0,
	${kernel_args_1st_moment('ov', const=True)}
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${shared_mem_propagation_vars()}
	${load_node_type()}
	Dist d0;
	if (!isPropagationOnly(type)) {
		${guo_density_node_index_shift_intro()}
		float g${grid_idx}m0 = gg${grid_idx}m0[gi];
		${sc_calculate_force(grid_idx=grid_idx)}

		// Cache the distributions in local variables.
		getDist(&d0, dist1_in, gi ${iteration_number_arg_if_required()});

		// Macroscopic quantities for the current cell.
		float v[${dim}];
		v[0] = ovx[gi];
		v[1] = ovy[gi];
		${'v[2] = ovz[gi]' if dim == 3 else ''};

		${guo_density_restore_index()}

		precollisionBoundaryConditions(&d0, ncode, type, orientation, &g${grid_idx}m0, v
									   ${', dist1_out, gi' if access_pattern == 'AA' and nt.NTDoNothing in node_types else ''}
									   ${iteration_number_arg_if_required()});
		${relaxate(bgk_args_sc, grid_idx)}

		// FIXME: In order for the half-way bounce back boundary condition to work, a layer of unused
		// nodes currently has to be placed behind the wall layer.
		postcollisionBoundaryConditions(&d0, ncode, type, orientation, &g${grid_idx}m0, v, gi, dist1_out
										${iteration_number_arg_if_required()});
		${guo_density_node_index_shift_final()}
		${check_invalid_values()}
	}  // propagation only
	${propagate('dist1_out', 'd0')}
}
%endfor

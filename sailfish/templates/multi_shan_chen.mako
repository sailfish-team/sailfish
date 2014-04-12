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
	g${grid_idx}m0, v, sca${grid_idx}
</%def>

${kernel} void ShanChenPrepareMacroFields(
	${nodes_array_if_required()}
	${global_ptr} ${const_ptr} int *__restrict__ map,
	%for grid_idx in range(0,len(grids)):
		${global_ptr} ${const_ptr} float *__restrict__ dist${grid_idx}_in,
	%endfor
	%for grid_idx in range(0,len(grids)):
		${global_ptr} float *__restrict__ og${grid_idx}m0,
	%endfor
	${kernel_args_1st_moment('ov')}
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${indirect_index()}
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
	float ${', '.join(['g{}m0'.format(grid_idx) for grid_idx in range(0, len(grids))])};
	float v[${dim}];

	// Velocity requires input from both lattices, so we calculate it here so
	// that it can just be read out in the collision kernel.
	v[0] = 0.0f;
	v[1] = 0.0f;
	${'v[2] = 0.0f;' if dim == 3 else ''}
	// TODO: try moving these lines earlier and see what the performance impact is.
	%for grid_idx in range(0,len(grids)):
		getDist(
			${nodes_array_arg_if_required()}
			&fi, dist${grid_idx}_in, gi
			${dense_gi_arg_if_required()}
			${iteration_number_arg_if_required()});
		get0thMoment(&fi, type, orientation, &g${grid_idx}m0);
		compute_1st_moment(&fi, v, ${grid_idx}, tau${grid_idx}_inv);
		og${grid_idx}m0[gi] = g${grid_idx}m0;
	%endfor

	// Velocity becomes a weighted average of the values for invidual components.
	const float total_rho = ${' + '.join(['tau{}_inv * g{}m0'.format(c,c) for c in range(0,len(grids))])};
	%for i in range(0, dim):
		v[${i}] /= total_rho;
	%endfor

	ovx[gi] = v[0];
	ovy[gi] = v[1];
	${'ovz[gi] = v[2];' if dim == 3 else ''}
}

%for grid_idx1 in range(0, len(grids)):
${kernel} void ShanChenCollideAndPropagate${grid_idx1}(
	${nodes_array_if_required()}
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${global_ptr} ${const_ptr} float *__restrict__ dist${grid_idx1}_in,
	${global_ptr} float *__restrict__ dist${grid_idx1}_out,
	%for grid_idx2 in range(0,len(grids)):
		${global_ptr} ${const_ptr} float *__restrict__ gg${grid_idx2}m0,
	%endfor
	${kernel_args_1st_moment('ov', const=True)}
	int options
	${scratch_space_if_required()}
	${iteration_number_if_required()})
{
	${local_indices_split()}
	${indirect_index()}
	${shared_mem_propagation_vars()}
	${load_node_type()}
	Dist d${grid_idx1};
	if (!isPropagationOnly(type)) {
		${guo_density_node_index_shift_intro()}
		float g${grid_idx1}m0 = gg${grid_idx1}m0[gi];
		${sc_calculate_force(grid_idx=grid_idx1)}

		// Cache the distributions in local variables.
		getDist(
			${nodes_array_arg_if_required()}
			&d${grid_idx1}, dist${grid_idx1}_in, gi
			${dense_gi_arg_if_required()}
			${iteration_number_arg_if_required()});

		// Macroscopic quantities for the current cell.
		float v[${dim}];
		v[0] = ovx[gi];
		v[1] = ovy[gi];
		${'v[2] = ovz[gi]' if dim == 3 else ''};

		${guo_density_restore_index()}

		precollisionBoundaryConditions(&d${grid_idx1}, ncode, type, orientation, &g${grid_idx1}m0, v
										${precollision_arguments()}
										${iteration_number_arg_if_required()});
		${relaxate(bgk_args_sc, grid_idx1)}

		// FIXME: In order for the half-way bounce back boundary condition to work, a layer of unused
		// nodes currently has to be placed behind the wall layer.
		postcollisionBoundaryConditions(&d${grid_idx1}, ncode, type, orientation, &g${grid_idx1}m0, v, gi, dist${grid_idx1}_out
										${iteration_number_arg_if_required()});
		${guo_density_node_index_shift_final()}
		${check_invalid_values(grid_idx1)}
	}  // propagation only
	${propagate('dist{}_out'.format(grid_idx1), 'd{}'.format(grid_idx1))}
}
%endfor

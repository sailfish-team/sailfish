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
	float g${grid_idx}m0, float *iv0, float *ea${grid_idx}
</%def>

<%def name="bgk_args_decl_fe(grid_idx)">
	%if grid_idx == 0:
		float rho, float phi, float lap1, float *iv0, float *grad1
	%else:
		${'float rho, ' if phi_needs_rho else ''} float phi, float lap1, float *iv0
	%endif
</%def>

%if simtype == 'shan-chen':
	%for grid_idx in range(0, len(grids)):
		// Relaxation time for fluid component ${grid_idx}
		${const_var} float tau${grid_idx} = ${pageargs['tau{}'.format(grid_idx)]}f;
		${const_var} float tau${grid_idx}_inv = 1.0f / ${pageargs['tau{}'.format(grid_idx)]}f;
	%endfor
%elif simtype == 'free-energy':
	// Relaxation time for the order parameter field.
	${const_var} float tau1 = ${tau_phi}f;
	${const_var} float tau1_inv = 1.0f / ${tau_phi}f;
%endif

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

	%for eq, dist_name in zip([f(g, config) for f, g in zip(equilibria, grids)], ['dist{}_in'.format(grid_idx) for grid_idx in range(0, len(grids))]):
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
	${nodes_array_if_required()}
	${global_ptr} ${const_ptr} int *__restrict__ map,
	%for grid_idx in range(0, len(grids)):
		${global_ptr} float *dist${grid_idx}_in,
	%endfor
	${kernel_args_1st_moment('iv')}
	%if simtype == 'shan-chen':
		%for grid_idx in range(0, len(grids)):
			${global_ptr} ${const_ptr} float *__restrict__ ig${grid_idx}m0${'' if grid_idx == len(grids)-1 else ','}
		%endfor
	%elif simtype == 'free-energy':
		${global_ptr} ${const_ptr} float *__restrict__ irho,
		${global_ptr} ${const_ptr} float *__restrict__ iphi
	%endif
	)
{
	${local_indices()}
	${indirect_index(orig=None)}

	int ncode = map[gi];
	int type = decodeNodeType(ncode);
	if (!isWetNode(type)) {
		%if nt.NTFullBBWall in node_types:
			// Full BB nodes need special treatment as they reintroduce distributions
			// into the simulation domain with a time lag of 2 steps.
			if (!isNTFullBBWall(type)) {
				%for i in range(0, grid.Q):
					%for grid_idx in range(0, len(grids)):
						${get_odist('dist{}_in'.format(grid_idx), i)} = INFINITY;
					%endfor
				%endfor
				return;
			}
		%endif
	}

	// Cache macroscopic fields in local variables.
	%if simtype == 'shan-chen':
		%for grid_idx in range(0, len(grids)):
			float g${grid_idx}m0 = ig${grid_idx}m0[gi];
		%endfor
	%elif simtype == 'free-energy':
			float rho = irho[gi];
			float phi = iphi[gi];
	%endif
	float v0[${dim}];

	v0[0] = ivx[gi];
	v0[1] = ivy[gi];
	%if dim == 3:
		v0[2] = ivz[gi];
	%endif

	${init_dist_with_eq()}
}

%if simtype == 'shan-chen':
<%include file="multi_shan_chen.mako"/>
%elif simtype == 'free-energy':
<%include file="multi_free_energy.mako"/>
%endif

<%include file="util_kernels.mako"/>

<%!
    from sailfish import sym
    import sailfish.node_type as nt
    import sympy
%>

<%def name="bgk_args_decl_sc(grid_idx)">
	%if grid_idx == 0:
		float rho, float *iv0, float *ea${grid_idx}
	%elif grid_idx == 1:
		float phi, float *iv0, float *ea${grid_idx}
	%else:
		float theta, float *iv0, float *ea${grid_idx}
	%endif
</%def>

// TODO(nlooije): incorporate multiple species for FE model

// Relaxation time for the 1st fluid component.
${const_var} float tau0 = ${tau}f;		// relaxation time
${const_var} float tau0_inv = 1.0f / ${tau}f;

// Relaxation time for the 2nd fluid component.
${const_var} float tau1 = ${tau_phi}f;
${const_var} float tau1_inv = 1.0f / ${tau_phi}f;

// Relaxation time for the 3rd fluid component
${const_var} float tau2 = ${tau_theta}f;
${const_var} float tau2_inv = 1.0f / ${tau_theta}f;

<%namespace file="../opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="../kernel_common.mako" import="*" name="kernel_common"/>
%if simtype == 'shan-chen':
	<%namespace file="../shan_chen.mako" import="*" name="shan_chen"/>
	${kernel_common.body(bgk_args_decl_sc)}
	${shan_chen.body()}
%endif
<%namespace file="../mako_utils.mako" import="*"/>
<%namespace file="../boundary.mako" import="*" name="boundary"/>
<%namespace file="../relaxation.mako" import="*" name="relaxation"/>
<%namespace file="../propagation.mako" import="*"/>

<%def name="init_dist_with_eq()">
	%for eq, dist_name in zip([f(g, config) for f, g in zip(equilibria, grids)], ['dist1_in', 'dist2_in', 'dist3_in']):
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
	${global_ptr} float *dist1_in,
	${global_ptr} float *dist2_in,
	${global_ptr} float *dist3_in,
	${kernel_args_1st_moment('iv')}
	${global_ptr} ${const_ptr} float *__restrict__ irho,
	${global_ptr} ${const_ptr} float *__restrict__ iphi,
	${global_ptr} ${const_ptr} float *__restrict__ itheta)
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
					${get_odist('dist1_in', i)} = INFINITY;
					${get_odist('dist2_in', i)} = INFINITY;
					${get_odist('dist3_in', i)} = INFINITY;
				%endfor
				return;
			}
		%endif
	}

	// Cache macroscopic fields in local variables.
	float rho = irho[gi];
	float phi = iphi[gi];
	float theta = itheta[gi];
	float v0[${dim}];

	v0[0] = ivx[gi];
	v0[1] = ivy[gi];
	%if dim == 3:
		v0[2] = ivz[gi];
	%endif

	${init_dist_with_eq()}
}

<%include file="ternary_shan_chen.mako"/>

<%include file="../kernel_utils.mako"/>

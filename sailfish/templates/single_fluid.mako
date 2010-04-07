<%!
    from sailfish import sym
%>

%if 'gravity' in context.keys():
	${const_var} float gravity = ${gravity}f;
%endif

<%def name="bgk_args_decl()">
	float rho, float *iv0
	%if simtype == 'shan-chen':
		, float *ea0
	%endif
</%def>

<%def name="bgk_args()">
	g0m0, v
	%if simtype == 'shan-chen':
		, sca0
	%endif
</%def>

${const_var} float tau0 = ${tau}f;		// relaxation time
${const_var} float visc = ${visc}f;		// viscosity

<%namespace file="kernel_common.mako" import="*" name="kernel_common"/>
${kernel_common.body(bgk_args_decl)}

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>
<%namespace file="propagation.mako" import="*"/>

<%include file="tracers.mako"/>

// A kernel to set the node distributions using the equilibrium distributions
// and the macroscopic fields.
${kernel} void SetInitialConditions(
	${global_ptr} float *dist1_in,
	${kernel_args_1st_moment('iv')}
	${global_ptr} float *irho)
{
	${local_indices()}

	// Cache macroscopic fields in local variables.
	float rho = irho[gi];
	float v0[${dim}];

	v0[0] = ivx[gi];
	v0[1] = ivy[gi];
	%if dim == 3:
		v0[2] = ivz[gi];
	%endif

	%for local_var in bgk_equilibrium_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
	%endfor

	%for i, (feq, idx) in enumerate(bgk_equilibrium[0]):
		${get_odist('dist1_in', i)} = ${cex(feq, vectors=True)};
	%endfor
}

${kernel} void PrepareMacroFields(
	${global_ptr} int *map,
	${global_ptr} float *dist1_in,
	${global_ptr} float *orho)
{
	${local_indices()}

	int type, orientation;
	decodeNodeType(map[gi], &orientation, &type);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type))
		return;

	Dist fi;
	float out;

	getDist(&fi, dist1_in, gi);
	get0thMoment(&fi, type, orientation, &out);
	orho[gi] = out;
}

${kernel} void CollideAndPropagate(
	${global_ptr} int *map,
	${global_ptr} float *dist_in,
	${global_ptr} float *dist_out,
	${global_ptr} float *orho,
	${kernel_args_1st_moment('ov')}
	int save_macro
%if simtype == 'shan-chen':
	,${global_ptr} float *gg0m0
%endif
	)
{
	${local_indices()}

	// shared variables for in-block propagation
	%for i in sym.get_prop_dists(grid, 1):
		${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
	%endfor
	%for i in sym.get_prop_dists(grid, 1):
		#define prop_${grid.idx_name[grid.idx_opposite[i]]} prop_${grid.idx_name[i]}
	%endfor

	int type, orientation;
	decodeNodeType(map[gi], &orientation, &type);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type))
		return;

	// cache the distributions in local variables
	Dist d0;
	getDist(&d0, dist_in, gi);

	%if simtype == 'shan-chen':
		${sc_calculate_accel()}
	%endif

	// macroscopic quantities for the current cell
	float g0m0, v[${dim}];

	%if simtype == 'shan-chen':
		${sc_macro_fields()}
	%else:
		getMacro(&d0, type, orientation, &g0m0, v);
	%endif

	boundaryConditions(&d0, type, orientation, &g0m0, v);

	${relaxate(bgk_args)}

	// only save the macroscopic quantities if requested to do so
	if (save_macro == 1) {
		orho[gi] = g0m0;
		ovx[gi] = v[0];
		ovy[gi] = v[1];
		%if dim == 3:
			ovz[gi] = v[2];
		%endif
	}

	${propagate('dist_out', 'd0')}
}


<%!
    from sailfish import sym
%>

%if 'gravity' in context.keys():
	${const_var} float gravity = ${gravity}f;
%endif

<%def name="bgk_args_decl()">
	float rho, float *v0
</%def>

<%def name="bgk_args()">
	rho, phi
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

${kernel} void CollideAndPropagate(
	${global_ptr} int *map,
	${global_ptr} float *dist_in,
	${global_ptr} float *dist_out,
	${global_ptr} float *orho,
	${kernel_args_1st_moment('ov')}
	int save_macro)
{
	${local_indices()}

	// shared variables for in-block propagation
	%for i in sym.get_prop_dists(grid, 1):
		${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
	%endfor
	%for i in sym.get_prop_dists(grid, -1):
		${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
	%endfor

	int type, orientation;
	decodeNodeType(map[gi], &orientation, &type);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type))
		return;

	// cache the distributions in local variables
	Dist d1;
	getDist(&d1, dist_in, gi);

	// macroscopic quantities for the current cell
	float rho, v[${dim}];

	getMacro(&d1, type, orientation, &rho, v);
	boundaryConditions(&d1, type, orientation, &rho, v);
	${barrier()}

	// only save the macroscopic quantities if requested to do so
	if (save_macro == 1) {
		orho[gi] = rho;
		ovx[gi] = v[0];
		ovy[gi] = v[1];
		%if dim == 3:
			ovz[gi] = v[2];
		%endif
	}

	${relaxate(bgk_args)}
	${propagate('dist_out', 'd1')}
}


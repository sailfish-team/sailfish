<%!
    from sailfish import sym
%>

<%namespace file="code_common.mako" import="*"/>

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

<%def name="get_boundary_velocity(node_type, mx, my, mz, rho=0, moments=False)">
	int idx = (${node_type} - GEO_BCV) * ${dim};
	%if moments:
		${mx} = geo_params[idx] * ${rho};
		${my} = geo_params[idx+1] * ${rho};
		%if dim == 3:
			${mz} = geo_params[idx+2] * ${rho};
		%endif
	%else:
		${mx} = geo_params[idx];
		${my} = geo_params[idx+1];
		%if dim == 3:
			${mz} = geo_params[idx+2];
		%endif
	%endif
</%def>

<%def name="get_boundary_pressure(node_type, rho)">
	int idx = (GEO_BCP-GEO_BCV) * ${dim} + (${node_type} - GEO_BCP);
	${rho} = geo_params[idx] * 3.0f;
</%def>

<%def name="get_boundary_params(node_type, mx, my, mz, rho, moments=False)">
	if (${node_type} >= GEO_BCV) {
		// Velocity boundary condition.
		if (${node_type} < GEO_BCP) {
			${get_boundary_velocity(node_type, mx, my, mz, rho, moments)}
		// Pressure boundary condition.
		} else {
			${get_boundary_pressure(node_type, rho)}
		}
	}
</%def>

<%def name="fill_missing_distributions()">
	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			case ${i}: {
				%for lvalue, rvalue in sym.fill_missing_dists(grid, 'fi', missing_dir=i):
					${lvalue} = ${rvalue};
				%endfor
				break;
			}
		%endfor
	}
</%def>

<%def name="external_force(node_type, vx, vy, vz, rho=0, momentum=False)">
	%if ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0:
		if (!isWallNode(${node_type})) {
			%if momentum:
				%if ext_accel_x != 0.0:
					${vx} += ${rho} * ${'%.20ff' % (0.5 * ext_accel_x)};
				%endif
				%if ext_accel_y != 0.0:
					${vy} += ${rho} * ${'%.20ff' % (0.5 * ext_accel_y)};
				%endif
				%if dim == 3 and ext_accel_z != 0.0:
					${vz} += ${rho} * ${'%.20ff' % (0.5 * ext_accel_z)};
				%endif
			%else:
				%if ext_accel_x != 0.0:
					${vx} += ${'%.20ff' % (0.5 * ext_accel_x)};
				%endif
				%if ext_accel_y != 0.0:
					${vy} += ${'%.20ff' % (0.5 * ext_accel_y)};
				%endif
				%if dim == 3 and ext_accel_z != 0.0:
					${vz} += ${'%.20ff' % (0.5 * ext_accel_z)};
				%endif
			%endif
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

${device_func} inline void compute_macro_quant(Dist *fi, float *rho, float *v)
{
	*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
	%for d in range(0, grid.dim):
		v[${d}] = ${cex(sym.ex_velocity(grid, 'fi', d), pointers=True)};
	%endfor
}

%if bc_wall == 'zouhe' or bc_velocity == 'zouhe' or bc_pressure == 'zouhe':
${device_func} void zouhe_bb(Dist *fi, int orientation, float *rho, float *v)
{
	// Bounce-back of the non-equilibrium parts.
	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			${noneq_bb(i)}
		%endfor
		case ${geo_dir_other}:
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
	nvx = *rho * v[0] - nvx;
	nvy = *rho * v[1] - nvy;
	%if dim == 3:
		nvz = *rho * v[2] - nvz;
	%endif

	switch (orientation) {
		%for i in range(1, grid.dim*2+1):
			${zouhe_fixup(i)}
		%endfor
	}
}
%endif

//
// Get macroscopic density rho and velocity v given a distribution fi, and
// the node class node_type.
//
${device_func} inline void getMacro(Dist *fi, int node_type, int orientation, float *rho, float *v)
{
	if (isFluidOrWallNode(node_type) || orientation == ${geo_dir_other}) {
		compute_macro_quant(fi, rho, v);
		if (isWallNode(node_type)) {
			v[0] = 0.0f;
			v[1] = 0.0f;
			%if dim == 3:
				v[2] = 0.0f;
			%endif
			%if bc_wall == 'zouhe':
				zouhe_bb(fi, orientation, rho, v);
			%endif
		}
	} else if (isVelocityNode(node_type)) {
		%if bc_velocity == 'zouhe':
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			${get_boundary_velocity('node_type', 'v[0]', 'v[1]', 'v[2]')}
			zouhe_bb(fi, orientation, rho, v);
		// We're dealing with a boundary node, for which some of the distributions
		// might be meaningless.  Fill them with the values of the opposite
		// distributions.
		%elif bc_velocity == 'equilibrium':
			${fill_missing_distributions()}
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			${get_boundary_velocity('node_type', 'v[0]', 'v[1]', 'v[2]')}

			switch (orientation) {
				%for i in range(1, grid.dim*2+1):
					case ${i}:
						*rho = ${cex(sym.ex_rho(grid, 'fi', incompressible, missing_dir=i), pointers=True)};
						break;
				%endfor
			}
		%else:
			compute_macro_quant(fi, rho, v);
		%endif
	} else if (isPressureNode(node_type)) {
		%if bc_pressure == 'zouhe' or bc_pressure == 'equilibrium':
			${fill_missing_distributions()}
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
			float par_rho;
			${get_boundary_pressure('node_type', 'par_rho')}

			switch (orientation) {
				%for i in range(1, grid.dim*2+1):
					case ${i}: {
						%for d in range(0, grid.dim):
							v[${d}] = ${cex(sym.ex_velocity(grid, 'fi', d, missing_dir=i, par_rho='par_rho'), pointers=True)};
						%endfor
						break;
					 }
				%endfor
			}

			%if bc_pressure == 'zouhe':
				zouhe_bb(fi, orientation, &par_rho, v);
				compute_macro_quant(fi, rho, v);
			%endif
			*rho = par_rho;
		%else:
			compute_macro_quant(fi, rho, v);
		%endif
	}

	${external_force('node_type', 'v[0]', 'v[1]', 'v[2]')}
}

${device_func} inline void boundaryConditions(Dist *fi, int node_type, int orientation, float *rho, float *v)
{
	%if bc_wall == 'fullbb':
		if (isWallNode(node_type)) {
			bounce_back(fi);
		}
	%endif

	%if bc_velocity == 'fullbb':
		if (isVelocityNode(node_type)) {
			bounce_back(fi);
			${get_boundary_velocity('node_type', 'v[0]', 'v[1]', 'v[2]')}
			%for i, ve in enumerate(grid.basis):
				fi->${grid.idx_name[i]} += ${cex(
					grid.rho0 * 2 * grid.weights[i] * grid.v.dot(ve) / grid.cssq, pointers=True)};
			%endfor
			*rho = ${sym.ex_rho(grid, 'fi', incompressible)};
		}
	%endif

	%if bc_velocity == 'equilibrium':
		if (isVelocityNode(node_type)) {
			%for feq, idx in bgk_equilibrium:
				fi->${idx} = ${cex(feq, pointers=True)};
			%endfor
		}
	%endif

	%if bc_pressure == 'equilibrium':
		if (isPressureNode(node_type)) {
			%for feq, idx in bgk_equilibrium:
				fi->${idx} = ${cex(feq, pointers=True)};
			%endfor
		}
	%endif
}


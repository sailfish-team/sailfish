<%!
    from sailfish import sym
%>

<%page args="bgk_args_decl"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="external_force"/>

% if model == 'mrt':
//
// Relaxation in moment space.
//
${device_func} void MS_relaxate(Dist *fi, int node_type)
{
	DistM fm, feq;

	%for mrt, val in sym.bgk_to_mrt(grid, 'fi', 'fm'):
		${mrt} = ${val};
	%endfor

	${external_force('node_type', 'fm.mx', 'fm.my', 'fm.mz', 'fm.rho', momentum=True)}

	#define mx fm.mx
	#define my fm.my
	#define mz fm.mz

	// Calculate equilibrium distributions in moment space.
	%for i, eq in enumerate(grid.mrt_equilibrium):
		%if grid.mrt_collision[i] != 0:
			feq.${grid.mrt_names[i]} = ${cex(eq, rho='fm.rho')};
		%endif
	%endfor

	// Relexate the non-conserved moments,
	%if bc_velocity == 'equilibrium':
		if (isVelocityNode(node_type)) {
			%for i, coll in enumerate(grid.mrt_collision):
				%if coll != 0:
					fm.${grid.mrt_names[i]} = feq.${grid.mrt_names[i]};
				%endif
			%endfor
		}
	%endif

	%if bc_pressure == 'equilibrium':
		if (isPressureNode(node_type)) {
			%for i, coll in enumerate(grid.mrt_collision):
				%if coll != 0:
					fm.${grid.mrt_names[i]} = feq.${grid.mrt_names[i]};
				%endif
			%endfor
		}
	%endif

	%for i, name in enumerate(grid.mrt_names):
		%if grid.mrt_collision[i] != 0:
			fm.${name} -= ${sym.make_float(grid.mrt_collision[i])} * (fm.${name} - feq.${name});
		%endif
	%endfor

	#undef mx
	#undef my
	#undef mz

	${external_force('node_type', 'fm.mx', 'fm.my', 'fm.mz', 'fm.rho', momentum=True)}

	%for bgk, val in sym.mrt_to_bgk(grid, 'fi', 'fm'):
		${bgk} = ${val};
	%endfor
}
% endif	## model == mrt

% if model == 'bgk':
//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
${device_func} void BGK_relaxate(${bgk_args_decl()},
%for i in range(0, len(grids)):
	Dist *d${i},
%endfor
	 int node_type)
{
	%for i in range(0, len(grids)):
		Dist feq${i};
	%endfor

	%for local_var in bgk_equilibrium_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
	%endfor

	%for i, eq in enumerate(bgk_equilibrium):
		%for feq, idx in eq:
			feq${i}.${idx} = ${cex(feq, vectors=True)};
		%endfor
	%endfor

	%for i in range(0, len(grids)):
		%for idx in grid.idx_name:
			d${i}->${idx} += (feq${i}.${idx} - d${i}->${idx}) / tau${i};
		%endfor

		%if ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0:
			if (!isWallNode(node_type))
			{
				// External acceleration.
				#define eax ${'%.20ff' % ext_accel_x}
				#define eay ${'%.20ff' % ext_accel_y}
				#define eaz ${'%.20ff' % ext_accel_z}
				float pref = ${sym.bgk_external_force_pref()};

				%for val, idx in sym.bgk_external_force(grid):
					d${i}->${idx} += ${cex(val)};
				%endfor
			}
		%endif
	%endfor
}
%endif

<%def name="_relaxate(bgk_args)">
	%if model == 'bgk':
		BGK_relaxate(${bgk_args()},
%for i in range(0, len(grids)):
	&d${i},
%endfor
	type);
	%else:
		MS_relaxate(&d1, type);
	%endif
</%def>

## TODO: This could be optimized.
<%def name="relaxate(bgk_args)">
	if (isFluidNode(type)) {
		${_relaxate(bgk_args)}
	}
	%if bc_wall_.wet_nodes:
		else if (isWallNode(type)) {
			${_relaxate(bgk_args)}
		}
	%endif
	%if bc_velocity_.wet_nodes:
		else if (isVelocityNode(type)) {
			${_relaxate(bgk_args)}
		}
	%endif
	%if bc_pressure_.wet_nodes:
		else if (isPressureNode(type)) {
			${_relaxate(bgk_args)}
		}
	%endif
</%def>

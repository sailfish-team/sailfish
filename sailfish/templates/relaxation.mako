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

%if model == 'femrt':
${device_func} inline void FE_MRT_relaxate(${bgk_args_decl()},
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

	float tau0 = tau_b + (phi + 1.0f) * (tau_a - tau_b) / 2.0f;

	if (phi < -1.0f) {
		tau0 = tau_b;
	} else if (phi > 1.0f) {
		tau0 = tau_a;
	}

	%for i, eq in enumerate(bgk_equilibrium):
		%if (ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0) and i == 1:
			%for j in range(0, dim):
				v0[${j}] += 0.5f * ea0[${j}] / rho;
			%endfor
		%endif

		%for feq, idx in eq:
			feq${i}.${idx} = ${cex(feq, vectors=True)};
		%endfor
	%endfor

	#define ea1 ea0

	%for i in range(0, len(grids)):
		%for idx in grid.idx_name:
			%if i == 1:
				d${i}->${idx} += (feq${i}.${idx} - d${i}->${idx}) / tau${i};
			%else:
				feq${i}.${idx} = d${i}->${idx} - feq${i}.${idx};
			%endif
		%endfor
	%endfor

	%for dst, src in sym.free_energy_mrt(sim.grid, 'd0', 'feq0'):
		${dst} -= ${sym.make_float(src)};
	%endfor

	%if (ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0):
		if (!isWallNode(node_type))
		{
			%for val, idx in sym.free_energy_external_force(sim, grid_num=0):
				d0->${idx} += ${cex(val, vectors=True)};
			%endfor
		}
	%endif
}
%endif  ## model == femrt

% if model == 'bgk':
//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
${device_func} inline void BGK_relaxate(${bgk_args_decl()},
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

	%if simtype == 'free-energy':
		float tau0 = tau_b + (phi + 1.0f) * (tau_a - tau_b) / 2.0f;
		if (phi < -1.0f) {
			tau0 = tau_b;
		} else if (phi > 1.0f) {
			tau0 = tau_a;
		}
	%endif

	%for i, eq in enumerate(bgk_equilibrium):
		%if simtype == 'shan-chen':
			%for j in range(0, dim):
				v0[${j}] += tau${i} * ea${i}[${j}];
			%endfor
		%elif simtype == 'free-energy':
			%if (ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0) and i == 1:
				%for j in range(0, dim):
					v0[${j}] += 0.5f * ea0[${j}] / rho;
				%endfor
			%endif
		%endif

		%for feq, idx in eq:
			feq${i}.${idx} = ${cex(feq, vectors=True)};
		%endfor

		%if simtype == 'shan-chen':
			%for j in range(0, dim):
				v0[${j}] -= tau${i} * ea${i}[${j}];
			%endfor
		%endif
	%endfor

	%for i in range(0, len(grids)):
		%for idx in grid.idx_name:
			d${i}->${idx} += (feq${i}.${idx} - d${i}->${idx}) / tau${i};
		%endfor

		%if simtype == 'shan-chen':
			if (!isWallNode(node_type))
			{
				float pref = ${sym.bgk_external_force_pref(grid_num=i)};
				%for j in range(0, dim):
					v0[${j}] += 0.5f * ea${i}[${j}];
				%endfor

				%for val, idx in sym.bgk_external_force(grid, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor

				%for j in range(0, dim):
					v0[${j}] -= 0.5f * ea${i}[${j}];
				%endfor
			}
		%elif simtype == 'free-energy':
			%if (ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0) and i == 0:
				if (!isWallNode(node_type))
				{
					%for val, idx in sym.free_energy_external_force(sim, grid_num=i):
						d${i}->${idx} += ${cex(val, vectors=True)};
					%endfor
				}
			%endif
		%else:
			%if (ext_accel_x != 0.0 or ext_accel_y != 0.0 or ext_accel_z != 0.0):
				if (!isWallNode(node_type))
				{
					float pref = ${sym.bgk_external_force_pref(grid_num=i)};
					%for val, idx in sym.bgk_external_force(grid, grid_num=i):
						d${i}->${idx} += ${cex(val, vectors=True)};
					%endfor
				}
			%endif
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
	%elif model == 'femrt':
		FE_MRT_relaxate(${bgk_args()},
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

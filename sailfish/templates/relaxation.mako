<%!
    from sailfish import sym
%>

<%page args="bgk_args_decl"/>
<%namespace file="code_common.mako" import="*"/>

<%def name="fluid_momentum(igrid)">
	%if igrid in force_for_eq and equilibrium:
		fm.mx += ${cex(0.5 * sym.fluid_accel(sim, force_for_eq[igrid], 0, forces, force_couplings), vectors=True)};
		fm.my += ${cex(0.5 * sym.fluid_accel(sim, force_for_eq[igrid], 1, forces, force_couplings), vectors=True)};
		%if dim == 3:
			fm.mz += ${cex(0.5 * sym.fluid_accel(sim, force_for_eq[igrid], 2, forces, force_couplings), vectors=True)};
		%endif
	%else:
		fm.mx += ${cex(0.5 * sym.fluid_accel(sim, igrid, 0, forces, force_couplings), vectors=True)};
		fm.my += ${cex(0.5 * sym.fluid_accel(sim, igrid, 1, forces, force_couplings), vectors=True)};
		%if dim == 3:
			fm.mz += ${cex(0.5 * sym.fluid_accel(sim, igrid, 2, forces, force_couplings), vectors=True)};
		%endif
	%endif
</%def>

<%def name="body_force()">
	// Body force acceleration.
	%for i in range(0, len(grids)):
		%if sym.needs_accel(i, forces, force_couplings):
			%if not sym.needs_coupling_accel(i, force_couplings):
				float ea${i}[${dim}];
				%for j in range(0, dim):
					ea${i}[${j}] = ${cex(sym.body_force_accel(i, j, forces), vectors=True)};
				%endfor
			%else:
				%for j in range(0, dim):
					ea${i}[${j}] += ${cex(sym.body_force_accel(i, j, forces), vectors=True)};
				%endfor
			%endif
		%endif
	%endfor
</%def>


## TODO: support multiple grids with MRT?
% if model == 'mrt' and simtype == 'fluid':
//
// Relaxation in moment space.
//
${device_func} void MS_relaxate(Dist *fi, int node_type, float *iv0)
{
	DistM fm, feq;

	%for mrt, val in sym.bgk_to_mrt(grid, 'fi', 'fm'):
		${mrt} = ${val};
	%endfor

	${body_force()}
	${fluid_momentum(0)}

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

	${fluid_momentum(0)}

	%for bgk, val in sym.mrt_to_bgk(grid, 'fi', 'fm'):
		${bgk} = ${val};
	%endfor

	if (!isWallNode(node_type)) {
		${fluid_velocity(0, save=True)}
	}
}
% endif	## model == mrt

<%def name="fluid_velocity(igrid, equilibrium=False, save=False)">
	%if save:
		## For now, we assume a single fluid velocity regardless of the number of
		## components and grids used.  If this assumption is ever changed, the code
		## below will have to be extended appropriately.
		%for j in range(0, dim):
			iv0[${j}] += ${cex(0.5 * sym.fluid_accel(sim, igrid, j, forces, force_couplings), vectors=True)};
		%endfor
	%else:
		%for j in range(0, dim):
			%if igrid in force_for_eq and equilibrium:
				v0[${j}] = iv0[${j}] + ${cex(0.5 * sym.fluid_accel(sim, force_for_eq[igrid], j, forces, force_couplings), vectors=True)};
			%else:
				v0[${j}] = iv0[${j}] + ${cex(0.5 * sym.fluid_accel(sim, igrid, j, forces, force_couplings), vectors=True)};
			%endif
		%endfor
	%endif
</%def>

<%def name="bgk_relaxation_preamble()">
	%for i in range(0, len(grids)):
		Dist feq${i};
	%endfor

	%for local_var in bgk_equilibrium_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
	%endfor

	float v0[${dim}];
	${body_force()}

	%if simtype == 'free-energy':
		float tau0 = tau_b + (phi + 1.0f) * (tau_a - tau_b) / 2.0f;
		if (phi < -1.0f) {
			tau0 = tau_b;
		} else if (phi > 1.0f) {
			tau0 = tau_a;
		}
	%endif

	%for i, eq in enumerate(bgk_equilibrium):
		${fluid_velocity(i, True)};

		%for feq, idx in eq:
			feq${i}.${idx} = ${cex(feq, vectors=True)};
		%endfor
	%endfor
</%def>

%if model == 'femrt' and simtype == 'free-energy':
${device_func} inline void FE_MRT_relaxate(${bgk_args_decl()},
%for i in range(0, len(grids)):
	Dist *d${i},
%endfor
	int node_type)
{
	${bgk_relaxation_preamble()}

	%for i in range(0, len(grids)):
		%for idx in grid.idx_name:
			## Use the BGK approximation for the relaxation of the order parameter field.
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

	%for i in range(0, len(grids)):
		## Is there a force acting on the current grid?
		%if sym.needs_accel(i, forces, force_couplings):
			if (!isWallNode(node_type)) {
				${fluid_velocity(i)};

				%for val, idx in sym.free_energy_external_force(sim, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			}
		%endif
	%endfor

	if (!isWallNode(node_type)) {
		${fluid_velocity(0, save=True)}
	}
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
	${bgk_relaxation_preamble()}

	%for i in range(0, len(grids)):
		%for idx in grid.idx_name:
			d${i}->${idx} += (feq${i}.${idx} - d${i}->${idx}) / tau${i};
		%endfor

		## Is there a force acting on the current grid?
		%if sym.needs_accel(i, forces, force_couplings):
			if (!isWallNode(node_type)) {
				${fluid_velocity(i)};

				%if simtype == 'shan-chen':
					float pref = ${sym.bgk_external_force_pref(grid_num=i)};
					%for val, idx in sym.bgk_external_force(grid, grid_num=i):
						d${i}->${idx} += ${cex(val, vectors=True)};
					%endfor
				%elif simtype == 'free-energy':
					%for val, idx in sym.free_energy_external_force(sim, grid_num=i):
						d${i}->${idx} += ${cex(val, vectors=True)};
					%endfor
				%else:
					float pref = ${sym.bgk_external_force_pref(grid_num=i)};
					%for val, idx in sym.bgk_external_force(grid, grid_num=i):
						d${i}->${idx} += ${cex(val, vectors=True)};
					%endfor
				%endif
			}
		%endif
	%endfor

	if (!isWallNode(node_type)) {
		${fluid_velocity(0, save=True)}
	}
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
		MS_relaxate(&d0, type, v);
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

<%!
    from sailfish import sym
%>

<%page args="bgk_args_decl"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="boundary.mako" import="get_boundary_pressure"/>

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

<%def name="body_force(accel=True, need_vars_declaration=True, grid_id=None)">
	%if accel:
		// Body force acceleration.
	%else:
		// Body force.
	%endif
	%for i in range(0, len(grids)):
		%if (grid_id is None or grid_id == i) and sym.needs_accel(i, forces, force_couplings):
			%if not sym.needs_coupling_accel(i, force_couplings):
				%if need_vars_declaration:
					float ea${i}[${dim}];
				%endif
				%for j in range(0, dim):
					ea${i}[${j}] = ${cex(sym.body_force_accel(i, j, forces, accel=accel), vectors=True)};
				%endfor
			%else:
				%for j in range(0, dim):
					ea${i}[${j}] += ${cex(sym.body_force_accel(i, j, forces, accel=accel), vectors=True)};
				%endfor
			%endif
		%endif
	%endfor
</%def>


## TODO: support multiple grids with MRT?
% if model == 'mrt' and simtype == 'lbm':
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

	${fluid_velocity(0, save=True)}
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

	%if subgrid == 'les-smagorinsky':
		// Compute relaxation time using the standard viscosity-relaxation time relation.
		%for i in range(0, len(grids)):
			float tau${i} = 0.5f + 3.0f * visc;
		%endfor

		## FIXME: This will not work properly for multifluid models.
		// Modify the relaxation time proportionally to the modulus of the local strain rate tensor.
		// The 2nd order tensor formed from the equilibrium distributions is rho / 3 * \delta_{ab} + rho u_a u_b
		{
			float tmp, strain;

			%for i in range(0, len(grids)):
				strain = 0.0f;

				// Off-diagonal components, count twice for symmetry reasons.
				%for a in range(0, dim):
					%for b in range(a+1, dim):
						 tmp = ${cex(sym.ex_flux(grid, 'd%d' % i, a, b), pointers=True)} - ${cex(sym.S.rho * grid.v[a] * grid.v[b], vectors=True)};
						 strain += 2.0f * tmp * tmp;
					%endfor
				%endfor

				// Diagonal components.
				%for a in range(0, dim):
					tmp = ${cex(sym.ex_flux(grid, 'd%d' % i, a, a), pointers=True)} - ${cex(sym.S.rho * (grid.v[a] * grid.v[b] + grid.cssq), vectors=True)};
					strain += tmp * tmp;
				%endfor

				// Form of the relaxation time correction as in comp-gas/9401004v1.
				tau${i} += (sqrtf(visc*visc + 18.0f * ${cex(smagorinsky_const**2)} * sqrtf(strain)) - visc) / 2.0f;
			%endfor
		}
	%endif
</%def>

%if model == 'mrt' and simtype == 'free-energy':
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
			${fluid_velocity(i)};

			%for val, idx in sym.free_energy_external_force(sim, grid_num=i):
				d${i}->${idx} += ${cex(val, vectors=True)};
			%endfor
		%endif
	%endfor

	${fluid_velocity(0, save=True)}
}
%endif  ## model == mrt && simtype == 'free-energy'

% if model == 'bgk':
//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
${device_func} inline void BGK_relaxate(${bgk_args_decl()},
%for i in range(0, len(grids)):
	Dist *d${i},
%endfor
	int node_type, int ncode)
{
	${bgk_relaxation_preamble()}

	%for i in range(0, len(grids)):
		%for idx in grid.idx_name:
			d${i}->${idx} += (feq${i}.${idx} - d${i}->${idx}) / tau${i};
		%endfor

		%if bc_pressure == 'guo':
			// The total form of the postcollision boundary node distribution value
			// with the Guo boundary conditions is as follows:
			//
			// f_post(O) = f_eq(O) + f(B) - f_eq(B) + omega * (f_eq(B) - f(B))
			//
			// where O is the boundary node and B is the fluid node pointed to by the
			// boundary node normal vector.  The Guo boudary condtiions are implemented
			// so that all the standard processing proceeds for node B first, and the
			// correction for node O is added as a postcollision boundary condition.
			//
			// The code below takes care of the -f_eq(B) of the formula.
			if (isPressureNode(node_type)) {
				%for idx in grid.idx_name:
					d${i}->${idx} -= feq${i}.${idx};
				%endfor
			}
		%endif

		## Is there a force acting on the current grid?
		%if sym.needs_accel(i, forces, force_couplings):
			${fluid_velocity(i)};
			${body_force(accel=False, need_vars_declaration=False, grid_id=i)}

			%if simtype == 'shan-chen':
			{
				float pref = ${sym.bgk_external_force_pref(grid_num=i)};
				%for val, idx in sym.bgk_external_force(grid, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			}
			%elif simtype == 'free-energy':
				%for val, idx in sym.free_energy_external_force(sim, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			%else:
			{
				float pref = ${sym.bgk_external_force_pref(grid_num=i)};
				%for val, idx in sym.bgk_external_force(grid, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			}
			%endif
		%endif
	%endfor

	${body_force(need_vars_declaration=False)}

	// FIXME: This should be moved to postcollision boundary conditions.
	%if bc_pressure == 'guo':
		if (isPressureNode(node_type)) {
			int node_param = decodeNodeParam(ncode);
			float par_rho;
			${get_boundary_pressure('node_param', 'par_rho')}
			float par_phi = 1.0f;

			%for local_var in bgk_equilibrium_vars:
				float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True, rho='par_rho', phi='par_phi')};
			%endfor

			tau0 = tau_a;

			%for i, eq in enumerate(bgk_equilibrium):
				%for feq, idx in eq:
					d${i}->${idx} += ${cex(feq, vectors=True, rho='par_rho', phi='par_phi')};
				%endfor
			%endfor
		}
	%endif

	${fluid_velocity(0, save=True)}
}
%endif

<%def name="_relaxate(bgk_args)">
	%if model == 'bgk':
		BGK_relaxate(${bgk_args()},
%for i in range(0, len(grids)):
	&d${i},
%endfor
	type, ncode);
	%elif model == 'mrt' and simtype == 'free-energy':
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
	%if relaxation_enabled:
		if (isWetNode(type)) {
			${_relaxate(bgk_args)}
		}
	%endif
</%def>

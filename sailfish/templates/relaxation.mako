<%!
  from sailfish import sym, sym_force
  import sailfish.node_type as nt
%>

<%page args="bgk_args_decl"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="utils.mako" import="*"/>

<%def name="fluid_momentum(igrid)">
	%if forces is not UNDEFINED:
		%if igrid in force_for_eq and equilibrium:
			fm.mx += ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], 0, forces, force_couplings), vectors=True)};
			fm.my += ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], 1, forces, force_couplings), vectors=True)};
			%if dim == 3:
				fm.mz += ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], 2, forces, force_couplings), vectors=True)};
			%endif
		%else:
			fm.mx += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, 0, forces, force_couplings), vectors=True)};
			fm.my += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, 1, forces, force_couplings), vectors=True)};
			%if dim == 3:
				fm.mz += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, 2, forces, force_couplings), vectors=True)};
			%endif
		%endif
	%endif
</%def>


## Defines the actual force/acceleration vectors.
<%def name="body_force(accel=True)">
	%if forces is not UNDEFINED and (forces.numeric or forces.symbolic):
		%if accel:
			// Body force acceleration.
		%else:
			// Body force.
		%endif
		%if forces.symbolic and time_dependence:
			float phys_time = get_time_from_iteration(iteration_number);
		%endif
		%for i in range(0, len(grids)):
			%if sym_force.needs_accel(i, forces, {}):
				%if not sym_force.needs_coupling_accel(i, force_couplings):
					float ea${i}[${dim}];
					%for j in range(0, dim):
						ea${i}[${j}] = ${cex(sym_force.body_force_accel(i, j, forces, accel=accel), vectors=True)};
					%endfor
				%else:
					## If the current grid has a Shan-Chen force acting on it, the acceleration vector
					## is already externally defined in the Shan-Chen code.
					%for j in range(0, dim):
						%if i in forces.symbolic or i in forces.numeric:
							ea${i}[${j}] += ${cex(sym_force.body_force_accel(i, j, forces, accel=accel), vectors=True)};
						%endif
					%endfor
				%endif
			%endif
		%endfor
	%endif
</%def>


## TODO: support multiple grids with MRT?
% if model == 'mrt' and simtype == 'lbm':
//
// Relaxation in moment space.
//
${device_func} void MS_relaxate(Dist *fi, int node_type, float *iv0 ${dynamic_val_args_decl()})
{
	DistM fm, feq;

	%for mrt, val in sym.bgk_to_mrt(grid, 'fi', 'fm'):
		${mrt} = ${val};
	%endfor

	${body_force(accel=True)}
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
	%if nt.NTEquilibriumVelocity in node_types:
		if (isNTEquilibriumVelocity(node_type)) {
			%for i, coll in enumerate(grid.mrt_collision):
				%if coll != 0:
					fm.${grid.mrt_names[i]} = feq.${grid.mrt_names[i]};
				%endif
			%endfor
		}
	%endif

	%if nt.NTEquilibriumDensity in node_types:
		if (isNTEquilibriumDensity(node_type)) {
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

## TODO(michalj): Split this function into two.
## igrid: grid ID
## equilibrium: if True, calculate velocity for the equilibrium distribution
## save: if True, calculate velocity for the simulation output
<%def name="fluid_velocity(igrid, equilibrium=False, save=False)">
	%if forces is not UNDEFINED:
		%if save:
			## For now, we assume a single fluid velocity regardless of the number of
			## components and grids used.  If this assumption is ever changed, the code
			## below will have to be extended appropriately.
			%for j in range(0, dim):
				iv0[${j}] += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, j, forces, force_couplings), vectors=True)};
			%endfor
		%else:
			%for j in range(0, dim):
				%if igrid in force_for_eq and equilibrium:
					v0[${j}] = iv0[${j}] + ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], j, forces, force_couplings), vectors=True)};
				%else:
					v0[${j}] = iv0[${j}] + ${cex(0.5 * sym_force.fluid_accel(sim, igrid, j, forces, force_couplings), vectors=True)};
				%endif
			%endfor
		%endif
	%else:
		%if not save:
			%for j in range(0, dim):
				v0[${j}] = iv0[${j}];
			%endfor
		%endif
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
	${body_force(accel=True)}

	%if simtype == 'free-energy':
		float tau0 = tau_b + (phi + 1.0f) * (tau_a - tau_b) / 2.0f;
		if (phi < -1.0f) {
			tau0 = tau_b;
		} else if (phi > 1.0f) {
			tau0 = tau_a;
		}
	%endif

	%for i, eq in enumerate(bgk_equilibrium):
		${fluid_velocity(i, equilibrium=True)};

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
	int node_type
	${dynamic_val_args_decl()})
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
		%if sym_force.needs_accel(i, forces, force_couplings):
			${fluid_velocity(i)};

			%for val, idx in sym_force.free_energy_external_force(sim, grid_num=i):
				d${i}->${idx} += ${cex(val, vectors=True)};
			%endfor
		%endif
	%endfor

	${fluid_velocity(0, save=True)}
}
%endif  ## model == mrt && simtype == 'free-energy'

%if model == 'elbm':
<%include file="entropic.mako"/>

${device_func} inline void ELBM_relaxate(${bgk_args_decl()}, Dist* d0
	${dynamic_val_args_decl()}
%if alpha_output:
	, int options,
	${global_ptr} float* alpha_out
%endif
)
{
	%for i in range(0, len(grids)):
		Dist fneq${i};
	%endfor

	<%
		if grid is sym.D3Q15:
			elbm_eq, elbm_eq_vars = sym.elbm_d3q15_equilibrium(grid)
		else:
			elbm_eq, elbm_eq_vars = sym.elbm_equilibrium(grid)
	%>

	float v0[${dim}];
	${body_force()}

	${fluid_velocity(0, True)};

	%for local_var in elbm_eq_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
	%endfor

	%for i, eq in enumerate(elbm_eq):
		%for feq, idx in eq:
			fneq${i}.${idx} = ${cex(feq, vectors=True)} - d0->${idx};
		%endfor
	%endfor

	float alpha;

	if (SmallEquilibriumDeviation(d0, &fneq0)) {
		alpha = EstimateAlphaSeries(d0, &fneq0);
	} else {
		alpha = EstimateAlphaFromEntropy(d0, &fneq0);
	}

	%if model == 'elbm' and alpha_output:
		if (options & OPTION_SAVE_MACRO_FIELDS) {
			*alpha_out = alpha;
		}
	%endif

	alpha *= 1.0f / (2.0f * tau0 + 1.0f);

	%for idx in grid.idx_name:
		d0->${idx} += alpha * fneq0.${idx};
	%endfor

	${fluid_velocity(0, save=True)}
}
%endif  ## model == elbm

%if model == 'bgk':
//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
//
${device_func} inline void BGK_relaxate(${bgk_args_decl()},
%for i in range(0, len(grids)):
	Dist *d${i},
%endfor
	int node_type, int ncode
	${dynamic_val_args_decl()})
{
	${bgk_relaxation_preamble()}

	%for i in range(0, len(grids)):
		%for idx in grid.idx_name:
			d${i}->${idx} += (feq${i}.${idx} - d${i}->${idx}) / tau${i};
		%endfor

		%if nt.NTGuoDensity in node_types:
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
			if (isNTGuoDensity(node_type)) {
				%for idx in grid.idx_name:
					d${i}->${idx} -= feq${i}.${idx};
				%endfor
			}
		%endif

		## Is there a force acting on the current grid?
		%if sym_force.needs_accel(i, forces, force_couplings):
			// Body force implementation follows Guo's method, eqs. 19 and 20 from
			// 10.1103/PhysRevE.65.046308.
			${fluid_velocity(i)};

			%if simtype == 'shan-chen':
			{
				float pref = ${cex(sym_force.bgk_external_force_pref(grids[i], grid_num=i))};
				%for val, idx in sym_force.bgk_external_force(grid, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			}
			%elif simtype == 'free-energy':
				%for val, idx in sym_force.free_energy_external_force(sim, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			%else:
			{
				float pref = ${cex(sym_force.bgk_external_force_pref(grids[i], grid_num=i))};
				%for val, idx in sym_force.bgk_external_force(grid, grid_num=i):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			}
			%endif
		%endif
	%endfor

	// FIXME: This should be moved to postcollision boundary conditions.
	%if nt.NTGuoDensity in node_types:
		if (isNTGuoDensity(node_type)) {
			int node_param_idx = decodeNodeParamIdx(ncode);
			float par_rho = node_params[node_param_idx];
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
	type, ncode ${dynamic_val_call_args()});
	%elif model == 'mrt' and simtype == 'free-energy':
		FE_MRT_relaxate(${bgk_args()},
%for i in range(0, len(grids)):
	&d${i},
%endfor
	type ${dynamic_val_call_args()});
	%elif model == 'elbm':
		ELBM_relaxate(${bgk_args()}, &d0 ${dynamic_val_call_args()} ${cond(alpha_output, ', options, alpha + gi')});
	%else:
		MS_relaxate(&d0, type, v ${dynamic_val_call_args()});
	%endif
</%def>

<%def name="relaxate(bgk_args)">
	%if relaxation_enabled:
		if (isWetNode(type)) {
			${_relaxate(bgk_args)}
		}
	%endif
</%def>

<%!
  from sailfish import sym, sym_force
  import sailfish.node_type as nt
%>

<%page args="bgk_args_decl"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="utils.mako" import="*"/>
<%namespace file="relaxation_common.mako" import="*"/>

%if model == 'mrt' and simtype == 'lbm':
<%namespace file="relaxation_mrt.mako" import="*" name="mrt"/>
${mrt.body()}
%endif

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

			%for val, idx in zip(sym_force.free_energy_external_force(sim, grid_num=i), grid.idx_name):
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

	${fluid_velocity(0, save=True)};

	%for local_var in elbm_eq_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
	%endfor

	%for i, eq in enumerate(elbm_eq):
		%for feq, idx in zip(eq, grid.idx_name):
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
				const float pref = ${cex(sym_force.bgk_external_force_pref(grids[i], grid_num=i))};
				%for val, idx in zip(sym_force.bgk_external_force(grid, grid_num=i), grid.idx_name):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			}
			%elif simtype == 'free-energy':
				%for val, idx in zip(sym_force.free_energy_external_force(sim, grid_num=i), grid.idx_name):
					d${i}->${idx} += ${cex(val, vectors=True)};
				%endfor
			%else:
			{
				const float pref = ${cex(sym_force.bgk_external_force_pref(grids[i], grid_num=i))};
				%for val, idx in zip(sym_force.bgk_external_force(grid, grid_num=i), grid.idx_name):
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
				const float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True, rho='par_rho', phi='par_phi')};
			%endfor

			tau0 = tau_a;

			%for i, eq in enumerate(bgk_equilibrium):
				%for feq, idx in zip(eq, grid.idx_name):
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

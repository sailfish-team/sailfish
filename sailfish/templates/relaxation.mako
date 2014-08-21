<%!
  from sailfish import sym, sym_force, sym_equilibrium, sym_codegen
  import sailfish.node_type as nt
%>

<%page args="bgk_args_decl"/>
<%namespace file="mako_utils.mako" import="*"/>
<%namespace file="relaxation_common.mako" import="*"/>

%if model == 'mrt' and simtype == 'lbm':
<%namespace file="relaxation_mrt.mako" import="*" name="mrt"/>
${mrt.body()}
%endif

%if model == 'mrt' and simtype == 'free-energy':
%for grid_idx in range(0, len(grids)):
${device_func} inline void FE_MRT_relaxate${grid_idx}(${bgk_args_decl(grid_idx)}, Dist *d0,
	int node_type
	${dynamic_val_args_decl()})
{
	${body_force(force_for_eq.get(grid_idx, grid_idx))}
	${bgk_relaxation_preamble(grid_idx)}
	## The acceleration vector needs to declared if no force was set in the
	## previous call to body_force().
	${body_force(grid_idx, vector_decl=(force_for_eq.get(grid_idx, -1) is None))}

	%for idx in grid.idx_name:
		## Use the BGK approximation for the relaxation of the order parameter field.
		%if grid_idx == 1:
			d0->${idx} += (feq0.${idx} - d0->${idx}) / tau1;
		%else:
			feq0.${idx} = d0->${idx} - feq0.${idx};
		%endif
	%endfor

	%if grid_idx == 0:
		%for dst, src in sym.free_energy_mrt(sim.grid, 'd0', 'feq0'):
			${dst} -= ${sym_codegen.make_float(src)};
		%endfor
	%endif

	## Is there a force acting on the current grid?
	%if sym_force.needs_accel(grid_idx, forces, force_couplings):
		${fluid_velocity(grid_idx)};

		%for val, idx in zip(sym_force.free_energy_external_force(sim, grid_num=grid_idx), grid.idx_name):
			d0->${idx} += ${cex(val)};
		%endfor
	%endif

	${fluid_output_velocity(grid_idx)}
}
%endfor
%endif  ## model == mrt && simtype == 'free-energy'

%if model == 'elbm':
<%include file="entropic.mako"/>

${device_func} inline void ELBM_relaxate(${bgk_args_decl()}, Dist* d0
	${dynamic_val_args_decl()}
%if alpha_output:
	, ${global_ptr} float* alpha_out
%endif
)
{
	Dist fneq;
	<% eq = sym_equilibrium.get_equilibrium(config, equilibria, grids, 0) %>

	float v0[${dim}];
	${body_force(grid_idx=0)}
	${fluid_velocity(0, equilibrium=True)};

	## Local variables used by the equilibrium.
	%for local_var in eq.local_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs)};
	%endfor

	%for feq, idx in zip(eq.expression, grid.idx_name):
		fneq.${idx} = ${cex(feq)} - d0->${idx};
	%endfor

	float alpha = EntropicRelaxationParam(d0, &fneq ${cond(alpha_output, ', alpha_out')});
	// alpha * beta
	alpha *= 1.0f / (2.0f * tau0 + 1.0f);
	%for idx in grid.idx_name:
		d0->${idx} += alpha * fneq.${idx};
	%endfor

	## Is there a force acting on the current grid?
	%if sym_force.needs_accel(0, forces, force_couplings):
		${fluid_velocity(0)};
		${apply_body_force(0, no_feq=True, subs=dict(tau0='1.0/alpha'))};
	%endif

	${fluid_output_velocity(0)}
}
%endif  ## model == elbm

%if model == 'bgk':
//
// Performs the relaxation step in the BGK model given the density rho,
// the velocity v and the distribution fi.
%for grid_idx in range(len(grids)):
${device_func} inline void BGK_relaxate${grid_idx}(${bgk_args_decl(grid_idx)},
	Dist *d0, int node_type, int ncode
	${dynamic_val_args_decl()})
{
	${body_force(grid_idx)}
	${bgk_relaxation_preamble(grid_idx)}

	%if grid_idx == 1:
		float omega = ${cex(1.0 / tau_phi)};
	%elif grid_idx == 2:
		float omega = ${cex(1.0 / tau_theta)};
	%elif subgrid == 'les-smagorinsky' or (simtype == 'free-energy' and grid_idx == 0):
		float omega = 1.0 / tau0;
	%else:
		float omega = ${cex(1.0 / tau)};
	%endif

	%for idx in grid.idx_name:
		d0->${idx} += omega * (feq0.${idx} - d0->${idx});
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
				d0->${idx} -= feq0.${idx};
			%endfor
		}
	%endif

	## Is there a force acting on the current grid?
	%if sym_force.needs_accel(grid_idx, forces, force_couplings):
		${fluid_velocity(grid_idx)};
		${apply_body_force(grid_idx)};
	%endif

	// FIXME: This should be moved to postcollision boundary conditions.
	%if nt.NTGuoDensity in node_types:
		if (isNTGuoDensity(node_type)) {
			int node_param_idx = decodeNodeParamIdx(ncode);
			float par_rho = node_params[node_param_idx];
			float par_phi = 1.0f;
			tau0 = tau_a;

			<% eq = equilibria[grid_idx](grids[grid_idx], config) %>
			%for local_var in eq.local_vars:
				const float ${cex(local_var.lhs)} =
					${cex(local_var.rhs, rho='par_rho', phi='par_phi')};
			%endfor
			%for feq, idx in zip(eq.expression, grid.idx_name):
				d0->${idx} += ${cex(feq, rho='par_rho', phi='par_phi')};
			%endfor
		}
	%endif

	${fluid_output_velocity(grid_idx)}
}
%endfor
%endif

<%def name="_relaxate(bgk_args, grid_idx)">
	%if model == 'bgk':
		BGK_relaxate${grid_idx}(${bgk_args(grid_idx)}, &d0, type, ncode ${dynamic_val_call_args()});
	%elif model == 'mrt' and simtype == 'free-energy':
		FE_MRT_relaxate${grid_idx}(${bgk_args(grid_idx)}, &d0, type ${dynamic_val_call_args()});
	%elif model == 'elbm':
		ELBM_relaxate(${bgk_args()}, &d0 ${dynamic_val_call_args()} ${cond(alpha_output, ', alpha + gi')});
	%else:
		MS_relaxate(&d0, type, v ${dynamic_val_call_args()});
	%endif
</%def>

<%def name="relaxate(bgk_args, grid_idx=0)">
	%if relaxation_enabled:
		if (isWetNode(type)) {
			${_relaxate(bgk_args, grid_idx)}
		}
	%endif
</%def>

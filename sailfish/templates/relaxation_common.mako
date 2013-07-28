<%!
  from sailfish import sym, sym_codegen, sym_force
  import sympy
%>

<%namespace file="code_common.mako" import="*"/>

## Defines the actual acceleration vectors.
<%def name="body_force(grid_idx=0, vector_decl=True)">
	%if forces is not UNDEFINED and (forces.numeric or forces.symbolic):
		%if sym_force.needs_accel(grid_idx, forces, {}):
			%if forces.symbolic and time_dependence:
				float phys_time = get_time_from_iteration(iteration_number);
			%endif
			// Body force acceleration.
			%if not sym_force.needs_coupling_accel(grid_idx, force_couplings):
				%if vector_decl:
					float ea${grid_idx}[${dim}];
				%endif
				%for j in range(0, dim):
					ea${grid_idx}[${j}] = ${cex(sym_force.body_force_accel(grid_idx, j, forces, accel=True))};
				%endfor
			%else:
				## If the current grid has a Shan-Chen force acting on it, the acceleration vector
				## is already externally defined in the Shan-Chen code.
				%for j in range(0, dim):
					%if grid_idx in forces.symbolic or grid_idx in forces.numeric:
						ea${grid_idx}[${j}] += ${cex(sym_force.body_force_accel(grid_idx, j, forces, accel=True))};
					%endif
				%endfor
			%endif
		%endif
	%endif
</%def>

<%def name="guo_force(i, subs)">
{
	// Guo's method, eqs. 19 and 20 from 10.1103/PhysRevE.65.046308.
	const float pref = ${cex(sym_force.guo_external_force_pref(grids[i], config, grid_num=i).subs(subs))};
	%for val, idx in zip(sym_force.guo_external_force(grid, grid_num=i), grid.idx_name):
		d0->${idx} += ${cex(val.subs(subs))};
	%endfor
}
</%def>

<%def name="edm_force(i)">
{
	// Exact difference method.
	%for feq_shifted, idx in zip(sym_force.edm_shift_velocity(equilibria[i](grids[i], config).expression, grid, i), grid.idx_name):
		d0->${idx} += ${cex(feq_shifted)} - feq0.${idx};
	%endfor
}
</%def>

<%def name="edm_force_no_feq(i)">
{
	// Exact difference method.
	%for feq_shifted, feq, idx in zip(sym_force.edm_shift_velocity(equilibria[i](grids[i], config).expression, grid, i), equilibria[i](grids[i], config).expression, grid.idx_name):
		d0->${idx} += ${cex(sym_codegen.poly_factorize(sympy.simplify(feq_shifted - feq)))};
	%endfor
}
</%def>

## Applies a body force in the relaxation step. This function effectively
## modifies the collision operator (BGK, ELBM, etc). At the time this function
## is called d${i} is already a post-relaxation distribution.
<%def name="apply_body_force(i, no_feq=False, subs={})">
	%if simtype == 'free-energy':
		%for val, idx in zip(sym_force.free_energy_external_force(sim, grid_num=i), grid.idx_name):
			d0->${idx} += ${cex(val)};
		%endfor
	%else:
		%if force_implementation == 'guo':
			${guo_force(i, subs)}
		%elif force_implementation == 'edm':
			%if no_feq:
				${edm_force_no_feq(i)}
			%else:
				${edm_force(i)}
			%endif
		%elif force_implementation != 'velocity_shift':
			__NOT_IMPLEMENTED_${force_implementation}
		%endif
	%endif
</%def>

## Computes the fluid velocity for use in the relaxation process.
## igrid: grid ID
## equilibrium: if True, calculate velocity for the equilibrium distribution
<%def name="fluid_velocity(igrid=0, equilibrium=False, ignore_forces=False)">
	%if forces is not UNDEFINED and not ignore_forces:
		%if force_implementation == 'guo':
			## The half time-step velocity shift is only used in Guo's method.
			%for j in range(0, dim):
				%if igrid in force_for_eq and equilibrium:
					v0[${j}] = iv0[${j}] + ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], j, forces, force_couplings))};
				%else:
					v0[${j}] = iv0[${j}] + ${cex(0.5 * sym_force.fluid_accel(sim, igrid, j, forces, force_couplings))};
				%endif
			%endfor
		%elif force_implementation == 'velocity_shift':
			%for j in range(0, dim):
				v0[${j}] = iv0[${j}] + tau${igrid} * ${cex(sym_force.fluid_accel(sim, igrid, j, forces, force_couplings))};
			%endfor
		%else:
			%for j in range(0, dim):
				v0[${j}] = iv0[${j}];
			%endfor
		%endif
	%else:
		%for j in range(0, dim):
			v0[${j}] = iv0[${j}];
		%endfor
	%endif
</%def>

## Updates the input velocity (iv0) if necessary (i.e. there are body forces
## acting on the fluid). The updated value is used as the simulation output.
## This should be called at the end of the relaxation process.
<%def name="fluid_output_velocity(igrid=0)">
	%if forces is not UNDEFINED:
		## For now, we assume a single fluid velocity regardless of the number of
		## components and grids used. If this assumption is ever changed, the code
		## below will have to be extended appropriately.
		%for j in range(0, dim):
			iv0[${j}] += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, j, forces, force_couplings))};
		%endfor
	%endif
</%def>

## Note: update BGK_relaxate* if a new condition is added below for which
## a modified relaxation time is used.
<%def name="update_relaxation_time(grid_idx)">
	## In models where the relaxation time is constant everywhere, tau0 is a global
	## constant and does not need to be declared here.
	%if simtype == 'free-energy' and grid_idx == 0:
		// Linear interpolation of relaxation time.
		float tau0 = tau_b + (phi + 1.0f) * (tau_a - tau_b) * 0.5f;
		if (phi < -1.0f) {
			tau0 = tau_b;
		} else if (phi > 1.0f) {
			tau0 = tau_a;
		}
	%endif

	%if subgrid == 'les-smagorinsky':
		// Relaxation time using the standard viscosity-relaxation time relation.
		float tau0 = 0.5f + 3.0f * visc;

		// Formulation of: Huidan Yu, Sharath S. Girimaji, Li-Shi Luo
		// Journal of Computational Physics 209 (2005) 599–616.

		// Q = T_ab T_ab
		// nonequilibrium stress tensor: T = e_ia e_ib (f_i - f_i^eq)

		// The 2nd order tensor formed from the equilibrium distributions is:
		//   rho / 3 * \delta_{ab} + rho u_a u_b
		// and this has to be subtracted from ex_flux (e_ia e_ib f_i).

		// TODO(michalj): Fix this for multifluid models.
		{
			float tmp, strain;

			strain = 0.0f;

			// Off-diagonal components, count twice for symmetry reasons.
			%for a in range(0, dim):
				%for b in range(a + 1, dim):
					 tmp = ${cex(sym.ex_flux(grid, 'd0', a, b, config), pointers=True)} -
						   ${cex(sym.ex_eq_flux(grid, a, b))};
					 strain += 2.0f * tmp * tmp;
				%endfor
			%endfor

			// Diagonal components.
			%for a in range(0, dim):
				tmp = ${cex(sym.ex_flux(grid, 'd0', a, a, config), pointers=True)} -
					  ${cex(sym.ex_eq_flux(grid, a, a))};
				strain += tmp * tmp;
			%endfor

			tau0 += 0.5f * (sqrtf(tau0 * tau0 + 36.0f * ${cex(smagorinsky_const**2)} * sqrtf(strain)) - tau0);
		}
	%endif
</%def>

## Code common to all BGK-like relaxation models.
<%def name="bgk_relaxation_preamble(grid_idx=0)">
	float v0[${dim}];

	<% eq = equilibria[grid_idx](grids[grid_idx], config) %>
	Dist feq0;
	${fluid_velocity(grid_idx, equilibrium=True)};

	%for local_var in eq.local_vars:
		float ${cex(local_var.lhs)} = ${cex(local_var.rhs)};
	%endfor

	%for feq, idx in zip(eq.expression, grid.idx_name):
		feq0.${idx} = ${cex(feq)};
	%endfor

	${update_relaxation_time(grid_idx)}

	## TODO: If the regularization is never used with collision operators
	## other than LBGK, move this directly to BGK relaxate and replace
	## the normal collision step with: fi = feq + (1 - omega) freg.
	%if regularized:
		float flux[${flux_components}];
		compute_noneq_2nd_moment(d0, rho, v0, flux);
		<% reg_diff = sym.reglb_flux_tensor(grid) %>

		%for feq, idx, reg in zip(eq.expression, grid.idx_name, reg_diff):
			d0->${idx} = feq0.${idx} + ${cex(reg, pointers=True)};
		%endfor
	%endif
</%def>

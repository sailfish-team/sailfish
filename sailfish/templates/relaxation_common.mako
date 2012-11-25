<%!
  from sailfish import sym_force
%>

<%namespace file="code_common.mako" import="*"/>

## Defines the actual acceleration vectors.
<%def name="body_force()">
	%if forces is not UNDEFINED and (forces.numeric or forces.symbolic):
		// Body force acceleration.
		%if forces.symbolic and time_dependence:
			float phys_time = get_time_from_iteration(iteration_number);
		%endif
		%for i in range(0, len(grids)):
			%if sym_force.needs_accel(i, forces, {}):
				%if not sym_force.needs_coupling_accel(i, force_couplings):
					float ea${i}[${dim}];
					%for j in range(0, dim):
						ea${i}[${j}] = ${cex(sym_force.body_force_accel(i, j, forces, accel=True), vectors=True)};
					%endfor
				%else:
					## If the current grid has a Shan-Chen force acting on it, the acceleration vector
					## is already externally defined in the Shan-Chen code.
					%for j in range(0, dim):
						%if i in forces.symbolic or i in forces.numeric:
							ea${i}[${j}] += ${cex(sym_force.body_force_accel(i, j, forces, accel=True), vectors=True)};
						%endif
					%endfor
				%endif
			%endif
		%endfor
	%endif
</%def>

<%def name="guo_force(i)">
{
	// Guo's method, eqs. 19 and 20 from 10.1103/PhysRevE.65.046308.
	const float pref = ${cex(sym_force.guo_external_force_pref(grids[i], grid_num=i))};
	%for val, idx in zip(sym_force.guo_external_force(grid, grid_num=i), grid.idx_name):
		d${i}->${idx} += ${cex(val, vectors=True)};
	%endfor
}
</%def>

<%def name="edm_force(i)">
{
	// Exact difference method.
	%for feq_shifted, idx in zip(sym_force.edm_shift_velocity(equilibria[i](grids[i]).expression), grid.idx_name):
		d${i}->${idx} += ${cex(feq_shifted, vectors=True)} - feq${i}.${idx};
	%endfor
}
</%def>

## Applies a body force in the relaxation step. This functioan effectively
## modifies the collision operator (BGK, ELBM, etc). At the time this function
## is called d${i} is already a post-relaxation distribution.
<%def name="apply_body_force(i)">
	%if simtype == 'free-energy':
		%for val, idx in zip(sym_force.free_energy_external_force(sim, grid_num=i), grid.idx_name):
			d${i}->${idx} += ${cex(val, vectors=True)};
		%endfor
	%else:
		%if force_implementation == 'guo':
			${guo_force(i)}
		%elif force_implementation == 'edm':
			${edm_force(i)}
		%else:
			__NOT_IMPLEMENTED_${force_implementation}
		%endif
	%endif
</%def>

## Computes the fluid velocity for use in the relaxation process.
## igrid: grid ID
## equilibrium: if True, calculate velocity for the equilibrium distribution
<%def name="fluid_velocity(igrid=0, equilibrium=False)">
	%if forces is not UNDEFINED and force_implementation == 'guo':
		## The half time-step velocity shift is only used in Guo's method.
		%for j in range(0, dim):
			%if igrid in force_for_eq and equilibrium:
				v0[${j}] = iv0[${j}] + ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], j, forces, force_couplings), vectors=True)};
			%else:
				v0[${j}] = iv0[${j}] + ${cex(0.5 * sym_force.fluid_accel(sim, igrid, j, forces, force_couplings), vectors=True)};
			%endif
		%endfor
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
			iv0[${j}] += ${cex(sym_force.fluid_accel(sim, igrid, j, forces, force_couplings), vectors=True)};
		%endfor
	%endif
</%def>

<%def name="update_relaxation_time()">
	## In models where the relaxation time is constant everywhere, tau0 is a global
	## constant and does not need to be declared here.
	%if simtype == 'free-energy':
		// Linear interpolation of relaxation time.
		float tau0 = tau_b + (phi + 1.0f) * (tau_a - tau_b) / 2.0f;
		if (phi < -1.0f) {
			tau0 = tau_b;
		} else if (phi > 1.0f) {
			tau0 = tau_a;
		}
	%endif

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
						 tmp = ${cex(sym.ex_flux(grid, 'd%d' % i, a, b), pointers=True)} -
							   ${cex(sym.S.rho * grid.v[a] * grid.v[b], vectors=True)};
						 strain += 2.0f * tmp * tmp;
					%endfor
				%endfor

				// Diagonal components.
				%for a in range(0, dim):
					tmp = ${cex(sym.ex_flux(grid, 'd%d' % i, a, a), pointers=True)} -
						  ${cex(sym.S.rho * (grid.v[a] * grid.v[b] + grid.cssq), vectors=True)};
					strain += tmp * tmp;
				%endfor

				// Form of the relaxation time correction as in comp-gas/9401004v1.
				tau${i} += (sqrtf(visc*visc + 18.0f * ${cex(smagorinsky_const**2)} *
							sqrtf(strain)) - visc) / 2.0f;
			%endfor
		}
	%endif
</%def>

## Code common to all BGK-like relaxation models.
<%def name="bgk_relaxation_preamble()">
	float v0[${dim}];
	${body_force()}

	%for i, eq in enumerate([f(g) for f, g in zip(equilibria, grids)]):
		Dist feq${i};
		${fluid_velocity(i, equilibrium=True)};

		%for local_var in eq.local_vars:
			float ${cex(local_var.lhs)} = ${cex(local_var.rhs, vectors=True)};
		%endfor

		%for feq, idx in zip(eq.expression, grid.idx_name):
			feq${i}.${idx} = ${cex(feq, vectors=True)};
		%endfor
	%endfor

	${update_relaxation_time()}
</%def>

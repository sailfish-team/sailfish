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

## Code common to all BGK-like relaxation models.
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
		${fluid_velocity(i, equilibrium=True)};

		%for feq, idx in zip(eq, grid.idx_name):
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



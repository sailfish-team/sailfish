<%
  from sailfish import sym, sym_force
  import sailfish.node_type as nt
%>

<%namespace file="utils.mako" import="*"/>
<%namespace file="code_common.mako" import="*"/>
<%namespace file="relaxation_common.mako" import="*"/>

## TODO: support multiple grids with MRT?
<%def name="fluid_momentum(igrid=0)">
	%if forces is not UNDEFINED:
		%if igrid in force_for_eq and equilibrium:
			fm.mx += ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], 0, forces, force_couplings))};
			fm.my += ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], 1, forces, force_couplings))};
			%if dim == 3:
				fm.mz += ${cex(0.5 * sym_force.fluid_accel(sim, force_for_eq[igrid], 2, forces, force_couplings))};
			%endif
		%else:
			fm.mx += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, 0, forces, force_couplings))};
			fm.my += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, 1, forces, force_couplings))};
			%if dim == 3:
				fm.mz += ${cex(0.5 * sym_force.fluid_accel(sim, igrid, 2, forces, force_couplings))};
			%endif
		%endif
	%endif
</%def>

//
// Relaxation in moment space.
//
${device_func} void MS_relaxate(Dist *fi, int node_type, float *iv0 ${dynamic_val_args_decl()})
{
	DistM fm, feq;

	%for mrt, val in sym.bgk_to_mrt(grid, 'fi', 'fm'):
		${mrt} = ${val};
	%endfor

	${body_force()}
	${fluid_momentum()}

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

	${fluid_momentum()}

	%for bgk, val in sym.mrt_to_bgk(grid, 'fi', 'fm'):
		${bgk} = ${val};
	%endfor

	${fluid_output_velocity()}
}

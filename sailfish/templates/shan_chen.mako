<%
	from sailfish import sym
%>

<%namespace file="utils.mako" import="get_field_loc,get_field_off"/>
<%namespace file="code_common.mako" import="cex"/>
<%namespace file="kernel_common.mako" import="*"/>

## Declares and evaluate the Shan-Chen forces.
<%def name="sc_calculate_force(grid_idx=0)">
	float sca0[${dim}]={};

	if (isWetNode(type)) {
		%for dists, coupling_const in force_couplings.iteritems():
			%if (dists[0] == grid_idx) and (constants[coupling_const] != 0.0):
				shan_chen_force(gi, g${grid_idx}m0, gg${dists[1]}m0,
								${coupling_const}, sca0, ${position()});
			%endif
		%endfor

		// Convert momentum and force into velocity and acceleration.
		%for i in range(0, dim):
			sca0[${i}] /= g${grid_idx}m0;
		%endfor
	}
</%def>

// Calculates the Shan-Chan pseudopotential.
${device_func} inline float sc_ppot(${global_ptr} ${const_ptr} float *__restrict__ field, int gi)
{
	float lfield = field[gi];
	return ${cex(sym.SHAN_CHEN_POTENTIALS[sc_potential]('lfield'))};
}

// Calculates the Shan-Chen force between two fluid components.
//
//  F = -G * \phi_A(x) \sum_i w_i e_i \phi_B(x + e_i)
//
// i: global node index
// rho: density at the current node
// field: (density) fielf of the other fluid component
// f1, f2: fields
// cc: coupling constant
// out: Shan-Chen force (output variable)
// x, y, [z]: position of the node
${device_func} inline void shan_chen_force(int i, float rho, ${global_ptr} ${const_ptr} float *__restrict__ field,
float cc, float *out, ${position_decl(prefix='')})
{
	float psi;
	float force[${dim}] = {};

	%if block.envelope_size != 0:
		int off;
	%endif

	int gi;		// global index

	%for i, ve in enumerate(grid.basis):
		%if ve.dot(ve) != 0.0:
			// ${ve}
			%if block.envelope_size == 0:
				${get_field_loc(*ve)};
			%else:
				${get_field_off(*ve)}
				gi = i + off;
			%endif

			psi = sc_ppot(field, gi);

			%for j, component in enumerate(ve):
				%if component != 0:
					force[${j}] += psi * ${component * grid.weights[i]};
				%endif
			%endfor
		%endif
	%endfor

	// Local node -- no offset.
	psi = ${cex(sym.SHAN_CHEN_POTENTIALS[sc_potential]('rho'))};

	%for i in range(0, dim):
		force[${i}] *= - psi * cc;
		out[${i}] += force[${i}];
	%endfor
}

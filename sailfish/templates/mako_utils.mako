<%!
    from sailfish import sym_codegen
%>

<%def name="cex(ex, pointers=False, rho=None, vectors=True, phi=None, vel=None)">
${sym_codegen.cexpr(sim, incompressible, pointers, ex, rho=rho, vectors=vectors, phi=phi, vel=vel)}
</%def>

<%def name="dump_dists(name, short=False, precision=5)">
	<%
		format_str = ""
		values = []
		for dname in grid.idx_name:
			if short:
				format_str += '%.' + str(precision) + 'e '
			else:
				format_str += dname + ':%.' + str(precision) + 'e '
			values.append('%s.%s' % (name, dname))

		values = ', '.join(values)
	%>
	printf("${format_str}\n", ${values});
</%def>

<%def name="cond(cond, text)" filter="trim">
	%if cond:
		${text}
	%endif
</%def>

<%def name="eval_dist(ex, dest)">
	%for val, idx in zip(ex, grid.idx_name):
		${dest}.${idx} = ${cex(val)}
	%endfor
</%def>

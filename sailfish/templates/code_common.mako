<%!
    from sailfish import sym
%>

<%def name="cex(ex, pointers=False, rho=None, vectors=True, phi=None)">
${sym.cexpr(sim, incompressible, pointers, ex, rho=rho, vectors=vectors, phi=phi)}
</%def>

<%def name="dump_dists(name, short=False)">
	<%
		format_str = ""
		values = []
		for dname in grid.idx_name:
			if short:
				format_str += '%.5e '
			else:
				format_str += dname + ':%.5e '
			values.append('%s.%s' % (name, dname))

		values = ', '.join(values)
	%>
	printf("dists: ${format_str}\n", ${values});
</%def>

<%def name="cond(cond, text)">
	%if cond:
		${text}
	%endif
</%def>


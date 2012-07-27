<%!
    from sailfish import sym
%>

<%def name="cex(ex, pointers=False, rho=None, vectors=False, phi=None)">
${sym.cexpr(sim, incompressible, pointers, ex, rho=rho, vectors=vectors, phi=phi)}
</%def>

<%def name="dump_dists(name)">
	<%
		format_str = ""
		values = []
		for dname in grid.idx_name:
			format_str += dname + ':%.5e '
			values.append('%s.%s' % (name, dname))

		values = ', '.join(values)
	%>
	printf("dists: ${format_str}\n", ${values});
</%def>

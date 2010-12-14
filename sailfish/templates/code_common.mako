<%!
    from sailfish import sym
%>

<%def name="cex(ex, pointers=False, rho=None, vectors=False, phi=None)">
${sym.cexpr(sim, incompressible, pointers, ex, rho=rho, vectors=vectors, phi=phi)}
</%def>


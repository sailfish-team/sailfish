<%!
    from sailfish import sym
%>

<%def name="cex(ex, pointers=False, rho=None)">
${sym.cexpr(grid, incompressible, pointers, ex, rho)}
</%def>


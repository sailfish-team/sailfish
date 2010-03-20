<%namespace file="propagation.mako" import="rel_offset"/>

<%def name="get_field_off(xoff, yoff, zoff)">
	off = ${rel_offset(xoff, yoff, zoff)};

	%if periodicity[0] and xoff != 0:
		nx = x + ${xoff};
		if (nx < 0 || nx > ${lat_nx-1})
			off += ${pbc_offsets[0][int(xoff)]};
	%endif

	%if periodicity[1] and yoff != 0:
		ny = y + ${yoff};
		if (ny < 0 || ny > ${lat_ny-1})
			off += ${pbc_offsets[1][int(yoff)]};
	%endif

	%if periodicity[2] and zoff != 0:
		nz = z + ${zoff};
		if (nz < 0 || nz > ${lat_nz-1})
			off += ${pbc_offsets[2][int(zoff)]};
	%endif
</%def>


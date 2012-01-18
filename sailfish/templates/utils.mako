<%namespace file="propagation.mako" import="rel_offset"/>

<%def name="get_field_off(xoff, yoff, zoff=0)">
	off = ${rel_offset(xoff, yoff, zoff)};
	%if periodicity[0] and xoff != 0:
	{
		int nx = x + ${xoff};
		%if xoff > 0:
			if (nx > ${lat_nx-1}) {
				nx = 0;
		%else:
			if (nx < 0) {
				nx = ${lat_nx-1};
		%endif
				off += ${pbc_offsets[0][int(xoff)]};
			}
	}
	%endif

	%if periodicity[1] and yoff != 0:
	{
		int ny = y + ${yoff};
		%if yoff > 0:
			if (ny > ${lat_ny-1}) {
				ny = 0;
		%else:
			if (ny < 0) {
				ny = ${lat_ny-1};
		%endif
				off += ${pbc_offsets[1][int(yoff)]};
			}
	}
	%endif

	%if periodicity[2] and zoff != 0:
	{
		int nz = z + ${zoff};
		%if zoff > 0:
			if (nz > ${lat_nz-1}) {
				nz = 0;
		%else:
			if (nz < 0) {
				nz = ${lat_nz-1};
		%endif
				off += ${pbc_offsets[2][int(zoff)]};
			}
	}
	%endif
</%def>

<%def name="fld_args()">
	nx, ny
	%if dim == 3:
		, nz
	%endif
</%def>

<%def name="fld_arg_decl()">
	int nx, int ny
	%if dim == 3:
		, int nz
	%endif
</%def>

<%def name="nonlocal_fld(fld_id)">
	f${fld_id}[idx]
</%def>


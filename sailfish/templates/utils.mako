<%namespace file="propagation.mako" import="rel_offset"/>

<%def name="get_field_off(xoff, yoff, zoff)">
	off = ${rel_offset(xoff, yoff, zoff)};
	nx = x + ${xoff};
	ny = y + ${yoff};
	%if dim == 3:
		nz = z + ${zoff};
	%endif

	%if periodicity[0] and xoff != 0:
		%if xoff > 0:
			if (nx > ${lat_nx-1}) {
				nx = 0;
		%else:
			if (nx < 0) {
				nx = ${lat_nx-1};
		%endif
				off += ${pbc_offsets[0][int(xoff)]};
			}
	%endif

	%if periodicity[1] and yoff != 0:
		%if yoff > 0:
			if (ny > ${lat_ny-1}) {
				ny = 0;
		%else:
			if (ny < 0) {
				ny = ${lat_ny-1};
		%endif
				off += ${pbc_offsets[1][int(yoff)]};
			}
	%endif

	%if periodicity[2] and zoff != 0:
		%if zoff > 0:
			if (nz > ${lat_nz-1}) {
				nz = 0;
		%else:
			if (nz < 0) {
				nz = ${lat_nz-1}
		%endif
				off += ${pbc_offsets[2][int(zoff)]};
			}
	%endif
</%def>

<%def name="fld_args()">
	nx, ny
	%if dim == 3:
		, nz
	%endif
</%def>

<%def name="nonlocal_fld(fld_id)">
	%if fld_id in image_fields:
		tex2D(img_f${fld_id}, nx, ny)
	%else:
		f${fld_id}[idx]
	%endif
</%def>


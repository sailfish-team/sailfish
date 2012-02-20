<%namespace file="propagation.mako" import="rel_offset"/>

## Calculates an offset for the global index given a
## vector from the current node.  Takes periodic boundary
## conditions into account.
##
## Note that with PBC implemented using ghost nodes, the
## periodicity code will never be enabled and the whole
## function reduces to retrieving the offset.
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

## Calculates global location of a node given a vector from the
## current node.  Takes periodic boundary conditions and mirror
## boundary conditions into account.
<%def name="get_field_loc(xoff, yoff, zoff=0)">
{
	int nx = x + ${xoff};
	int ny = y + ${yoff};
	%if dim == 3:
		int nz = z + ${zoff};
	%endif

	%if xoff != 0:
		%if periodicity[0]:
			// Periodic boundary conditions.
			%if xoff > 0:
				if (nx > ${lat_nx-1}) {	nx = 0;				}
			%else:
				if (nx < 0) {			nx = ${lat_nx-1};	}
			%endif
		%else:
			// Mirror boundary conditions.
			%if xoff > 0:
				if (nx > ${lat_nx-1}) {	nx = ${lat_nx-1};	}
			%else:
				if (nx < 0) {			nx = 0;				}
			%endif
		%endif
	%endif

	%if yoff != 0:
		%if periodicity[1]:
			%if yoff > 0:
				if (ny > ${lat_ny-1}) {	ny = 0;				}
			%else:
				if (ny < 0) {			ny = ${lat_ny-1};	}
			%endif
		%else:
			%if yoff > 0:
				if (ny > ${lat_ny-1}) {	ny = ${lat_ny-1};	}
			%else:
				if (ny < 0) {			ny = 0;				}
			%endif
		%endif
	%endif

	%if zoff != 0:
		%if periodicity[2]:
			%if zoff > 0:
				if (nz > ${lat_nz-1}) {	nz = 0;				}
			%else:
				if (nz < 0) {			nz = ${lat_nz-1};	}
			%endif
		%else:
			%if zoff > 0:
				if (nz > ${lat_nz-1}) {	nz = ${lat_nz-1};	}
			%else:
				if (nz < 0) {			nz = 0;				}
			%endif
		%endif
	%endif

	%if dim == 2:
		gi = getGlobalIdx(nx, ny);
	%else:
		gi = getGlobalIdx(nx, ny, nz);
	%endif
}
</%def>

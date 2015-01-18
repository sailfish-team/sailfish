## Mako helper functions. This file should not contain any code that is
## rendered directly (i.e. not inside a Mako function).

<%namespace file="propagation.mako" import="rel_offset"/>

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

<%def name="cond(cond, text, text2=None)" filter="trim">
	%if cond:
		${text}
  %elif text2 is not None:
    ${text2}
	%endif
</%def>

<%def name="eval_dist(ex, dest)">
	%for val, idx in zip(ex, grid.idx_name):
		${dest}.${idx} = ${cex(val)}
	%endfor
</%def>

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

<%def name="zero_gradient_at_boundaries()">
	// If PBC are not enabled, there is no meaningful way to calculate gradients
	// at boundaries -- assume 0.0.  If a different value is required, a row of
	// 'unused' nodes can be used to work around this.
	if (0
		%if not block.periodic_x and not block.has_face_conn(block.X_LOW):
			|| x == 1
		%elif not block.periodic_x and not block.has_face_conn(block.X_HIGH):
			|| x == ${lat_nx}
		%endif
		%if not block.periodic_y and not block.has_face_conn(block.Y_LOW):
			|| y == 1
		%elif not block.periodic_y and not block.has_face_conn(block.Y_HIGH):
			|| y == ${lat_ny}
		%endif
		%if dim == 3 and not block.periodic_z:
			%if not block.has_face_conn(block.Z_LOW):
				|| z == 1
			%elif not block.has_face_conn(block.Z_HIGH):
				|| z == ${lat_nz}
			%endif
		%endif
		) {
		%for i in range(dim):
			grad[${i}] = 0.0f;
		%endfor
		laplacian[0] = 0.0f;
		return;
	}
</%def>

## Provides declarations of the arguments required for functions using
## dynamically evaluated node parameters.
<%def name="dynamic_val_args_decl()">
	%if time_dependence:
		, unsigned int iteration_number
	%endif
	%if space_dependence:
		, int gx,
		int gy
		%if dim == 3:
			, int gz
		%endif
	%endif
</%def>


## Provides values of the arguments required for functions using dynamically
## evaluated node parameters.
<%def name="dynamic_val_args()">
	%if time_dependence:
		, iteration_number
	%endif
	%if space_dependence:
		, gx, gy
		%if dim == 3:
			, gz
		%endif
	%endif
</%def>

## Use to render arguments to the first call of a function using dynamically
## evaluated node paramters. Takes care of calculating the node's logical
## global position.
<%def name="dynamic_val_call_args()">
	%if time_dependence:
		, iteration_number
	%endif
	%if space_dependence:
		, gx + ${x_local_device_to_global_offset},
		gy + ${y_local_device_to_global_offset}
		%if dim == 3:
			, gz + ${z_local_device_to_global_offset}
		%endif
	%endif
</%def>

<%def name="ifdim3(val)" filter='trim'>
	%if dim == 3:
		val
	%endif
</%def>

## TODO(michalj): Consider using only a single float field for this with a
## known offset to minimize number of arguments.
<%def name="force_field_if_required()">
	%if force_field:
		, ${global_ptr} float *__restrict__ force_x
		, ${global_ptr} float *__restrict__ force_y
		%if dim == 3:
			, ${global_ptr} float *__restrict__ force_z
		%endif
	%endif
</%def>

<%def name="force_field_arg_if_required()">
	%if force_field:
		, force_x, force_y ${ifdim3(', force_z')}
	%endif
</%def>

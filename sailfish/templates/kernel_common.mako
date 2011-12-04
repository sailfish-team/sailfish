<%page args="bgk_args_decl"/>

<%def name="nonlocal_fields_decl()">
	%if backend == 'cuda':
		%for i in image_fields:
			%if precision == 'single':
				texture<float, 2> img_f${i};
			%else:
				texture<int2, 2> img_f${i};
			%endif
		%endfor
	%endif
</%def>

<%def name="kernel_args_1st_moment(name)">
	${global_ptr} float *${name}x,
	${global_ptr} float *${name}y,
%if dim == 3:
	${global_ptr} float *${name}z,
%endif
</%def>

## Convenience function to call getGlobalIdx without an explicit conditional
## clause in the template code.
<%def name="get_global_idx(x='gx', y='gy', z='gz')" filter="trim">
	%if dim == 2:
		getGlobalIdx(${x}, ${y})
	%else:
		getGlobalIdx(${x}, ${y}, ${z})
	%endif
</%def>

## Defines local indices for kernels that do not distinguish between
## bulk and boundary regions.
<%def name="local_indices()">
	int lx = get_local_id(0);	// ID inside the current block
	%if dim == 2:
		int gx = get_global_id(0);
		int gy = get_group_id(1);
	%else:
		// This is a workaround for the limitations of current CUDA devices.
		// We would like the grid to be 3 dimensional, but only 2 dimensions
		// are supported.  We thus encode the first two dimensions (x, y) of
		// the simulation grid into the x dimension of the CUDA/OpenCL grid
		// as:
		//   x_dev = y * num_blocks + x.
		//
		// This works fine, as x is relatively small, since:
		//   x = x_sim / block_size.
		int gx = get_global_id(0) % ${arr_nx};
		int gy = get_global_id(0) / ${arr_nx};
		int gz = get_global_id(1);
	%endif

	int gi = ${get_global_idx()};

	// Nothing to do if we're outside of the simulation domain.
	if (gx > ${lat_nx-1}) {
		return;
	}
</%def>

## Defines local indices for bulk kernels.
## This is the same as local_indices(), but with proper offsets to skip
## the boundary.
<%def name="local_indices_bulk()">
	lx = get_local_id(0);	// ID inside the current block
	%if dim == 2:
		<%
			if block.has_face_conn(block.X_LOW):
				xoff = block_size
			else:
				xoff = 0

			if block.has_face_conn(block.Y_LOW):
				yoff = boundary_size
			else:
				yoff = 0
		%>

		gx = ${xoff} + get_global_id(0);
		gy = ${yoff} + get_group_id(1);
	%else:
		gx = ${block_size} + get_global_id(0) % ${arr_nx - 2 * block_size};
		gy = ${boundary_size} + get_global_id(0) / ${arr_nx - 2 * block_size};
		gz = ${boundary_size} + get_global_id(1);
	%endif

	gi = ${get_global_idx()};

	// Nothing to do if we're outside of the simulation domain.
	if (gx > ${lat_nx-1}) {
		return;
	}
</%def>

## Defines local indices for boundary kernels.
<%def name="local_indices_boundary()">
	lx = get_local_id(0);	// ID inside the current block
	int gid = get_group_id(0) + get_group_id(1) * get_global_size(0) / get_local_size(0);
	%if dim == 2:
		<%
			has_ylow = int(block.has_face_conn(block.Y_LOW))
			has_yhigh = int(block.has_face_conn(block.Y_HIGH))
			has_xlow = int(block.has_face_conn(block.X_LOW))
			has_xhigh = int(block.has_face_conn(block.X_HIGH))

			ns_conns = has_ylow + has_yhigh
			xblocks = arr_nx / block_size
			yblocks = arr_ny - ns_conns * boundary_size

			padding = arr_nx - lat_nx
			if block.has_face_conn(block.X_HIGH) and block_size - padding >= boundary_size:
				aux_ew = 1	# 2 blocks on the right due to misalignment
			else:
				aux_ew = 0	# 1 block on the right

			bottom_idx = has_ylow * boundary_size * xblocks
			left_idx = bottom_idx + has_yhigh * boundary_size * xblocks
			right_idx = left_idx + has_xlow * yblocks
			right2_idx = right_idx + has_xhigh * yblocks
			max_idx = right2_idx + aux_ew * yblocks
		%>
		// x: ${xblocks}, y: ${yblocks}
		if (0) {;}
		%if block.has_face_conn(block.Y_LOW):
			else if (gid < ${bottom_idx}) {
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gy = gid / ${xblocks};
			}
		%endif
		%if block.has_face_conn(block.Y_HIGH):
			else if (gid < ${left_idx}) {
				gid -= ${bottom_idx};
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gy = ${lat_ny-1} - gid / ${xblocks};
			}
		%endif
		%if block.has_face_conn(block.X_LOW):
			else if (gid < ${right_idx}) {
				gx = lx;
				gy = gid + ${has_ylow * boundary_size - left_idx};
			}
		%endif
		%if block.has_face_conn(block.X_HIGH):
			else if (gid < ${right2_idx}) {
				gx = ${arr_nx - block_size} + lx;
				gy = gid + ${has_ylow * boundary_size - right_idx};
			} else if (gid < ${max_idx}) {
				gx = ${arr_nx - 2*block_size} + lx;
				gy = gid + ${has_ylow * boundary_size - right2_idx};
			}
		%endif
		else {
			return;
		}
	%else:
		<%
			xblocks = arr_nx / block_size
			yblocks = arr_ny - 2 * boundary_size
			zblocks = arr_nz - 2 * boundary_size
			ortho_blocks = yblocks * zblocks

			bottom_idx = boundary_size * xblocks * arr_nz * 2
			left_idx = bottom_idx + xblocks * yblocks * boundary_size * 2
			right_idx = left_idx + ortho_blocks
			right2_idx = right_idx + ortho_blocks
			max_idx = right2_idx + ortho_blocks
		%>
		// N/S faces, Y-connection.  Face area is arr_nx * arr_nz.
		if (gid < ${bottom_idx}) {
			gx = (gid % ${xblocks}) * ${block_size} + lx;
			gid = gid / ${xblocks};
			gz = gid % ${arr_nz};
			gid = gid / ${arr_nz};
			if (gid < ${boundary_size}) {
				gy = gid;
			} else {
				gy = ${arr_ny - boundary_size} + gid - ${boundary_size};
			}
		// B/T faces, Z-connection.  Face area is arr_nx * (arr_ny-2)
		} else if (gid < ${left_idx}) {
			gid -= ${bottom_idx};
			gx = (gid % ${xblocks}) * ${block_size} + lx;
			gid = gid / ${xblocks};
			// The area within the boundary is handled in the first case
			// above (N/S faces.
			gy = ${boundary_size} + gid / ${2 * boundary_size};
			gz = gid % ${2 * boundary_size};
			if (gz >= ${boundary_size}) {
				gz = ${arr_nz - boundary_size} + gz - ${boundary_size};
			}
		// E face
		} else if (gid < ${right_idx}) {
			gid -= ${left_idx};
			gx = lx;
			gy = ${boundary_size} + gid % ${yblocks};
			gz = ${boundary_size} + gid / ${yblocks};
		// W face (part 1)
		} else if (gid < ${right2_idx}) {
			gid -= ${right_idx};
			gx = ${arr_nx - block_size} + lx;
			gy = ${boundary_size} + gid % ${yblocks};
			gz = ${boundary_size} + gid / ${yblocks};
		// W face (part 2)
		} else if (gid < ${max_idx}) {
			gid -= ${right2_idx};
			gx = ${arr_nx - 2*block_size} + lx;
			gy = ${boundary_size} + gid % ${yblocks};
			gz = ${boundary_size} + gid / ${yblocks};
		} else {
			return;
		}
	%endif

	gi = ${get_global_idx()};

	// Nothing to do if we're outside of the simulation domain.
	if (gx > ${lat_nx-1}) {
		return;
	}
</%def>

<%def name="get_dist(array, i, idx, offset=0)" filter="trim">
	${array}[${idx} + DIST_SIZE * ${i} + ${offset}]
</%def>

## FIXME: This should work in 3D.  Right now, there is no use case for that
## so we leave it 2D only.
<%def name="wrap_coords()">
	if (gx < 0) {
		%if periodic_x:
			gx += ${lat_nx};
		%else:
			return;
		%endif
	}

	if (gx > ${lat_nx-1}) {
		%if periodic_x:
			gx -= ${lat_nx};
		%else:
			return;
		%endif
	}

	if (gy < 0) {
		%if periodic_y:
			gy += ${lat_ny};
		%else:
			return;
		%endif
	}

	if (gy > ${lat_ny-1}) {
		%if periodic_y:
			gy -= ${lat_ny};
		%else:
			return;
		%endif
	}
</%def>

#define BLOCK_SIZE ${block_size}
#define DIST_SIZE ${dist_size}
#define GEO_FLUID ${geo_fluid}
#define OPTION_SAVE_MACRO_FIELDS 1
#define OPTION_BULK 2

#define DT 1.0f

%for name, val in constants:
	${const_var} float ${name} = ${val}f;
%endfor

%if geo_params:
	// Additional geometry parameters (velocities, pressures, etc)
	${const_var} float geo_params[${len(geo_params)}] = {
	%for param in geo_params:
		${param}f,
	%endfor
	};
%else:
	${const_var} float geo_params[1] = {0};
%endif

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>

%if precision == 'double':
${device_func} inline double get_img_field(texture<int2, 2> t, int x, int y)
{
	int2 v = tex2D(t, x, y);
	return __hiloint2double(v.y, v.x);
}
%else:
#define get_img_field(t, x, y) tex2D(t, x, y)
%endif

${opencl_compat.body()}
<%include file="geo_helpers.mako"/>
${boundary.body()}
${relaxation.body(bgk_args_decl)}


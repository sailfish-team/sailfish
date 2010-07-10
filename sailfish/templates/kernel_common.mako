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

<%def name="local_indices()">
	int lx = get_local_id(0);	// ID inside the current block
	%if dim == 2:
		int gx = get_global_id(0);
		int gy = get_group_id(1);
		int gi = gx + ${arr_nx}*gy;
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

		## FIXME: there should be a single mako function for calculating the
		## global index.
		int gi = gx + gy*${arr_nx} + ${arr_nx*arr_ny}*gz;
	%endif

	// Nothing to do if we're outside of the simulation domain.
	if (gx > ${lat_nx-1}) {
		return;
	}
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

#define DT 1.0f

%for name, val in constants:
	${const_var} float ${name} = ${val}f;
%endfor

${const_var} float geo_params[${num_params+1}] = {
%for param in geo_params:
	${param}f,
%endfor
0};		// geometry parameters

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


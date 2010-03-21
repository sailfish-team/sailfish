<%page args="bgk_args_decl"/>

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
		int gi = gx + ${lat_nx}*gy;
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
		int gx = get_global_id(0) % ${lat_nx};
		int gy = get_global_id(0) / ${lat_nx};
		int gz = get_global_id(1);
		int gi = gx + gy*${lat_nx} + ${lat_nx*lat_ny}*gz;
	%endif
</%def>

#define BLOCK_SIZE ${block_size}
#define DIST_SIZE ${dist_size}
#define GEO_FLUID ${geo_fluid}
#define GEO_BCV ${geo_bcv}
#define GEO_BCP ${geo_bcp}

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

${opencl_compat.body()}
<%include file="geo_helpers.mako"/>
${boundary.body()}
${relaxation.body(bgk_args_decl)}


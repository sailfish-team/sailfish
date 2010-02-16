<%def name="kernel_args()">
	${global_ptr} int *map, ${global_ptr} float *dist_in,
	${global_ptr} float *dist_out, ${global_ptr} float *orho, ${global_ptr} float *ovx,
	${global_ptr} float *ovy,
%if dim == 3:
	${global_ptr} float *ovz,
%endif
	int save_macro
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


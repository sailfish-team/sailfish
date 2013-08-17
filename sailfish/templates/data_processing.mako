<%namespace file="propagation.mako" import="rel_offset"/>
<%namespace file="kernel_common.mako" import="kernel_args_1st_moment,local_indices" />

%if dim == 3:
${kernel} void ComputeSquareVelocityAndVorticity(
	${global_ptr} ${const_ptr} int *__restrict__ map,
	${kernel_args_1st_moment('v')}
	${global_ptr} float *usq,
	${global_ptr} float *vort_sq) {

	// We will decide ourselves whether a ghost can be skipped. In order for GPUArray
	// reductions to work, the ghost nodes have to be filled with 0s.
	${local_indices(no_outside=False)}
	if (gi >= ${dist_size}) {
		return;
	}

	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	if (isExcludedNode(type) || gx >= ${lat_nx-1} || gy >= ${lat_ny-1} || gz >= ${lat_nz-1}) {
		usq[gi] = 0.0f;
		vort_sq[gi] = 0.0f;
		return;
	}
	float lvx = vx[gi];
	float lvy = vy[gi];
	float lvz = vz[gi];
	usq[gi] = lvx * lvx + lvy * lvy + lvz * lvz;

	float duz_dy, dux_dy;
	// TODO(mjanusz): Modify this for variable-size neighborhood.
	if (gy > 1 && gy < ${lat_ny-2}) {
		duz_dy = (vz[gi + ${rel_offset(0, 1, 0)}] -
				  vz[gi + ${rel_offset(0, -1, 0)}]) * 0.5f;
		dux_dy = (vx[gi + ${rel_offset(0, 1, 0)}] -
				  vx[gi + ${rel_offset(0, -1, 0)}]) * 0.5f;
	} else if (gy == ${lat_ny-2}) {
		duz_dy = lvz - vz[gi + ${rel_offset(0, -1, 0)}];
		dux_dy = lvx - vx[gi + ${rel_offset(0, -1, 0)}];
	} else if (gy == 1) {
		duz_dy = vz[gi + ${rel_offset(0, 1, 0)}] - lvz;
		dux_dy = vx[gi + ${rel_offset(0, 1, 0)}] - lvx;
	}

	float duy_dz, dux_dz;
	if (gz > 1 && gz < ${lat_nz-2}) {
		duy_dz = (vy[gi + ${rel_offset(0, 0, 1)}] -
				  vy[gi + ${rel_offset(0, 0, -1)}]) * 0.5f;
		dux_dz = (vx[gi + ${rel_offset(0, 0, 1)}] -
				  vx[gi + ${rel_offset(0, 0, -1)}]) * 0.5f;
	} else if (gz == ${lat_nz-2}) {
		duy_dz = lvy - vy[gi + ${rel_offset(0, 0, -1)}];
		dux_dz = lvx - vx[gi + ${rel_offset(0, 0, -1)}];
	} else if (gz == 1) {
		duy_dz = vy[gi + ${rel_offset(0, 0, 1)}] - lvy;
		dux_dz = vx[gi + ${rel_offset(0, 0, 1)}] - lvx;
	}

	float duz_dx, duy_dx;
	if (gx >= 2 && gx < ${lat_nx-2}) {
		duz_dx = (vz[gi + ${rel_offset(1, 0, 0)}] -
				  vz[gi + ${rel_offset(-1, 0, 0)}]) * 0.5f;
		duy_dx = (vy[gi + ${rel_offset(1, 0, 0)}] -
				  vy[gi + ${rel_offset(-1, 0, 0)}]) * 0.5f;
	} else if (gx == ${lat_nx-2}) {
		duz_dx = lvz - vz[gi + ${rel_offset(-1, 0, 0)}];
		duy_dx = lvy - vy[gi + ${rel_offset(-1, 0, 0)}];
	} else if (gx == 1) {
		duz_dx = vz[gi + ${rel_offset(1, 0, 0)}] - lvz;
		duy_dx = vy[gi + ${rel_offset(1, 0, 0)}] - lvy;
	}

	float vort_x = duz_dy - duy_dz;
	float vort_y = dux_dz - duz_dx;
	float vort_z = duy_dx - dux_dy;

	vort_sq[gi] = vort_x * vort_x + vort_y * vort_y + vort_z * vort_z;
}
%endif

<%def name="_compute_stats(num_inputs, stats, out_type)">
	// Cache variables from global memory.
	%for i in range(num_inputs):
		${out_type} lf${i} = f${i}[gi];
	%endfor
	%for i, cfg in enumerate(stats):
	{
		${out_type} t = 1.0f;
		%for var, power in cfg:
			t *= ${' * '.join(['lf%s' % var] * power)};
		%endfor
		acc${i} += t;
	}
	%endfor
</%def>

## Reduction where one of the axes being reduced is 'x'. This uses the standard
## GPU reduction algorithm for continuous arrays.
##
## This kernel is meant to be launched the following configration:
##  grid.y: spans the axis not being reduced over
##  block.x: spans as much of the X axis as possible
##
<%def name="reduction_x(axis, num_inputs, stats, out_type='float', block_size=1024, want_offset=False)">
	int lx = get_local_id(0);	// ID inside the current block
	int gx = get_global_id(0);	// global X coordinate within the subdomain
	int bx = get_group_id(0);	// block index
	int g_scan = get_global_id(1);

	%for i in range(len(stats)):
		__shared__ ${out_type} sdata${i}[${block_size}];
		sdata${i}[lx] = 0.0f;
		${out_type} acc${i} = 0.0f;
	%endfor

	// Skip ghost nodes.
	if (gx >= ${lat_nx - 1}) {
		return;
	}

	// Read data from global memory into sdata.
	// Only load global memory data for real nodes.
	if (gx > 0) {
		%if dim == 2:
			// +1 shift due to ghost nodes.
			int gi = getGlobalIdx(gx, g_scan + 1);
			${_compute_stats(num_inputs, stats, out_type)}
		%else:
			<% other_nx = lat_ny if axis == 2 else lat_nz %>
			for (int g_other = 1; g_other < ${other_nx - 1}; g_other++) {
				%if axis == 1:
					int gi = getGlobalIdx(gx, g_scan + 1, g_other);
				%else:
					int gi = getGlobalIdx(gx, g_other, g_scan + 1);
				%endif
				${_compute_stats(num_inputs, stats, out_type)}
			}
		%endif
		%for i in range(len(stats)):
			sdata${i}[lx] = acc${i};
		%endfor
	}
	__syncthreads();

	// Cross-warp aggregation.
	%for stride in (256, 128, 64):
		%if block_size >= stride * 2:
			if (lx < ${stride}) {
				%for i in range(len(stats)):
					sdata${i}[lx] = sdata${i}[lx] + sdata${i}[lx + ${stride}];
				%endfor
				__syncthreads();
			}
		%endif
	%endfor

	// Aggregation within the first warp -- no synchronization necessary.
	if (lx < 32) {
		%for i in range(len(stats)):
			// 'volatile' required according to Fermi compatibility guide 1.2.2
			volatile ${out_type} *smem${i} = sdata${i};
		%endfor
		%for stride in (32, 16, 8, 4, 2, 1):
			%if block_size >= stride * 2:
				if (lx < ${stride}) {
					%for i in range(len(stats)):
						smem${i}[lx] = smem${i}[lx] + smem${i}[lx + ${stride}];
					%endfor
				}
			%endif
		%endfor
	}

	if (lx == 0) {
		%for i in range(len(stats)):
			out${i}[g_scan * gridDim.x + bx ${'+ offset' if want_offset else ''}] = sdata${i}[0];
		%endfor
	}
</%def>

## TODO(michalj): Make this multistage so that many of these kernels can be run in parallel.
##
## Reduction where the 'x' axis is not being reduced over.
## This kernel is meant to be launched using:
##  x: spans the X not reduced over (via grid)
<%def name="reduction_nox(num_inputs, stats, out_type='float', want_offset=False)">
	%for i in range(len(stats)):
		${out_type} acc${i} = 0.0f;
	%endfor

	int gx = get_global_id(0);

	// Skip ghost nodes.
	if (gx >= ${lat_nx - 1} || gx == 0) {
		return;
	}

	// TODO(michalj): Modify the 1 below for variable ghost node layer.
	for (int gy = 1; gy < ${lat_ny - 1}; gy++) {
		%if dim == 3:
			for (int gz = 1; gz < ${lat_nz - 1}; gz++) {
				int gi = getGlobalIdx(gx, gy, gz);
				${_compute_stats(num_inputs, stats, out_type)}
			}
		%else:
			int gi = getGlobalIdx(gx, gy);
			${_compute_stats(num_inputs, stats, out_type)}
		%endif
	}

	// Skip ghost nodes.
	%for i in range(len(stats)):
		out${i}[gx - 1 ${'+ offset' if want_offset else ''}] = acc${i};
	%endfor
</%def>

<%def name="aggregate_slice(name, axis, num_inputs=1, stats=[[(0,1)]], out_type='float', block_size=1024)">
${kernel} void AggregateSlice${name}(
	%for i in range(num_inputs):
		${global_ptr} float *f${i},
	%endfor
	%for i in range(len(stats)):
		${global_ptr} ${out_type} *out${i} ${',' if i < len(stats) - 1 else ''}
	%endfor
	, int position,
	, int restart)
{
	%for i in range(len(stats)):
		${out_type} acc${i} = 0.0f;
	%endfor

	int ga = get_global_id(0);
	int gb = get_global_id(1);

	// Skip ghost nodes
	%if axis == 0:
		if (ga >= ${lat_ny - 1} || ga == 0 || gb == 0 || gb >= ${lat_nz - 1}) {
			return;
		}
		int gi = getGlobalIdx(position, ga, gb);
		<% width = lat_ny %>
	%elif axis == 1:
		if (ga >= ${lat_nx - 1} || ga == 0 || gb == 0 || gb >= ${lat_nz - 1}) {
			return;
		}
		int gi = getGlobalIdx(ga, position, gb);
		<% width = lat_nx %>
	%else:
		if (ga >= ${lat_nx - 1} || ga == 0 || gb == 0 || gb >= ${lat_ny - 1}) {
			return;
		}
		int gi = getGlobalIdx(ga, gb, position);
		<% width = lat_nx %>
	%endif
	${_compute_stats(num_inputs, stats, out_type)}

	<%
		if restart:
			op = '='
		else:
			op = '+='
	%>

	// Skip ghost nodes.
	%for i in range(len(stats)):
		out${i}[(ga - 1) + (gb - 1) * ${width - 2}] ${op} acc${i};
	%endfor
}
</%def>

## Builds a custom reduction kernel.
##
## Args:
##  name: string, unique name of the reduction kernel
##  axis: axis ID over which NOT to reduce
##  num_inputs: number of input fields
##  stats: list of statistics to compute. Every statistic is a list of tuples:
##	   (field_id, power)
##     Examples:
##		- [(0, 1)] indicates f_0
##		- [(0, 2)] indicates f_0^2
##		- [(0, 1), (1, 1)] indicates f_0 f_1
##		- [(0, 2), (1, 1)] indicates f_0^2 f_1
##  out_type: 'float' or 'double', identifies the precision of the reduction
##            operation and the final value
##  block_size: CUDA block size for the reduction kernel
<%def name="reduction(name, axis, num_inputs=1, stats=[[(0,1)]], out_type='float', block_size=1024, want_offset=False)">
<%
	need_finalize = axis != 0 and lat_nx >= block_size
	need_offset = want_offset and not need_finalize
%>

${kernel} void Reduce${name}(
	%for i in range(num_inputs):
		${global_ptr} float *f${i},
	%endfor
	%for i in range(len(stats)):
		${global_ptr} ${out_type} *out${i} ${',' if i < len(stats) - 1 else ''}
	%endfor
	%if need_offset:
		, int offset
	%endif
) {
	%if axis == 0:
		${reduction_nox(num_inputs, stats, out_type, need_offset)}
	%else:
		${reduction_x(axis, num_inputs, stats, out_type, block_size, need_offset)}
	%endif
}

%if need_finalize:
// Reduction is 2-step -- Reduce${name} is applied first, and FinalizeReduce${name}
// has to be called to compute the final value.
<%
	import math
	real_block_size = (lat_nx + block_size - 1) / block_size
	# Round to the next nearest power of two.
	phys_block_size = int(pow(2, math.ceil(math.log(real_block_size, 2))))
%>
${kernel} void FinalizeReduce${name}(
		${global_ptr} ${out_type} *in,
		${global_ptr} ${out_type} *out
%if want_offset:
		, int offset
%endif
) {
	int gx = get_local_id(0);	// ID inside the current block
	int g_scan = get_global_id(1);

	__shared__ ${out_type} sdata[${phys_block_size}];
	// Read data from global memory into sdata.
	${out_type} acc = 0.0f;

	if (gx < ${real_block_size}) {
		int gi = gx + g_scan * ${real_block_size};
		acc = acc + in[gi];
	}
	sdata[gx] = acc;
	__syncthreads();

	// Cross-warp aggregation.
	%for stride in (256, 128, 64):
		%if phys_block_size >= stride * 2:
			if (gx < ${stride}) {
				sdata[gx] = sdata[gx] + sdata[gx + ${stride}];
				__syncthreads();
			}
		%endif
	%endfor

	// Aggregation within the first warp -- no synchronization necessary.
	if (gx < 32) {
		// 'volatile' required according to Fermi compatibility guide 1.2.2
		volatile ${out_type} *smem = sdata;
		%for stride in (32, 16, 8, 4, 2, 1):
			%if phys_block_size >= stride * 2:
				if (gx < ${stride}) {
					smem[gx] = smem[gx] + smem[gx + ${stride}];
				}
			%endif
		%endfor
	}

	if (gx == 0) {
		out[g_scan ${'+ offset' if want_offset else ''}] = sdata[0];
	}
}
%endif
</%def>


${kernel} void ExtractSlice(int axis, int position,
	${kernel_args_1st_moment('iv')}
	${global_ptr} float *out) {
	int c0 = get_global_id(0) + 1;
	int c1 = get_global_id(1) + 1;
	int gi;

	if (axis == 0) {
		if (c0 >= ${lat_ny-1} || c1 >= ${lat_nz-1}) {
			return;
		}
		gi = getGlobalIdx(1 + position, c0, c1);
	} else if (axis == 1) {
		if (c0 >= ${lat_nx-1} || c1 >= ${lat_nz-1}) {
			return;
		}
		gi = getGlobalIdx(c0, 1 + position, c1);
	} else {
		if (c0 >= ${lat_nx-1} || c1 >= ${lat_ny-1}) {
			return;
		}
		gi = getGlobalIdx(c0, c1, 1 + position);
	}

	float vx = ivx[gi];
	float vy = ivy[gi];
	float vz = ivz[gi];
	out[(c1 - 1) * ${lat_nx-2} + (c0 - 1)] = sqrtf(vx * vx + vy * vy + vz * vz);
}

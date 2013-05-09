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
<%def name="reduction_x(axis, num_inputs, stats, out_type='float', block_size=1024)">
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
			%for stride in (32, 16, 8, 4, 2, 1):
				%if block_size >= stride * 2:
					if (lx < ${stride}) {
						smem${i}[lx] = smem${i}[lx] + smem${i}[lx + ${stride}];
					}
				%endif
			%endfor
		%endfor
	}

	if (lx == 0) {
		%for i in range(len(stats)):
			out${i}[g_scan * gridDim.x + bx] = sdata${i}[0];
		%endfor
	}
</%def>

## TODO(michalj): Make this multistage so that many of these kernels can be run in parallel.
##
## Reduction where the 'x' axis is not being reduced over.
## This kernel is meant to be launched using:
##  x: spans the X not reduced over (via grid)
<%def name="reduction_nox(num_inputs, stats, out_type='float')">
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
		out${i}[gx - 1] = acc${i};
	%endfor
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
##		- [(0, 2), (1, 1)] indciates f_0^2 f_1
##  out_type: 'float' or 'double', identifies the precision of the reduction
##            operation and the final value
##  block_size: CUDA block size for the reduction kernel
<%def name="reduction(name, axis, num_inputs=1, stats=[[(0,1)]], out_type='float', block_size=1024)">
${kernel} void Reduce${name}(
	%for i in range(num_inputs):
		${global_ptr} float *f${i},
	%endfor
	%for i in range(len(stats)):
		${global_ptr} ${out_type} *out${i}
	%endfor
) {
	%if axis == 0:
		${reduction_nox(num_inputs, stats, out_type)}
	%else:
		${reduction_x(axis, num_inputs, stats, out_type, block_size)}
	%endif
}

%if axis != 0 and lat_nx >= block_size:
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
		${global_ptr} ${out_type} *out) {
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
		out[g_scan] = sdata[0];
	}
}
%endif

</%def>

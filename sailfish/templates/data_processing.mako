## Reduction where one of the axes being reduced is 'x'. This uses the standard
## GPU reduction algorithm for continuous arrays.
##
## This kernel is meant to be launched the following configration:
##  grid.y: spans the axis not being reduced over
##  block.x: spans as much of the X axis as possible
##

<%def name="reduction_x(name, axis, out_type='float', block_size=1024)">
	int gx = get_local_id(0);	// ID inside the current block
	int g_scan = get_global_id(1);

	__shared__ ${out_type} sdata[${block_size}];
	sdata[gx] = 0.0f;

	## Only a reduction within a single block is supported.
	<% assert block_size >= lat_nx %>

	// Skip ghost nodes.
	if (gx >= ${lat_nx - 1}) {
		return;
	}

	// Read data from global memory into sdata.
	${out_type} acc = 0.0f;

	// Only load global memory data for real nodes.
	if (gx > 0) {
		%if dim == 2:
			// +1 shift due to ghost nodes.
			int gi = getGlobalIdx(gx, g_scan + 1);
			acc = acc + f[gi];
		%else:
			<% other_nx = lat_ny if axis == 2 else lat_nz %>
			for (int g_other = 1; g_other < ${other_nx - 1}; g_other++) {
				%if axis == 1:
					int gi = getGlobalIdx(gx, g_scan + 1, g_other);
				%else:
					int gi = getGlobalIdx(gx, g_other, g_scan + 1);
				%endif
				acc = acc + f[gi];
			}
		%endif
		sdata[gx] = acc;
	}
	__syncthreads();

	// Cross-warp aggregation.
	%for stride in (256, 128, 64):
		%if block_size >= stride * 2:
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
			%if block_size >= stride * 2:
				smem[gx] = smem[gx] + smem[gx + ${stride}];
			%endif
		%endfor
	}

	if (gx == 0) {
		out[g_scan] = sdata[0];
	}
</%def>

## TODO(michalj): Make this multistage so that many of these kernels can be run in parallel.
##
## Reduction where the 'x' axis is not being reduced over.
## This kernel is meant to be launched using:
##  x: spans the X not reduced over (via grid)
<%def name="reduction_nox(name, out_type='float')">
	${out_type} acc = 0.0f;
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
				acc += f[gi];
			}
		%else:
			int gi = getGlobalIdx(gx, gy);
			acc += f[gi];
		%endif
	}

	// Skip ghost nodes.
	out[gx - 1] = acc;
</%def>

## Builds a custom reduction kernel.
##
## Args:
##  name: string, unique name of the reduction kernel
##  axis: axis ID over which NOT to reduce
##  out_type: 'float' or 'double', identifies the precision of the reduction
##            operation and the final value
##  block_size: CUDA block size for the reduction kernel
<%def name="reduction(name, axis, out_type='float', block_size=1024)">
${kernel} void Reduce${name}(${global_ptr} float *f,
							 ${global_ptr} ${out_type} *out) {
	%if axis == 0:
		${reduction_nox(name, out_type)}
	%else:
		${reduction_x(name, axis, out_type, block_size)}
	%endif
}
</%def>

<%!
  from sailfish import sym
  import sailfish.node_type as nt
%>

<%page args="bgk_args_decl"/>
<%namespace file="code_common.mako" import="cex"/>

<%def name="kernel_args_1st_moment(name, const=False)">
	${global_ptr} ${const_ptr if const else ''} float *__restrict__ ${name}x,
	${global_ptr} ${const_ptr if const else ''} float *__restrict__ ${name}y,
%if dim == 3:
	${global_ptr} ${const_ptr if const else ''} float *__restrict__ ${name}z,
%endif
</%def>

<%def name="iteration_number_if_required()" filter="trim">
	%if needs_iteration_num:
		, unsigned int iteration_number
	%endif
</%def>

<%def name="nodes_array_if_required()" filter="trim">
	%if node_addressing == 'indirect':
		${global_ptr} ${const_ptr} int *__restrict__ nodes,
	%endif
</%def>

<%def name="nodes_array_arg_if_required()" filter="trim">
	%if node_addressing == 'indirect':
		nodes,
	%endif
</%def>

<%def name="dense_gi_if_required()" filter="trim">
	%if node_addressing == 'indirect':
		, int dense_gi
	%endif
</%def>

<%def name="dense_gi_arg_if_required()" filter="trim">
	%if node_addressing == 'indirect':
		, dense_gi
	%endif
</%def>

<%def name="iteration_number_arg_if_required()" filter="trim">
	%if needs_iteration_num:
		, iteration_number
	%endif
</%def>

<%def name="scratch_space_if_required()" filter="trim">
	%if scratch_space:
		, ${global_ptr} float *__restrict__ node_scratch_space
	%endif
</%def>

<%def name="scratch_space_arg_if_required()" filter="trim">
	%if scratch_space:
		, node_scratch_space
	%endif
</%def>

<%def name="scalar_field_if_required(name, required)" filter="trim">
	%if required:
		, ${global_ptr} float *__restrict__ ${name}
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
<%def name="local_indices(no_outside=True)">
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
		int gx = get_global_id(0) % ${grid_nx};
		int gy = get_global_id(0) / ${grid_nx};
		int gz = get_global_id(1);
	%endif

	int gi = ${get_global_idx()};

	%if no_outside:
		// Nothing to do if we're outside of the simulation domain.
		if (gx > ${lat_nx-1}) {
			return;
		}
	%endif
</%def>

## Defines local indices for kernels that can be split into bulk and boundary.
## Automatically handles the case when the split is disabled.
##
## Args:
##  no_outside: if True, will call 'return' in case the indices end up pointing
##              outside of the simulation domain
<%def name="local_indices_split(no_outside=True)">
	%if boundary_size > 0:
		int gx, gy, lx, gi;
		%if dim == 3:
			int gz;
		%endif

		if (options & OPTION_BULK) {
			${local_indices_bulk(no_outside=no_outside)}
		} else {
			${local_indices_boundary(no_outside=no_outside)}
		}
	%else:
		${local_indices(no_outside=no_outside)}
	%endif
</%def>

<%def name="indirect_index(orig='dense_gi', position_warning=True)">
	%if node_addressing == 'indirect':
		// In the indirect access mode, the original global index is used for
		// the 'nodes' table, while the index from that table is used for all
		// fields and distributions.
		%if orig is not None:
			int ${orig} = gi;
		%endif
		gi = nodes[gi];
		if (gi == INVALID_NODE) {
			return;
		}
		if (gi >= DIST_SIZE || gi < 0) {
			%if position_warning:
				%if dim == 3:
					printf("invalid index %d @ %d %d %d\n", gi, gx, gy, gz);
				%else:
					printf("invalid index %d @ %d %d\n", gi, gx, gy);
				%endif
			%else:
				printf("invalid node index detected\n");
			%endif
			return;
		}
	%endif
</%def>

<%def name="shared_mem_propagation_vars()">
	%if not propagate_on_read and propagation_enabled and node_addressing != 'indirect':
		%if supports_shuffle and propagate_with_shuffle:
			// Shared variables for cross-warp propagation.
			%for i in sym.get_prop_dists(grid, 1):
				${shared_var} float prop_${grid.idx_name[i]}[${(block_size + warp_size - 1) / warp_size}];
			%endfor
			%for i in sym.get_prop_dists(grid, -1):
				${shared_var} float prop_${grid.idx_name[i]}[${(block_size + warp_size - 1) / warp_size}];
			%endfor
		%else:
			// Shared variables for in-block propagation
			%for i in sym.get_prop_dists(grid, 1):
				${shared_var} float prop_${grid.idx_name[i]}[BLOCK_SIZE];
			%endfor
			%for i in sym.get_prop_dists(grid, 1):
				#define prop_${grid.idx_name[grid.idx_opposite[i]]} prop_${grid.idx_name[i]}
			%endfor
		%endif
	%endif
</%def>

<%def name="load_node_type()">
	int ncode = map[gi];
	int type = decodeNodeType(ncode);

	// Unused nodes do not participate in the simulation.
	if (isExcludedNode(type))
		return;

	int orientation = decodeNodeOrientation(ncode);
</%def>

<%def name="check_invalid_values()">
	%if gpu_check_invalid_values:
		## Grad outflow nodes use invalid values to tag directions lacking distribution
		## data.
		if (isWetNode(type) ${'&& !isNTGradFreeflow(type)' if nt.NTGradFreeflow in node_types else ''}) {
			checkInvalidValues(&d0, ${position()});
		}
	%endif
</%def>

<%def name="save_macro_fields(density=True, velocity=True)">
	// Only save the macroscopic quantities if requested to do so.
	if ((options & OPTION_SAVE_MACRO_FIELDS) && isWetNode(type)
		## Nodes using the Grad approximation use the velocity from the
		## previous time step to compute the approximated distributions.
		${'|| isNTGradFreeflow(type)' if nt.NTGradFreeflow in node_types else ''}) {
		gg0m0[gi] = g0m0 ${' +1.0f' if config.minimize_roundoff else ''};

		%if not initialization and velocity:
			ovx[gi] = v[0];
			ovy[gi] = v[1];
			${'ovz[gi] = v[2]' if dim == 3 else ''};
		%endif
	}
</%def>

## Defines local indices for bulk kernels.
## This is the same as local_indices(), but with proper offsets to skip
## the boundary.
<%def name="local_indices_bulk(no_outside=True)">
	lx = get_local_id(0);	// ID inside the current block
	<%
		if block.has_face_conn(block.X_LOW) or block.periodic_x:
			xoff = block_size
		else:
			xoff = 0

		if block.has_face_conn(block.Y_LOW) or block.periodic_y:
			yoff = boundary_size
		else:
			yoff = 0
	%>
	%if dim == 2:
		gx = ${xoff} + get_global_id(0);
		gy = ${yoff} + get_group_id(1);
	%else:
		<%
			if block.has_face_conn(block.Z_LOW) or block.periodic_z:
				zoff = boundary_size
			else:
				zoff = 0

			if block.has_face_conn(block.X_HIGH) or block.periodic_x:
				xconns = xoff + block_size
				padding = grid_nx - lat_nx
				if block_size - padding >= boundary_size:
					xconns += block_size
			else:
				xconns = xoff
		%>
		## Also see how _kernel_grid_bulk is set in block_runnner.py
		gx = ${xoff} + get_global_id(0) % ${grid_nx - xconns};
		gy = ${yoff} + get_global_id(0) / ${grid_nx - xconns};
		gz = ${zoff} + get_global_id(1);
	%endif

	gi = ${get_global_idx()};

	%if no_outside:
		// Nothing to do if we're outside of the simulation domain.
		if (gx > ${lat_nx-1}) {
			return;
		}
	%endif
</%def>

## Defines local indices for boundary kernels.
<%def name="local_indices_boundary(no_outside=True)">
	lx = get_local_id(0);	// ID inside the current block
	int gid = get_group_id(0) + get_group_id(1) * get_global_size(0) / get_local_size(0);

	<%
		# Code common to 2D and 3D cases.
		has_ylow = int(block.has_face_conn(block.Y_LOW) or block.periodic_y)
		has_yhigh = int(block.has_face_conn(block.Y_HIGH) or block.periodic_y)
		has_xlow = int(block.has_face_conn(block.X_LOW) or block.periodic_x)
		has_xhigh = int(block.has_face_conn(block.X_HIGH) or block.periodic_x)
		y_conns = has_ylow + has_yhigh
		padding = grid_nx - lat_nx
		bns = boundary_size

		if bool(has_xhigh) and block_size - padding >= boundary_size:
			aux_ew = 1	# 2 blocks on the right due to misalignment
		else:
			aux_ew = 0	# 1 block on the right
	%>
	%if dim == 2:
		<%
			xblocks = grid_nx / block_size
			yblocks = arr_ny - y_conns * boundary_size

			bottom_idx = has_ylow * bns * xblocks
			left_idx = bottom_idx + has_yhigh * bns * xblocks
			right_idx = left_idx + has_xlow * yblocks
			right2_idx = right_idx + has_xhigh * yblocks
			max_idx = right2_idx + aux_ew * yblocks
		%>
		// x: ${xblocks}, y: ${yblocks}
		if (0) {;}
		%if block.has_face_conn(block.Y_LOW) or block.periodic_y:
			else if (gid < ${bottom_idx}) {
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gy = gid / ${xblocks};
			}
		%endif
		%if block.has_face_conn(block.Y_HIGH) or block.periodic_y:
			else if (gid < ${left_idx}) {
				gid -= ${bottom_idx};
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gy = ${lat_ny-1} - gid / ${xblocks};
			}
		%endif
		%if block.has_face_conn(block.X_LOW) or block.periodic_x:
			else if (gid < ${right_idx}) {
				gx = lx;
				gy = gid + ${has_ylow * boundary_size - left_idx};
			}
		%endif
		%if block.has_face_conn(block.X_HIGH) or block.periodic_x:
			else if (gid < ${right2_idx}) {
				gx = ${grid_nx - block_size} + lx;
				gy = gid + ${has_ylow * boundary_size - right_idx};
			} else if (gid < ${max_idx}) {
				gx = ${grid_nx - 2*block_size} + lx;
				gy = gid + ${has_ylow * boundary_size - right2_idx};
			}
		%endif
		else {
			return;
		}
	%else:
		<%
			has_zlow = int(block.has_face_conn(block.Z_LOW) or block.periodic_z)
			has_zhigh = int(block.has_face_conn(block.Z_HIGH) or block.periodic_z)
			z_conns = has_zlow + has_zhigh

			xblocks = grid_nx / block_size
			yblocks = arr_ny - y_conns * boundary_size
			zblocks = arr_nz - z_conns * boundary_size
			yz_blocks = yblocks * zblocks

			x_face = yblocks * zblocks
			y_face = xblocks * zblocks
			z_face = xblocks * arr_ny

			zlow_idx = has_zlow * z_face * bns
			zhigh_idx = zlow_idx + has_zhigh * z_face * bns
			ylow_idx = zhigh_idx + has_ylow * y_face * bns
			yhigh_idx = ylow_idx + has_yhigh * y_face * bns
			xlow_idx = yhigh_idx + has_xlow * x_face
			xhigh_idx = xlow_idx + has_xhigh * x_face
			max_idx = xhigh_idx  + aux_ew * x_face
		%>
		// x: ${xblocks}, y: ${yblocks}, z: ${zblocks}
		if (0) {;}
		%if block.has_face_conn(block.Z_LOW) or block.periodic_z:
			// B face.  Face area is arr_nx * arr_ny.
			else if (gid < ${zlow_idx}) {
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gid = gid / ${xblocks};
				gy = gid % ${arr_ny};
				gz = gid / ${arr_ny};
			}
		%endif
		%if block.has_face_conn(block.Z_HIGH) or block.periodic_z:
			// T face.  Face area is arr_nx * arr_ny.
			else if (gid < ${zhigh_idx}) {
				gid -= ${zlow_idx};
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gid = gid / ${xblocks};
				gy = gid % ${arr_ny};
				gz = ${arr_nz-1} - gid / ${arr_ny};
			}
		%endif
		%if block.has_face_conn(block.Y_LOW) or block.periodic_y:
			// S face.
			else if (gid < ${ylow_idx}) {
				gid -= ${zhigh_idx};
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gid = gid / ${xblocks};
				gz = gid % ${zblocks} + ${has_zlow * boundary_size};
				gy = gid / ${zblocks};
			}
		%endif
		%if block.has_face_conn(block.Y_HIGH) or block.periodic_y:
			// N face.
			else if (gid < ${yhigh_idx}) {
				gid -= ${ylow_idx};
				gx = (gid % ${xblocks}) * ${block_size} + lx;
				gid = gid / ${xblocks};
				gz = gid % ${zblocks} + ${has_zlow * boundary_size};
				gy = ${arr_ny-1} - gid / ${zblocks};
			}
		%endif
		%if block.has_face_conn(block.X_LOW) or block.periodic_x:
			// W face.
			else if (gid < ${xlow_idx}) {
				gid -= ${yhigh_idx};
				gx = lx;
				gy = gid % ${yblocks} + ${has_ylow * boundary_size};
				gz = gid / ${yblocks} + ${has_zlow * boundary_size};
			}
		%endif
		%if block.has_face_conn(block.X_HIGH) or block.periodic_x:
			// E face (part 1)
			else if (gid < ${xhigh_idx}) {
				gid -= ${xlow_idx};
				gx = ${grid_nx - block_size} + lx;
				gy = gid % ${yblocks} + ${has_ylow * boundary_size};
				gz = gid / ${yblocks} + ${has_zlow * boundary_size};
			// E face (part 2)
			} else if (gid < ${max_idx}) {
				gid -= ${xhigh_idx};
				gx = ${grid_nx - 2*block_size} + lx;
				gy = gid % ${yblocks} + ${has_ylow * boundary_size};
				gz = gid / ${yblocks} + ${has_zlow * boundary_size};
			}
		%endif
		else {
			return;
		}
	%endif

	gi = ${get_global_idx()};

	%if no_outside:
		// Nothing to do if we're outside of the simulation domain.
		if (gx > ${lat_nx-1}) {
			return;
		}
	%endif
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
#define OPTION_SAVE_MACRO_FIELDS 1
#define OPTION_BULK 2

## Indicates an unallocated node when using indirect node addressing.
#define INVALID_NODE 0xffffffff

#define DT 1.0f

%if backend == 'cuda':
#include <stdio.h>
%endif

%for name, val in constants.iteritems():
	${const_var} float ${name} = ${val}f;
%endfor

%if node_params:
	// Additional geometry parameters (velocities, pressures, etc)
	${const_var} float node_params[${len(node_params)}] = {
	%for param in node_params:
		${cex(param)},
	%endfor
	};
%else:
	${const_var} float node_params[1] = {0};
%endif

<%namespace file="opencl_compat.mako" import="*" name="opencl_compat"/>

%if not unit_test:
<%namespace file="boundary.mako" import="*" name="boundary"/>
<%namespace file="relaxation.mako" import="*" name="relaxation"/>
%endif

<%def name="position_decl(prefix='g')">
	int ${prefix}x, int ${prefix}y
	%if dim == 3:
		, int ${prefix}z
	%endif
</%def>

<%def name="position()">
	gx, gy
	%if dim == 3:
		, gz
	%endif
</%def>

${opencl_compat.body()}
<%include file="geo_helpers.mako"/>

%if not unit_test:
	${boundary.body()}
	${relaxation.body(bgk_args_decl)}
%endif

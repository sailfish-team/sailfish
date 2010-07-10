//
// A kernel to update the position of tracer particles.
//
// Each thread updates the position of a single particle using Euler's algorithm.
//
${kernel} void LBMUpdateTracerParticles(${global_ptr} float *dist, ${global_ptr} int *map,
		${global_ptr} float *x, ${global_ptr} float *y \
%if dim == 3:
	, ${global_ptr} float *z \
%endif
		)
{
	float rho, v[${dim}];

	int gi = get_global_id(0);
	float cx = x[gi];
	float cy = y[gi];

	int ix = (int)(cx);
	int iy = (int)(cy);

	%if dim == 3:
		float cz = z[gi];
		int iz = (int)(cz);

		if (iz < 0)
			iz  = 0;

		if (iz > ${lat_nz-1})
			iz = ${lat_nz-1};
	%endif

	// Sanity checks.
	if (iy < 0)
		iy = 0;

	if (ix < 0)
		ix = 0;

	if (ix > ${lat_nx-1})
		ix = ${lat_nx-1};

	if (iy > ${lat_ny-1})
		iy = ${lat_ny-1};

	%if dim == 2:
		int idx = ix + ${lat_nx}*iy;
	%else:
		int idx = ix + ${lat_nx}*iy + ${lat_nx*lat_ny}*iz;
	%endif

	int ncode = map[idx];
	int type = decodeNodeType(ncode);
	int orientation = decodeNodeOrientation(ncode);

	// Unused nodes do not participate in the simulation.
	if (isUnusedNode(type)) {
		return;
	}

	Dist fc;

## HACK: If a call to getDist() is made below, the overall performance of the simulation
## will be decreased by a factor of 2, regardless of whether this kernel is even executed.
## This might be caused by the NVIDIA OpenCL compiler not inlining the getDist function.
## To avoid the performance loss, we temporarily inline getDist manually.
	// getDist(&fc, dist, idx);

	%for i, dname in enumerate(grid.idx_name):
		fc.${dname} = dist[idx + DIST_SIZE*${i}];
	%endfor

	## FIXME: We just need the velocity here.
	getMacro(&fc, ncode, type, orientation, &rho, v);

	cx = cx + v[0] * DT;
	cy = cy + v[1] * DT;
	%if dim == 3:
		cz = cz + v[2] * DT;
	%endif

	// Periodic boundary conditions.
	if (cx > ${lat_nx})
		cx = 0.0f;

	if (cy > ${lat_ny})
		cy = 0.0f;

	if (cx < 0.0f)
		cx = (float)${lat_nx};

	if (cy < 0.0f)
		cy = (float)${lat_ny};

	%if dim == 3:
		if (cz > ${lat_nz})
			cz = 0.0f;

		if (cz < 0.0f)
			cz = (float)(${lat_nz});

		z[gi] = cz;
	%endif
	x[gi] = cx;
	y[gi] = cy;
}


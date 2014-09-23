## Code supporting the immersed bondary method.
<%namespace file="mako_utils.mako" import="*"/>
<%namespace file="kernel_common.mako" import="kernel_args_1st_moment" />

// Interpolates fluid velocity and updates particle positions.
${kernel} void UpdateParticlePosition(
		${kernel_args_1st_moment('fluid_v')}
		${kernel_args_1st_moment('r')}
		const int num_particles) {
	// Particle ID.
	const int pid = get_global_id(0);
	if (pid >= num_particles) {
		return;
	}

	// Particle positions.
	const float lrx = rx[pid];
	const float lry = ry[pid];
	${ifdim3('const float lrz = rz[pid];')}

	const int xmin = lrx;
	const int ymin = lry;
	${ifdim3('const int zmin = lrz;')}

	// Particle velocity.
	float lvx = 0.0f, lvy = 0.0f ${ifdim3(', lvz = 0.0f')};

	// \phi_2 kernel: 1 - |x| for |x| <= 1. Diameter of support is 2 nodes.
	${ifdim3('for (int z = zmin; z < zmin + 1; z++)')}
	{
		for (int y = ymin; y <= ymin + 1; y++) {
			for (int x = xmin; x <= xmin + 1; x++) {
				const float dx = fabsf(lrx - x);
				const float dy = fabsf(lry - y);
				${ifdim3('const float dz = fabsf(lrz - z);')}
				const float w = (1.0f - dx) * (1.0f - dy) ${ifdim3('* (1.0f - dz)')};
				const int idx = getGlobalIdx(x, y ${ifdim3(', z')});
				lvx += fluid_vx[idx] * w;
				lvy += fluid_vy[idx] * w;
				${ifdim3('lvz += fluid_vz[idx] * w;')}
			}
		}
	}

	// Euler step.
	rx[pid] = lrx + lvx;
	ry[pid] = lry + lvy;
	${ifdim3('rz[pid] = lrz + lvz;')}
}

// Generate particle forces.
${kernel} void SpreadParticleForcesStiff(
		${global_ptr} float* stiffness,
		${kernel_args_1st_moment('r')}
		${kernel_args_1st_moment('ref')}
		${kernel_args_1st_moment('force')}
		const int num_particles)
{
	// Particle ID.
	const int pid = get_global_id(0);
	if (pid >= num_particles) {
		return;
	}

	// Particle positions.
	const float lrx = rx[pid];
	const float lry = ry[pid];
	${ifdim3('const float lrz = rz[pid];')}

	// Particle reference positions.
	const float lref_x = refx[pid];
	const float lref_y = refy[pid];
	${ifdim3('const float lref_z = refz[pid];')}

	const float lstiffness = stiffness[pid];

	const int xmin = lrx;
	const int ymin = lry;
	${ifdim3('const int zmin = lrz;')}

	// \phi_2 kernel: 1 - |x| for |x| <= 1. Diameter of support is 2 nodes.
	${ifdim3('for (int z = zmin; z < zmin + 1; z++)')}
	{
		for (int y = ymin; y <= ymin + 1; y++) {
			for (int x = xmin; x <= xmin + 1; x++) {
				const float dx = fabsf(lrx - x);
				const float dy = fabsf(lry - y);
				${ifdim3('const float dz = fabsf(lrz - z)')};
				const float w = (1.0f - dx) * (1.0f - dy) ${ifdim3('* (1.0f - dz)')};
				const int idx = getGlobalIdx(x, y ${ifdim3(', z')});
				// Hooke's law.
				atomicAdd(forcex + idx, -lstiffness * (lrx - lref_x) * w);
				atomicAdd(forcey + idx, -lstiffness * (lry - lref_y) * w);
				${ifdim3('atomicAdd(forcez + idx, -lstiffness * (lrz - lref_z) * w);')};
			}
		}
	}
}

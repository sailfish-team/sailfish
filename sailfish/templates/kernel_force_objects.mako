// Copies momentum transfer for a force object into a linear buffer
// so that a force can be computed easily via a sum reduction.
// TODO(michalj): Fuse this with summation to improve performance.
${kernel} void ComputeForceObjects(
	${global_ptr} ${const_ptr} int *__restrict__ idx,
	${global_ptr} ${const_ptr} int *__restrict__ idx2,
	${global_ptr} ${const_ptr} float *__restrict__ dist,
	${global_ptr} float *out,
	const int max_idx
	)
{
	const int gidx = get_global_id(0);
	if (gidx >= max_idx) {
		return;
	}
	const int gi = idx[gidx];
	const int gi2 = idx2[gidx];
	const float mx = dist[gi] + dist[gi2];
/*
	if (gidx == 95) {
		int gy, gx;
		decodeGlobalIdx(gi % ${dist_size}, &gx, &gy);
		printf("%d: %d %d %d | %d\n", gidx, gx, gy, gi, gi / ${dist_size});
		decodeGlobalIdx(gi2 % ${dist_size}, &gx, &gy);
		printf("%d: %d %d %d | %d\n", gidx, gx, gy, gi2, gi2 / ${dist_size});
		printf("%f %f\n", dist[gi], dist[gi2]);
	}*/
	out[gidx] = mx;
}

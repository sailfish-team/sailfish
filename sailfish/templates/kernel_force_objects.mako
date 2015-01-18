// Copies momentum transfer for a force object into a linear buffer
// so that a force can be computed easily via a sum reduction.
// TODO(michalj): Fuse this with summation to improve performance.
${kernel} void ComputeForceObjects(
  ${global_ptr} ${const_ptr} unsigned int *__restrict__ idx,
  ${global_ptr} ${const_ptr} unsigned int *__restrict__ idx2,
  ${global_ptr} ${const_ptr} float *__restrict__ dist,
  ${global_ptr} float *out,
  const unsigned int max_idx
  )
{
  const unsigned int gidx = get_global_id(0);
  if (gidx >= max_idx) {
    return;
  }
  const unsigned int gi = idx[gidx];
  const unsigned int gi2 = idx2[gidx];
  const float mx = dist[gi] + dist[gi2];
  out[gidx] = mx;
}

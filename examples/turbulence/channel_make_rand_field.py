"""
Creates a random, divergence free perturbation for channel simulations.

The data is saved in numpy arrays. They can be used to limit memory usage
when the channel simulation is starting.

Usage:
    ./channel_make_rand_field.py NX NY NZ H
"""

import numpy as np
import scipy.ndimage.filters
import sys

# Buffer size (used to make the random perturbation continuous
# along the streamwise direction.
B = 40
hB = B / 2

def make_rand(NX, NY, NZ, H):
    n1 = np.random.random((NZ + B, NY + B, NX)).astype(np.float32) * 2.0 - 1.0
    # Make the field continous along the streamwise and spanwise direction.
    n1[-hB:,:,:] = n1[hB:B,:,:]
    n1[:hB,:,:] = n1[-B:-hB,:,:]

    n1[:,-hB:,:] = n1[:,hB:B,:]
    n1[:,:hB,:] = n1[:,-B:-hB,:]

    nn1 = scipy.ndimage.filters.gaussian_filter(n1, 5 * H / 40)
    return nn1

NX = int(sys.argv[1])
NY = int(sys.argv[2])
NZ = int(sys.argv[3])
H = int(sys.argv[4])

tg = sys.argv[5]
np.random.seed(11341351351)

_, dy1, dz1 = [
        x[hB:-hB,hB:-hB,:] for x in np.gradient(make_rand(NX, NY, NZ, H))]
dx2, _, dz2 = [
        x[hB:-hB,hB:-hB,:] for x in np.gradient(make_rand(NX, NY, NZ, H))]
dx3, dy3, _ = [
        x[hB:-hB,hB:-hB,:] for x in np.gradient(make_rand(NX, NY, NZ, H))]

np.savez_compressed('%s/rng_%d_%d_%d_dvx.npz' % (tg, NX, NY, NZ), data=(dy3 - dz2))
np.savez_compressed('%s/rng_%d_%d_%d_dvy.npz' % (tg, NX, NY, NZ), data=(dz1 - dx3))
np.savez_compressed('%s/rng_%d_%d_%d_dvz.npz' % (tg, NX, NY, NZ), data=(dx2 - dy1))

"""
Utility functions for computing wall shear stress from simulation data."
"""
from scipy import ndimage
import numpy as np


def NearWallMask(walls, distance):
    r = distance
    hx, hy, hz = np.mgrid[-r:(r+1), -r:(r+1), -r:(r+1)]
    radius = np.sqrt(hx**2 + hy**2 + hz**2)
    radius[radius > r] = 0.0

    near_wall = ndimage.filters.convolve(walls.astype(np.uint8), radius, mode='constant') > 0
    mask = near_wall & (~walls)
    return mask


def ComputeDynamicNormals(fields, stress, radius=3):
    """
    Computes normal vectors for the velocity and stress fields.

    This method is based on the observation that n_alpha * u_alpha = 0 at
    the wall and that with the boundary being an isosurface of $\vec{u} = 0$,
    the gradient of velocity is orthogonal to that isosurface.

    This method does not necessarily correctly identify the direction of the
    normal vector.

    :param fields: dict of the density ('rho') and velocity ('v'). Wall nodes
        are expected to be marked 'nan' in these arrays.
    :param stress: dict of components of the stress tensor
    :param radius: int, maximum distance from the wall that a node can have in
        order for the normal vector to be computed for it

    :returns: 4D array of normal vectors; the last dimension is the component
        index
    :rtype ndarray:

    This method was first described in:
      Stahl, B., Chopard, B., and Latt, J. Measurements of wall shear stress with
      the lattice Boltzmann method and staircase approximation of boundaries.
      Computers & Fluids, 39(9):1625-1633, 2010
    """
    walls = np.isnan(fields['rho'])
    fluid = ~walls

    mask = NearWallMask(walls, radius)
    m = mask.flatten()
    s_mtx = np.zeros((np.sum(m), 3, 3), dtype=np.float64)
    s_mtx[:,0,0] = stress['xx'].flat[m]
    s_mtx[:,1,1] = stress['yy'].flat[m]
    s_mtx[:,2,2] = stress['zz'].flat[m]
    s_mtx[:,0,1] = stress['xy'].flat[m]
    s_mtx[:,1,0] = stress['xy'].flat[m]
    s_mtx[:,0,2] = stress['xz'].flat[m]
    s_mtx[:,2,0] = stress['xz'].flat[m]
    s_mtx[:,1,2] = stress['yz'].flat[m]
    s_mtx[:,2,1] = stress['yz'].flat[m]

    # Compute eigenvalues and eigenvectors and find the eigenvector corresponding
    # to the eigenvalue of the lowest magnitude.
    ei = np.linalg.eigh(s_mtx)

    e2_idx = np.argmin(np.abs(ei[0]), axis=1)
    # Repeat once per every vector component.
    e2_idx = np.repeat(e2_idx, 3).reshape((e2_idx.size, 3))
    e2 = np.choose(e2_idx, (ei[1][:,:,0], ei[1][:,:,1], ei[1][:,:,2]))

    # Compute normalized velocity.
    v = fields['v'][:,mask]
    vn = v / np.linalg.norm(v, axis=0)
    normals = np.cross(vn, e2, axisa=0)

    # Restore original shape.
    ret = np.zeros(list(fields['rho'].shape) + [3])
    ret[mask,:] = normals
    return ret


def ComputeLatticeNormals(geometry, radius=2, exp=1.0):
    """
    Computes normal vectors for near-wall nodes using wall facet normals.

    :param geometry: 3D ndarray of bool; True indicates wall nodes.
    :param radius: int, radius within which normals will be averaged.

    :returns: 4D array of normal vectors; the last dimension is the component index.
    :rtype: ndarray

    This method was first described in:
      Matyka, M., Koza, Z., and Miroslaw, L. Wall orientation and shear stress in the
      lattice Boltzmann model. Computers & Fluids, 73:115-123, 2013.
    """
    geo = geometry

    # Compute components of the facet normals.
    normal_xp = np.pad(geo[:,:,:-1] & ~geo[:,:,1:], ((0, 0), (0, 0), (0, 1)), mode='constant').astype(np.uint8).astype(np.float32)
    normal_xn = np.pad(geo[:,:,1:] & ~geo[:,:,:-1], ((0, 0), (0, 0), (1, 0)), mode='constant').astype(np.uint8).astype(np.float32)
    normal_yp = np.pad(geo[:,:-1,:] & ~geo[:,1:,:], ((0, 0), (0, 1), (0, 0)), mode='constant').astype(np.uint8).astype(np.float32)
    normal_yn = np.pad(geo[:,1:,:] & ~geo[:,:-1,:], ((0, 0), (1, 0), (0, 0)), mode='constant').astype(np.uint8).astype(np.float32)
    normal_zp = np.pad(geo[:-1,:,:] & ~geo[1:,:,:], ((0, 1), (0, 0), (0, 0)), mode='constant').astype(np.uint8).astype(np.float32)
    normal_zn = np.pad(geo[1:,:,:] & ~geo[:-1,:,:], ((1, 0), (0, 0), (0, 0)), mode='constant').astype(np.uint8).astype(np.float32)

    r = radius
    hx, hy, hz = np.mgrid[-r:(r+1), -r:(r+1), -r:(r+1)]
    radius = np.sqrt(hx**2 + hy**2 + hz**2)

    # Build a normalized weight kernel.
    weight = (1.0 / (1 + radius))**exp
    weight /= weight.sum()

    xpc = ndimage.filters.convolve(normal_xp, weight, mode='nearest')
    xnc = ndimage.filters.convolve(normal_xn, weight, mode='nearest')

    ypc = ndimage.filters.convolve(normal_yp, weight, mode='nearest')
    ync = ndimage.filters.convolve(normal_yn, weight, mode='nearest')

    zpc = ndimage.filters.convolve(normal_zp, weight, mode='nearest')
    znc = ndimage.filters.convolve(normal_zn, weight, mode='nearest')

    # Compute normal vector components.
    nz = znc - zpc
    nx = xnc - xpc
    ny = ync - ypc

    nlen = np.sqrt(nz**2 + nx**2 + ny**2)
    mask = (nlen != 0)
    nx[mask] /= nlen[mask]
    ny[mask] /= nlen[mask]
    nz[mask] /= nlen[mask]

    return np.concatenate((nx[:, :, :, np.newaxis],
                           ny[:, :, :, np.newaxis],
                           nz[:, :, :, np.newaxis]),
                           axis=3)


def ComputeWSS(normals, stress, visc):
    nx = normals[:,:,:,0]
    ny = normals[:,:,:,1]
    nz = normals[:,:,:,2]

    prefactor = -6. * visc / (1. + 6. * visc)
    full_contr = stress['xx'] * nx * nx + stress['yy'] * ny * ny + stress['zz'] * nz * nz + 2 * (
            stress['xy'] * nx * ny + stress['xz'] * nx * nz + stress['yz'] * ny * nz)

    wss_x = prefactor * (stress['xx'] * nx + stress['xy'] * ny + stress['xz'] * nz - nx * full_contr)
    wss_y = prefactor * (stress['xy'] * nx + stress['yy'] * ny + stress['yz'] * nz - ny * full_contr)
    wss_z = prefactor * (stress['xz'] * nx + stress['yz'] * ny + stress['zz'] * nz - nz * full_contr)
    wss_n = np.sqrt(wss_z**2 + wss_y**2 + wss_x**2)

    return wss_n, (wss_x, wss_y, wss_z)


def ComputeOSI(normals, stresses, visc):
    wss_sum_x = np.zeros(shape=normals[:,:,:,0].shape, dtype=np.float64)
    wss_sum_y = np.zeros(shape=normals[:,:,:,0].shape, dtype=np.float64)
    wss_sum_z = np.zeros(shape=normals[:,:,:,0].shape, dtype=np.float64)
    wss_sum_n = np.zeros(shape=normals[:,:,:,0].shape, dtype=np.float64)

    num_steps = 0
    for stress in stresses:
        wss_n, wss = ComputeWSS(normals, stress, visc)

        wss_sum_n += wss_n
        wss_sum_x += wss[0]
        wss_sum_y += wss[1]
        wss_sum_z += wss[2]
        num_steps += 1

    wss1 = np.sqrt(wss_sum_x**2 + wss_sum_y**2 + wss_sum_z**2) / num_steps
    wss2 = wss_sum_n / num_steps

    return 0.5 * (1 - wss1 / wss2)


def ComputeAngles(v1, v2):
    """Computes point-wise angles between two vector fields.

    :param v1: first vector field (4D ndarray)
    :param v2: second vector field (4D ndarray)

    :rtype: 3D ndarray
    :returns: array of angles (in degrees) between the two vectors
    """
    s = v1.shape
    v1 = v1.reshape((v1.size // 3, 3))
    v2 = v2.reshape((v2.size // 3, 3))
    v1v2sin = np.linalg.norm(np.cross(v1, v2), axis=1),
    v1v2cos = np.sum(np.prod(np.dstack((v1, v2)), axis=2),
                             axis=1)
    return np.arctan2(v1v2sin, v1v2cos).reshape(s[:3]) * 180. / np.pi

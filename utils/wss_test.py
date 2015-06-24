import unittest
import wss
import numpy as np

class NormalVectorsTest(unittest.TestCase):
    def _geo_and_normals(self):
        # Geometry of a pipe oriented in the 'z' direction.
        geo = np.zeros((8, 61, 61), dtype=np.bool)
        hz, hy, hx = np.mgrid[:8, -30:31, -30:31]
        geo[np.sqrt(hx**2 + hy**2) > 28] = True

        expected = np.zeros((8, 61, 61, 3))
        expected[:,:,:,0] = hx
        expected[:,:,:,1] = hy
        expected[:,:,:,2] = 0
        expected /= np.repeat(np.linalg.norm(expected, axis=3), 3).reshape(expected.shape)

        return geo, expected

    def test_geometric(self):
        geo, expected = self._geo_and_normals()
        normals = wss.ComputeLatticeNormals(geo, radius=5, exp=0.5)
        nw = wss.NearWallMask(geo, 3)
        self.assertTrue(np.all(
            wss.ComputeAngles(normals, expected)[nw] < 5))

    def test_dynamic(self):
        geo, expected = self._geo_and_normals()

        rho = np.ones(shape=geo.shape)
        rho[geo] = np.nan

        # Poiseuille profile.
        vel = np.zeros(shape=[3] + list(geo.shape))
        hz, hy, hx = np.mgrid[:8, -30:31, -30:31]
        r = np.sqrt(hx**2 + hy**2) / 28.5
        vel[2] = 1 - r**2

        # uz = 1 - x**2 - y**2
        # d_x u_z = -2 * x
        # d_y u_z = -2 * y

        # S_ij = 0.5 (\partial_i u_j + \partial_j u_i)
        xx = np.zeros(shape=geo.shape)
        yy = np.zeros(shape=geo.shape)
        zz = np.zeros(shape=geo.shape)
        xy = np.zeros(shape=geo.shape)
        xz = -1.0 * hx
        yz = -1.0 * hy

        normals = wss.ComputeDynamicNormals({'rho': rho, 'v': vel},
                                            {'xx': xx, 'yy': yy, 'zz': zz,
                                             'xy': xy, 'xz': xz, 'yz': yz}, radius=5)
        nw = wss.NearWallMask(geo, 3)

        # This method does not correctly recognize direction.
        self.assertTrue(np.all(
            (wss.ComputeAngles(normals, expected)[nw] < 0.1) |
            (wss.ComputeAngles(normals, expected)[nw] > 180 - 0.1)))

    def test_angles(self):
        v1 = np.array(
            [[1, 1, 0],
             [0, 1, 1]]).reshape((1, 2, 1, 3))
        v2 = np.array(
            [[0, 0, 1],
             [0, 0, 1]]).reshape((1, 2, 1, 3))
        np.testing.assert_equal(
            wss.ComputeAngles(v1, v2),
            np.array([90.0, 45.0]).reshape((1, 2, 1)))


class WSSTest(unittest.TestCase):
    pass

if __name__ == '__main__':
    unittest.main()

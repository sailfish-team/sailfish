import numpy as np
import unittest
from sailfish.config import LBConfig
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D, LBBlock3D
from sailfish.sym import D2Q9

vi = lambda x, y: D2Q9.vec_idx([x, y])

class TestBlock3D(unittest.TestCase):
    def _connection_helper(self, c0, type_, conn_loc, axis):
        """
        :param c0: lower coordinate along the connection axis
        """
        def f(a, b, c):
            if axis == 0:
                return [a, b, c]
            elif axis == 1:
                return [b, a, c]
            else:
                return [c, b, a]

        base = LBBlock3D((10, 10, 10), (10, 10, 10), id_=0)
        b1 = LBBlock3D(f(c0, 10, 10), f(5, 10, 10), id_=1)     # exact match
        b2 = LBBlock3D(f(c0, 5, 5), f(5, 10, 10), id_=2)       # quarter overlap
        b3 = LBBlock3D(f(c0, 5, 5), f(5, 20, 20), id_=3)       # total overlap (2nd block is larger)
        b4 = LBBlock3D(f(c0, 10, 10), f(5, 5, 5), id_=4)       # quarter overlap
        b5 = LBBlock3D(f(c0, 15, 14), f(5, 3, 2), id_=5)       # total overlap (2nd block is smaller)
        b6 = LBBlock3D(f(c0, 15, 14), f(5, 10, 10), id_=6)     # quarter overlap

        self.assertTrue(base.connect(b1))
        self.assertTrue(base.connect(b2))
        self.assertTrue(base.connect(b3))
        self.assertTrue(base.connect(b4))
        self.assertTrue(base.connect(b5))
        self.assertTrue(base.connect(b6))

        gcs = base.get_connection_selector
        self.assertEqual(gcs(type_, b1.id), f(conn_loc, slice(0, 10), slice(0, 10)))
        self.assertEqual(gcs(type_, b2.id), f(conn_loc, slice(0, 5), slice(0, 5)))
        self.assertEqual(gcs(type_, b3.id), f(conn_loc, slice(0, 10), slice(0, 10)))
        self.assertEqual(gcs(type_, b4.id), f(conn_loc, slice(0, 5), slice(0, 5)))
        self.assertEqual(gcs(type_, b5.id), f(conn_loc, slice(5, 8), slice(4, 6)))
        self.assertEqual(gcs(type_, b6.id), f(conn_loc, slice(5, 10), slice(4, 10)))

        bf1 = LBBlock3D(f(c0, 21, 5), f(5, 10, 20))     # too far along Y
        bf2 = LBBlock3D(f(c0, 5, 21), f(5, 20, 10))     # too far along Z
        bf3 = LBBlock3D(f(c0, 5, 10), f(5, 4, 10))      # too short along Y
        bf4 = LBBlock3D(f(c0, 10, 5), f(5, 10, 4))      # too short along Z
        bf5 = LBBlock3D(f(c0-1, 10, 10), f(5, 10, 10))  # wrong X
        bf6 = LBBlock3D(f(c0+1, 10, 10), f(5, 10, 10))  # wrong X

        self.assertFalse(base.connect(bf1))
        self.assertFalse(base.connect(bf2))
        self.assertFalse(base.connect(bf3))
        self.assertFalse(base.connect(bf4))
        self.assertFalse(base.connect(bf5))
        self.assertFalse(base.connect(bf6))

        # Edges
        b7 = LBBlock3D(f(c0, 5, 10), f(5, 5, 10), id_=7)    # lower Y edge
        b8 = LBBlock3D(f(c0, 10, 5), f(5, 10, 5), id_=8)    # lower Z edge
        b9 = LBBlock3D(f(c0, 20, 10), f(5, 5, 10), id_=9)   # higher Y edge
        b10 = LBBlock3D(f(c0, 10, 20), f(5, 10, 5), id_=10) # higher Z edge

        self.assertTrue(base.connect(b7))
        self.assertTrue(base.connect(b8))
        self.assertTrue(base.connect(b9))
        self.assertTrue(base.connect(b10))

        if axis == 0:
            self.assertEqual(gcs(type_, b7.id), f(conn_loc, slice(0, 0), slice(0, 10)))
            self.assertEqual(gcs(type_, b8.id), f(conn_loc, slice(0, 10), slice(0, 0)))
            self.assertEqual(gcs(type_, b9.id), f(conn_loc, slice(10, 10), slice(0, 10)))
            self.assertEqual(gcs(type_, b10.id), f(conn_loc, slice(0, 10), slice(10, 10)))
        elif axis == 1:
            self.assertEqual(gcs(type_, b8.id), [slice(0, 10), conn_loc, slice(0, 0)])
            self.assertEqual(gcs(type_, b10.id), [slice(0, 10), conn_loc, slice(10, 10)])
        elif type_ == LBBlock3D._Z_HIGH:
            self.assertEqual(gcs(LBBlock3D._Y_LOW, b7.id), [slice(0, 10), 0, slice(10, 10)])
            self.assertEqual(gcs(LBBlock3D._Y_HIGH, b9.id), [slice(0, 10), 9, slice(10, 10)])
        elif type_ == LBBlock3D._Z_LOW:
            self.assertEqual(gcs(LBBlock3D._Y_LOW, b7.id), [slice(0, 10), 0, slice(0, 0)])
            self.assertEqual(gcs(LBBlock3D._Y_HIGH, b9.id), [slice(0, 10), 9, slice(0, 0)])

        # Corners
        b11 = LBBlock3D(f(c0, 5, 5), f(5, 5, 5), id_=11)      # low Y, low Z
        b12 = LBBlock3D(f(c0, 5, 20), f(5, 5, 5), id_=12)     # low Y, high Z
        b13 = LBBlock3D(f(c0, 20, 5), f(5, 5, 5), id_=13)     # high Y, low Z
        b14 = LBBlock3D(f(c0, 20, 20), f(5, 5, 5), id_=14)    # high Y, high Z

        self.assertTrue(base.connect(b11))
        self.assertTrue(base.connect(b12))
        self.assertTrue(base.connect(b13))
        self.assertTrue(base.connect(b14))

        # Corners are always X-axis connections
        if axis == 0:
            # Corner connections are signified by empty slices.
            self.assertEqual(gcs(type_, b11.id), f(conn_loc, slice(0, 0), slice(0, 0)))
            self.assertEqual(gcs(type_, b12.id), f(conn_loc, slice(0, 0), slice(10, 10)))
            self.assertEqual(gcs(type_, b13.id), f(conn_loc, slice(10, 10), slice(0, 0)))
            self.assertEqual(gcs(type_, b14.id), f(conn_loc, slice(10, 10), slice(10, 10)))



class TestBlock2D(unittest.TestCase):

    def _verify_partial_map(self, conn, expected_map):
        self.assertEqual(conn.dst_partial_map.keys(),
                expected_map.keys())
        for key, val in expected_map.iteritems():
            self.assertTrue(np.all(val == conn.dst_partial_map[key]))

    def _test_block_conn(self, axis):
        # All coordinate tuples below are specified for the case
        # in which the two blocks are connected along the X axis.
        # Y-axis connection requires a swap.
        def f(a, b):
            if axis == 0:
                return (a,b)
            else:
                return (b,a)

        face_hi = LBBlock2D.axis_dir_to_face(axis, 1)
        face_lo = LBBlock2D.axis_dir_to_face(axis, -1)
        base = LBBlock2D(f(10, 10), f(10, 10), envelope_size=1, id_=0)

        # exact match
        b1 = LBBlock2D(f(20, 10), f(5, 10), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D2Q9))
        cpair = base.get_connection(face_hi, b1.id)
        self.assertEqual(set(cpair.src.dists),
                         set([vi(*f(1,0)), vi(*f(1,1)), vi(*f(1,-1))]))
        self.assertEqual(cpair.src.src_slice, [slice(1, 11)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 9)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 9)])
        expected_map = {vi(*f(1,-1)): np.array([[0]]),
                        vi(*f(1,1)): np.array([[9]]),
                        vi(*f(1,0)): np.array([[0],[9]])}
        self._verify_partial_map(cpair.src, expected_map)

        # partal overlap
        b2 = LBBlock2D(f(20, 5), f(5, 10), envelope_size=1, id_=2)
        self.assertTrue(base.connect(b2, grid=D2Q9))
        cpair = base.get_connection(face_hi, b2.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 6)])
        self.assertEqual(cpair.src.dst_low, [4])
        self.assertEqual(cpair.src.dst_slice, [slice(6, 10)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(2, 6)])
        expected_map = {vi(*f(1,-1)): np.array([[0], [1]]),
                        vi(*f(1,0)): np.array([[1]])}
        self._verify_partial_map(cpair.src, expected_map)

        # full overlap (2nd block is smaller)
        b3 = LBBlock2D(f(20, 12), f(5, 7), envelope_size=1, id_=3)
        self.assertTrue(base.connect(b3, grid=D2Q9))
        cpair = base.get_connection(face_hi, b3.id)
        self.assertEqual(cpair.src.src_slice, [slice(3, 10)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [slice(0, 7)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(0, 7)])
        self.assertFalse(cpair.src.dst_partial_map)

        # full overlap (2nd block is larger)
        b4 = LBBlock2D(f(20, 8), f(5, 14), envelope_size=1, id_=4)
        self.assertTrue(base.connect(b4, grid=D2Q9))
        cpair = base.get_connection(face_hi, b4.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 12)])
        self.assertEqual(cpair.src.dst_low, [1])
        self.assertEqual(cpair.src.dst_slice, [slice(3, 11)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(2, 10)])
        expected_map = {
                vi(*f(1,-1)): np.array([[0], [1]]),
                vi(*f(1,0)): np.array([[1], [10]]),
                vi(*f(1,1)): np.array([[10], [11]])}
        self._verify_partial_map(cpair.src, expected_map)

        # exact match at the bottom
        b5 = LBBlock2D(f(20, 10), f(5, 5), envelope_size=1, id_=5)
        self.assertTrue(base.connect(b5, grid=D2Q9))
        cpair = base.get_connection(face_hi, b5.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 6)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 5)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 5)])
        expected_map = {
                vi(*f(1,-1)): np.array([[0]]),
                vi(*f(1,0)): np.array([[0]])}
        self._verify_partial_map(cpair.src, expected_map)

        # disconnected blocks
        bf1 = LBBlock2D(f(20, 21), f(5, 10), envelope_size=1)
        bf2 = LBBlock2D(f(20, 5),  f(5, 4), envelope_size=1)
        bf3 = LBBlock2D(f(19, 10), f(5, 10), envelope_size=1)
        bf4 = LBBlock2D(f(21, 10), f(5, 10), envelope_size=1)
        self.assertFalse(base.connect(bf1))
        self.assertFalse(base.connect(bf2))
        self.assertFalse(base.connect(bf3))
        self.assertFalse(base.connect(bf4))

    def test_block_connection_x(self):
        self._test_block_conn(0)

    def test_block_connection_y(self):
        self._test_block_conn(1)

    def test_corner_connection(self):
        base = LBBlock2D((10, 10), (10, 10), envelope_size=1, id_=0)
        # corner match (low)
        b6 = LBBlock2D((20, 5), (5, 5), envelope_size=1, id_=6)
        self.assertTrue(base.connect(b6, grid=D2Q9))
        cpair = base.get_connection(LBBlock2D._X_HIGH, b6.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 1)])
        self.assertEqual(cpair.src.dst_low, [4])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi(1,-1): np.array([0])}
        self._verify_partial_map(cpair.src, expected_map)

        self.assertEqual(cpair.dst.src_slice, [slice(6, 7)])
        self.assertEqual(cpair.dst.dst_low, [0])
        self.assertEqual(cpair.dst.dst_slice, [])
        self.assertEqual(cpair.dst.dst_full_buf_slice, [])
        expected_map = {
                vi(-1,1): np.array([0])}
        self._verify_partial_map(cpair.dst, expected_map)

        # corner match (high)
        b7 = LBBlock2D((20, 20), (5, 5), envelope_size=1, id_=7)
        self.assertTrue(base.connect(b7, grid=D2Q9))
        cpair = base.get_connection(LBBlock2D._X_HIGH, b7.id)
        self.assertEqual(cpair.src.src_slice, [slice(11, 12)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi(1,1): np.array([0])}
        self._verify_partial_map(cpair.src, expected_map)

        self.assertEqual(cpair.dst.src_slice, [slice(0, 1)])
        self.assertEqual(cpair.dst.dst_low, [9])
        self.assertEqual(cpair.dst.dst_slice, [])
        self.assertEqual(cpair.dst.dst_full_buf_slice, [])
        expected_map = {
                vi(-1,-1): np.array([0])}
        self._verify_partial_map(cpair.dst, expected_map)

    def test_global_block_connection_xy(self):
        config = LBConfig()
        config.lat_nx = 64
        config.lat_ny = 64

        def _verify_slices(cpair):
            self.assertEqual(cpair.src.src_slice, [slice(1,33)])
            self.assertEqual(cpair.src.dst_low, [0])
            self.assertEqual(cpair.src.dst_slice, [slice(1,31)])
            self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1,31)])

        geo = LBGeometry2D(config)
        b1 = LBBlock2D((0, 0), (32, 32), envelope_size=1, id_=1)
        b2 = LBBlock2D((32, 0), (32, 32), envelope_size=1, id_=2)
        self.assertTrue(b1.connect(b2, geo, axis=0, grid=D2Q9))

        cpair = b1.get_connection(LBBlock2D._X_LOW, b2.id)
        self.assertEqual(set(cpair.src.dists),
                         set([vi(-1,0), vi(-1,1), vi(-1,-1)]))
        _verify_slices(cpair)
        expected_map = {
                vi(-1,-1): np.array([[0]]),
                vi(-1, 0): np.array([[0],[31]]),
                vi(-1, 1): np.array([[31]])}
        self._verify_partial_map(cpair.src, expected_map)

        cpair = b2.get_connection(LBBlock2D._X_HIGH, b1.id)
        _verify_slices(cpair)
        expected_map = {
                vi(1,-1): np.array([[0]]),
                vi(1, 0): np.array([[0],[31]]),
                vi(1, 1): np.array([[31]])}
        self._verify_partial_map(cpair.src, expected_map)

        b3 = LBBlock2D((0, 32), (32, 32), envelope_size=1, id_=3)
        self.assertTrue(b3.connect(b1, geo, axis=1, grid=D2Q9))
        cpair = b1.get_connection(LBBlock2D._Y_LOW, b3.id)
        _verify_slices(cpair)
        expected_map = {
                vi(-1,-1): np.array([[0]]),
                vi(0, -1): np.array([[0], [31]]),
                vi(1, -1): np.array([[31]])}
        self._verify_partial_map(cpair.src, expected_map)

        cpair = b3.get_connection(LBBlock2D._Y_HIGH, b1.id)
        _verify_slices(cpair)
        expected_map = {
                vi(-1,1): np.array([[0]]),
                vi(0, 1): np.array([[0], [31]]),
                vi(1, 1): np.array([[31]])}
        self._verify_partial_map(cpair.src, expected_map)


if __name__ == '__main__':
    unittest.main()

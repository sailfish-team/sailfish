import unittest
from sailfish.config import LBConfig
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D, LBBlock3D


class TestBlock3D(unittest.TestCase):
    def _connection_helper(self, c0, type_, conn_loc, axis):
        """
        :param c0: lower coordinate along the connection axis
        """
        def f(a, b, c):
            if axis == 0:
                return a, b, c
            elif axis == 1:
                return b, a, c
            else:
                return c, b, a

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
        self.assertEqual(gcs(type_, b1.id), list(f(conn_loc, slice(0, 10), slice(0, 10))))
        self.assertEqual(gcs(type_, b2.id), list(f(conn_loc, slice(0, 5), slice(0, 5))))
        self.assertEqual(gcs(type_, b3.id), list(f(conn_loc, slice(0, 10), slice(0, 10))))
        self.assertEqual(gcs(type_, b4.id), list(f(conn_loc, slice(0, 5), slice(0, 5))))
        self.assertEqual(gcs(type_, b5.id), list(f(conn_loc, slice(5, 8), slice(4, 6))))
        self.assertEqual(gcs(type_, b6.id), list(f(conn_loc, slice(5, 10), slice(4, 10))))

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
            self.assertEqual(gcs(type_, b7.id), list(f(conn_loc, slice(0, 0), slice(0, 10))))
            self.assertEqual(gcs(type_, b8.id), list(f(conn_loc, slice(0, 10), slice(0, 0))))
            self.assertEqual(gcs(type_, b9.id), list(f(conn_loc, slice(10, 10), slice(0, 10))))
            self.assertEqual(gcs(type_, b10.id), list(f(conn_loc, slice(0, 10), slice(10, 10))))
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
            self.assertEqual(gcs(type_, b11.id), list(f(conn_loc, slice(0, 0), slice(0, 0))))
            self.assertEqual(gcs(type_, b12.id), list(f(conn_loc, slice(0, 0), slice(10, 10))))
            self.assertEqual(gcs(type_, b13.id), list(f(conn_loc, slice(10, 10), slice(0, 0))))
            self.assertEqual(gcs(type_, b14.id), list(f(conn_loc, slice(10, 10), slice(10, 10))))



    def test_block_connection_x_high(self):
        self._connection_helper(20, LBBlock3D._X_HIGH, 9, 0)

    def test_block_connection_x_low(self):
        self._connection_helper(5, LBBlock3D._X_LOW, 0, 0)

    def test_block_connection_y_high(self):
        self._connection_helper(20, LBBlock3D._Y_HIGH, 9, 1)

    def test_block_connection_y_low(self):
        self._connection_helper(5, LBBlock3D._Y_LOW, 0, 1)

    def test_block_connection_z_high(self):
        self._connection_helper(20, LBBlock3D._Z_HIGH, 9, 2)

    def test_block_connection_z_low(self):
        self._connection_helper(5, LBBlock3D._Z_LOW, 0, 2)

class TestBlock2D(unittest.TestCase):

    def _connection_helper(self, c0, type_, conn_loc, x_axis):
        """
        :param c0: lower coordinate along the connection axis
        """
        def f(a, b):
            if x_axis:
                return a, b
            else:
                return b, a

        base = LBBlock2D(f(10, 10), f(10, 10), id_=0)

        b1 = LBBlock2D(f(c0, 10), f(5, 10), id_=1)
        b2 = LBBlock2D(f(c0, 5), f(5, 10), id_=2)
        b3 = LBBlock2D(f(c0, 5), f(5, 20), id_=3)
        b4 = LBBlock2D(f(c0, 10), f(5, 5), id_=4)
        b5 = LBBlock2D(f(c0, 15), f(5, 3), id_=5)
        b6 = LBBlock2D(f(c0, 15), f(5, 10), id_=6)

        #   rr
        # bb
        b7 = LBBlock2D(f(c0, 5), f(5, 5), id_=7)
        # bb
        #   rr
        b8 = LBBlock2D(f(c0, 20), f(5, 5), id_=8)

        self.assertTrue(base.connect(b1))
        self.assertTrue(base.connect(b2))
        self.assertTrue(base.connect(b3))
        self.assertTrue(base.connect(b4))
        self.assertTrue(base.connect(b5))
        self.assertTrue(base.connect(b6))
        self.assertTrue(base.connect(b7))
        self.assertTrue(base.connect(b8))

        gcs = base.get_connection_selector
        self.assertEqual(gcs(type_, b1.id), f(conn_loc, slice(0, 10)))
        self.assertEqual(gcs(type_, b2.id), f(conn_loc, slice(0, 5)))
        self.assertEqual(gcs(type_, b3.id), f(conn_loc, slice(0, 10)))
        self.assertEqual(gcs(type_, b4.id), f(conn_loc, slice(0, 5)))
        self.assertEqual(gcs(type_, b5.id), f(conn_loc, slice(5, 8)))
        self.assertEqual(gcs(type_, b6.id), f(conn_loc, slice(5, 10)))

        # Corner nodes are always marked as X-axis connections.
        if x_axis:
            # Special case: corner connection is signified by an empty slice.
            self.assertEqual(gcs(type_, b7.id), f(conn_loc, slice(0, 0)))
            self.assertEqual(gcs(type_, b8.id), f(conn_loc, slice(10, 10)))

        bf1 = LBBlock2D(f(c0, 21), f(5, 10))
        bf2 = LBBlock2D(f(c0, 5), f(5, 4))
        bf3 = LBBlock2D(f(c0-1, 10), f(5, 10))
        bf4 = LBBlock2D(f(c0+1, 10), f(5, 10))

        self.assertFalse(base.connect(bf1))
        self.assertFalse(base.connect(bf2))
        self.assertFalse(base.connect(bf3))
        self.assertFalse(base.connect(bf4))

    def test_block_connection_x_high(self):
        self._connection_helper(20, LBBlock2D._X_HIGH, 9, True)

    def test_block_connection_x_low(self):
        self._connection_helper(5, LBBlock2D._X_LOW, 0, True)

    def test_block_connection_y_high(self):
        self._connection_helper(20, LBBlock2D._Y_HIGH, 9, False)

    def test_block_connection_y_low(self):
        self._connection_helper(5, LBBlock2D._Y_LOW, 0, False)

    def test_global_block_connection_xy(self):
        config = LBConfig()
        config.lat_nx = 64
        config.lat_ny = 64

        geo = LBGeometry2D(config)
        b1 = LBBlock2D((0, 0), (32, 32))
        b1.id = 1
        b2 = LBBlock2D((32, 0), (32, 32))
        b2.id = 2
        self.assertTrue(b1.connect(b2, geo, axis=0))

        span = b1.get_connection_selector(LBBlock2D._X_LOW, b2.id)
        self.assertEqual(span, (0, slice(0, 32)))
        span = b2.get_connection_selector(LBBlock2D._X_HIGH, b1.id)
        self.assertEqual(span, (31, slice(0, 32)))

        b3 = LBBlock2D((0, 32), (32, 32))
        b3.id = 3
        self.assertTrue(b3.connect(b1, geo, axis=1))
        span = b1.get_connection_selector(LBBlock2D._Y_LOW, b3.id)
        self.assertEqual(span, (slice(0, 32), 0))
        span = b3.get_connection_selector(LBBlock2D._Y_HIGH, b1.id)
        self.assertEqual(span, (slice(0, 32), 31))

        # TODO(michalj): Consider more complex tests here like in
        # test_block_connection_*

    def test_2d_corner_block_connection(self):
        b1 = LBBlock2D((0, 0), (10, 10), id_=0)
        b2 = LBBlock2D((10, 0), (10, 10), id_=1)
        b3 = LBBlock2D((0, 10), (10, 10), id_=2)
        b4 = LBBlock2D((10, 10), (10, 10), id_=3)

        self.assertTrue(b1.connect(b2))
        self.assertTrue(b1.connect(b3))
        self.assertTrue(b1.connect(b4))

        gcs = b1.get_connection_selector
        self.assertEqual(gcs(LBBlock2D._X_HIGH, b2.id), (9, slice(0, 10, None)))
        self.assertEqual(gcs(LBBlock2D._X_HIGH, b4.id), (9, slice(10, 10, None)))
        self.assertEqual(b1.connecting_blocks(),
                [(LBBlock2D._X_HIGH, b2.id), (LBBlock2D._X_HIGH, b4.id),
                 (LBBlock2D._Y_HIGH, b3.id)])


if __name__ == '__main__':
    unittest.main()

import unittest
from sailfish.config import LBConfig
from sailfish.geo import LBGeometry2D
from sailfish.geo_block import LBBlock2D

class TestBlock2D(unittest.TestCase):

    def _connection_helper(self, c0, type_, span, x_axis):
        def f(a, b):
            if x_axis:
                return a, b
            else:
                return b, a

        base = LBBlock2D(f(10, 10), f(10, 10))
        base.id = 0

        b1 = LBBlock2D(f(c0, 10), f(5, 10))
        b2 = LBBlock2D(f(c0, 5), f(5, 10))
        b3 = LBBlock2D(f(c0, 5), f(5, 20))
        b4 = LBBlock2D(f(c0, 10), f(5, 5))
        b5 = LBBlock2D(f(c0, 15), f(5, 3))
        b6 = LBBlock2D(f(c0, 15), f(5, 10))

        #   rr
        # bb
        b7 = LBBlock2D(f(c0, 5), f(5, 5))
        # bb
        #   rr
        b8 = LBBlock2D(f(c0, 20), f(5, 5))
        b1.id = 1
        b2.id = 2
        b3.id = 3
        b4.id = 4
        b5.id = 5
        b6.id = 6
        b7.id = 7
        b8.id = 8

        self.assertTrue(base.connect(b1))
        self.assertTrue(base.connect(b2))
        self.assertTrue(base.connect(b3))
        self.assertTrue(base.connect(b4))
        self.assertTrue(base.connect(b5))
        self.assertTrue(base.connect(b6))
        self.assertTrue(base.connect(b7))
        self.assertTrue(base.connect(b8))

        gcs = base.get_connection_span
        self.assertEqual(gcs(type_, b1.id), f(span, slice(0, 10)))
        self.assertEqual(gcs(type_, b2.id), f(span, slice(0, 5)))
        self.assertEqual(gcs(type_, b3.id), f(span, slice(0, 10)))
        self.assertEqual(gcs(type_, b4.id), f(span, slice(0, 5)))
        self.assertEqual(gcs(type_, b5.id), f(span, slice(5, 8)))
        self.assertEqual(gcs(type_, b6.id), f(span, slice(5, 10)))

        # Corner nodes are always marked as X-axis connections.
        if x_axis:
            # Special case: corner connection is signified by an empty slice.
            self.assertEqual(gcs(type_, b7.id), f(span, slice(0, 0)))
            self.assertEqual(gcs(type_, b8.id), f(span, slice(10, 10)))

        bf1 = LBBlock2D(f(c0, 21), f(5, 10))
        bf2 = LBBlock2D(f(c0, 5), f(5, 4))
        bf3 = LBBlock2D(f(c0-1, 10), f(5, 10))
        bf4 = LBBlock2D(f(c0+1, 10), f(5, 10))

        self.assertFalse(base.connect(bf1))
        self.assertFalse(base.connect(bf2))
        self.assertFalse(base.connect(bf3))
        self.assertFalse(base.connect(bf4))

    def test_block_connectio_x_high(self):
        self._connection_helper(20, LBBlock2D._X_HIGH, 9, True)

    def test_block_connectio_x_low(self):
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

        span = b1.get_connection_span(LBBlock2D._X_LOW, b2.id)
        self.assertEqual(span, (0, slice(0, 32)))
        span = b2.get_connection_span(LBBlock2D._X_HIGH, b1.id)
        self.assertEqual(span, (31, slice(0, 32)))

        b3 = LBBlock2D((0, 32), (32, 32))
        b3.id = 3
        self.assertTrue(b3.connect(b1, geo, axis=1))
        span = b1.get_connection_span(LBBlock2D._Y_LOW, b3.id)
        self.assertEqual(span, (slice(0, 32), 0))
        span = b3.get_connection_span(LBBlock2D._Y_HIGH, b1.id)
        self.assertEqual(span, (slice(0, 32), 31))

        # TODO(michalj): Consider more complex tests here like in
        # test_block_connection_*


if __name__ == '__main__':
    unittest.main()

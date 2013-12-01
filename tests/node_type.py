import unittest
import numpy as np
from sailfish import node_type as nt


class LinearlyInterpolatedTimeSeriesTest(unittest.TestCase):
    def test_creation(self):
        a = nt.LinearlyInterpolatedTimeSeries([1,2,3], 2.0)
        b = nt.LinearlyInterpolatedTimeSeries((1,2), 3.0)
        c = nt.LinearlyInterpolatedTimeSeries(np.float32([4,5,6]))

        self.assertEqual(a._data.size, 3)
        self.assertEqual(b._data.size, 2)
        self.assertEqual(c._data.size, 3)
        self.assertEqual(a._step_size, 2.0)
        self.assertEqual(b._step_size, 3.0)
        self.assertEqual(c._step_size, 1.0)

    def test_hash_and_eq(self):
        a = nt.LinearlyInterpolatedTimeSeries([1,2,3], 2.0)
        b = nt.LinearlyInterpolatedTimeSeries([1,2,3], 3.0)
        c = nt.LinearlyInterpolatedTimeSeries(np.float64([1, 2, 3]), 2.0)

        self.assertNotEqual(a, 10)
        self.assertEqual(a, c)
        self.assertNotEqual(a, b)
        self.assertEqual(hash(a), hash(c))
        self.assertNotEqual(hash(a), hash(b))


if __name__ == '__main__':
    unittest.main()

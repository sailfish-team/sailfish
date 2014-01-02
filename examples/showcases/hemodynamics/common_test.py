import unittest
import common
import numpy as np

config = {
    'size': [10, 20, 30],
    'padding': [1, 0, 2, 3, 4, 2],
    'bounding_box': [(-0.5, 0.5), (-0.3, 0.6), (1.2, 1.4)],
    'axes': 'zxy',
}

class CoordinateConversionTest(unittest.TestCase):
    def test_from_lb(self):
        conv = common.CoordinateConverter(config)
        pos = [15, 5, 7]
        exp = [0.0, 0.0, 1.29]

        # All coordinates should match within dx (1/size).
        self.assertTrue(np.all(np.abs(np.array(conv.from_lb(pos)) - np.array(exp)) <
                               1.0 / np.array(config['size'])))

    def test_to_lb(self):
        conv = common.CoordinateConverter(config)
        pos = [0.0, 0.0, 1.29]
        exp = [15, 5, 7]
        self.assertEqual(exp, conv.to_lb(pos))

if __name__ == '__main__':
    unittest.main()

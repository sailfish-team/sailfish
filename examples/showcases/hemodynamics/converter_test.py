import unittest
import converter
import numpy as np

config = {
    'size': [10, 20, 30],
    'padding': [1, 0, 2, 3, 4, 2],
    'bounding_box': [(-0.5, 0.5), (-0.3, 0.6), (1.2, 1.4)],
    'axes': 'zxy',
}

class CoordinateConversionTest(unittest.TestCase):
    def test_from_lb(self):
        conv = converter.CoordinateConverter(config)
        pos = [15, 5, 7]
        exp = [0.0, 0.0, 1.29]

        # All coordinates should match within dx (1/size).
        self.assertTrue(np.all(np.abs(np.array(conv.from_lb(pos)) - np.array(exp)) <
                               1.0 / np.array(config['size'])))

    def test_to_lb(self):
        conv = converter.CoordinateConverter(config)
        pos = [0.0, 0.0, 1.29]
        exp = [15, 6, 7]
        self.assertEqual(exp, conv.to_lb(pos, round_=True))

    def test_ushape(self):
        config = {
            "padding": [1, 1, 1, 1, 1, 1],
            "cuts": [[5, 0], [0, 0], [0, 0]],
            "bounding_box": [[-0.25659, 0.076243], [-0.0762, 0.076239], [-0.0127, 0.0127]],
            "axes": "xyz",
            "size": [40, 231, 497]
        }
        conv = converter.CoordinateConverter(config)
        pos = [-0.254, -0.063461, 0]

        self.assertTrue(abs(conv.from_lb([0,0,0])[0] - pos[0]) < 1e-3)

if __name__ == '__main__':
    unittest.main()

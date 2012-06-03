import os
import random
import unittest
import tempfile
import numpy as np
from sailfish import config, util

class TestPbsUtils(unittest.TestCase):
    def test_gpufile_processing(self):
        fd, path = tempfile.mkstemp()
        f = os.fdopen(fd, 'w')
        f.write("node1.domain-gpu0\n"
                "node1.domain-gpu2\n"
                "node2.domain-gpu0")
        f.close()
        random.seed(1234)
        cluster = util.gpufile_to_clusterspec(path)
        os.unlink(path)
        self.assertEqual(
                [config.MachineSpec('socket=node1.domain:15732',
                    'node1.domain', [0, 2], None),
                 config.MachineSpec('socket=node2.domain:15732',
                     'node2.domain', [0], None)],
                cluster.nodes)


class TestMiscUtils(unittest.TestCase):
    def test_reverse_pairs(self):
        l1 = [1, 2, 3, 4]
        l2 = list(util.reverse_pairs(l1))
        self.assertEqual(l2, [2, 1, 4, 3])

        l3 = list(util.reverse_pairs(l1, 2))
        self.assertEqual(l3, [3, 4, 1, 2])

        ls = [1, 2]
        l4 = list(util.reverse_pairs(ls))
        self.assertEqual(l4, [2, 1])
        l5 = list(util.reverse_pairs(ls, 2))
        self.assertEqual(l5, [1, 2])

    def test_inanyd_fast(self):
        a = np.random.random_integers(0, 20, (128, 128))
        b = np.uint32([1, 4, 6, 10, 19])

        np.testing.assert_array_equal(
                util.in_anyd(a, b), util.in_anyd_fast(a, b))


if __name__ == '__main__':
    unittest.main()

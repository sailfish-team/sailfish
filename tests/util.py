import os
import random
import unittest
import tempfile
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


if __name__ == '__main__':
    unittest.main()

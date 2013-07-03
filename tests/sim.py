import unittest
import numpy as np

from sailfish import lb_base

class SimTest(lb_base.LBSim):
    @classmethod
    def fields(cls):
        return [lb_base.ScalarField('a'), lb_base.ScalarField('b')]

class DummyRunner(object):
    def make_scalar_field(self, dtype=None, name=None, register=True,
            async=False, gpu_array=False):
        if dtype is None:
            dtype = np.float
        buf = np.zeros([64, 64], dtype=dtype)
        return np.ndarray([64, 64], buffer=buf, dtype=dtype), None


class TestLBBase(unittest.TestCase):
    def test_overridden_fields(self):
        runner = DummyRunner()
        sim = SimTest(None)
        sim.init_fields(runner)
        sim.a = np.zeros([64, 64])
        self.assertRaises(AssertionError, sim.verify_fields)


if __name__ == '__main__':
    unittest.main()

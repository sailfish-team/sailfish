import unittest
from sailfish import sym

class TestDistComputations(unittest.TestCase):
    def test_get_interblock_dists(self):
        grid = sym.D2Q9
        gid = sym.get_interblock_dists

        def vecs_to_set(vecs):
            ret = set()
            for vec in vecs:
                ret.add(grid.vec_idx(vec))
            return ret

        self.assertEqual(set(gid(grid, [0,  1])), vecs_to_set([(0, 1), (1, 1), (-1, 1)]))
        self.assertEqual(set(gid(grid, [0, -1])), vecs_to_set([(0, -1), (1, -1), (-1, -1)]))
        self.assertEqual(set(gid(grid, [1,  0])), vecs_to_set([(1, -1), (1, 0), (1, 1)]))
        self.assertEqual(set(gid(grid, [-1, 0])), vecs_to_set([(-1, -1), (-1, 0), (-1, 1)]))

        self.assertEqual(set(gid(grid, [1,  1])), vecs_to_set([(1, 1)]))
        self.assertEqual(set(gid(grid, [1, -1])), vecs_to_set([(1, -1)]))
        self.assertEqual(set(gid(grid, [-1, 1])), vecs_to_set([(-1, 1)]))
        self.assertEqual(set(gid(grid, [-1, -1])), vecs_to_set([(-1, -1)]))

        grid = sym.D3Q19
        self.assertEqual(set(gid(grid, [0,0,1])), vecs_to_set([(0,0,1), (0,1,1), (0,-1,1), (1,0,1), (-1,0,1)]))
        self.assertEqual(set(gid(grid, [0,0,-1])), vecs_to_set([(0,0,-1), (0,1,-1), (0,-1,-1), (1,0,-1), (-1,0,-1)]))
        self.assertEqual(set(gid(grid, [0,1,0])), vecs_to_set([(0,1,0), (0,1,1), (0,1,-1), (1,1,0), (-1,1,0)]))
        self.assertEqual(set(gid(grid, [0,-1,0])), vecs_to_set([(0,-1,0), (0,-1,1), (0,-1,-1), (1,-1,0), (-1,-1,0)]))
        self.assertEqual(set(gid(grid, [1,0,0])), vecs_to_set([(1,0,0), (1,0,1), (1,0,-1), (1,1,0), (1,-1,0)]))
        self.assertEqual(set(gid(grid, [-1,0,0])), vecs_to_set([(-1,0,0), (-1,0,1), (-1,0,-1), (-1,1,0), (-1,-1,0)]))

        self.assertEqual(set(gid(grid, [0,1,1])), vecs_to_set([(0,1,1)]))
        self.assertEqual(set(gid(grid, [1,1,1])), vecs_to_set([]))
        self.assertEqual(set(gid(grid, [-1,1,0])), vecs_to_set([(-1,1,0)]))
        self.assertEqual(set(gid(grid, [-1,1,1])), vecs_to_set([]))
        self.assertEqual(set(gid(grid, [-1,1,-1])), vecs_to_set([]))

        self.assertEqual(set(gid(grid, [-1,0,1])), vecs_to_set([(-1,0,1)]))
        self.assertEqual(set(gid(grid, [-1,-1,0])), vecs_to_set([(-1,-1,0)]))
        self.assertEqual(set(gid(grid, [-1,0,-1])), vecs_to_set([(-1,0,-1)]))


        grid = sym.D3Q15
        self.assertEqual(set(gid(grid, [1,1,1])), vecs_to_set([(1,1,1)]))
        self.assertEqual(set(gid(grid, [0,1,1])), vecs_to_set([(1,1,1), (-1,1,1)]))


if __name__ == '__main__':
    unittest.main()

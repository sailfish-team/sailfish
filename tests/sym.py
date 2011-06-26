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
        vts = vecs_to_set

        self.assertEqual(set(gid(grid, [0,  1])), vts([(0, 1), (1, 1), (-1, 1)]))
        self.assertEqual(set(gid(grid, [0, -1])), vts([(0, -1), (1, -1), (-1, -1)]))
        self.assertEqual(set(gid(grid, [1,  0])), vts([(1, -1), (1, 0), (1, 1)]))
        self.assertEqual(set(gid(grid, [-1, 0])), vts([(-1, -1), (-1, 0), (-1, 1)]))

        self.assertEqual(set(gid(grid, [1,  1])), vts([(1, 1)]))
        self.assertEqual(set(gid(grid, [1, -1])), vts([(1, -1)]))
        self.assertEqual(set(gid(grid, [-1, 1])), vts([(-1, 1)]))
        self.assertEqual(set(gid(grid, [-1, -1])), vts([(-1, -1)]))

        grid = sym.D3Q19
        self.assertEqual(set(gid(grid, [0,0,1])), vts([(0,0,1), (0,1,1), (0,-1,1), (1,0,1), (-1,0,1)]))
        self.assertEqual(set(gid(grid, [0,0,-1])), vts([(0,0,-1), (0,1,-1), (0,-1,-1), (1,0,-1), (-1,0,-1)]))
        self.assertEqual(set(gid(grid, [0,1,0])), vts([(0,1,0), (0,1,1), (0,1,-1), (1,1,0), (-1,1,0)]))
        self.assertEqual(set(gid(grid, [0,-1,0])), vts([(0,-1,0), (0,-1,1), (0,-1,-1), (1,-1,0), (-1,-1,0)]))
        self.assertEqual(set(gid(grid, [1,0,0])), vts([(1,0,0), (1,0,1), (1,0,-1), (1,1,0), (1,-1,0)]))
        self.assertEqual(set(gid(grid, [-1,0,0])), vts([(-1,0,0), (-1,0,1), (-1,0,-1), (-1,1,0), (-1,-1,0)]))

        self.assertEqual(set(gid(grid, [0,1,1])), vts([(0,1,1)]))
        self.assertEqual(set(gid(grid, [1,1,1])), vts([]))
        self.assertEqual(set(gid(grid, [-1,1,0])), vts([(-1,1,0)]))
        self.assertEqual(set(gid(grid, [-1,1,1])), vts([]))
        self.assertEqual(set(gid(grid, [-1,1,-1])), vts([]))

        self.assertEqual(set(gid(grid, [-1,0,1])), vts([(-1,0,1)]))
        self.assertEqual(set(gid(grid, [-1,-1,0])), vts([(-1,-1,0)]))
        self.assertEqual(set(gid(grid, [-1,0,-1])), vts([(-1,0,-1)]))

        grid = sym.D3Q15
        self.assertEqual(set(gid(grid, [1,1,1])), vts([(1,1,1)]))
        self.assertEqual(set(gid(grid, [0,1,1])), vts([(1,1,1), (-1,1,1)]))

    def test_prop_dists(self):
        grid = sym.D2Q9
        gpd = sym.get_prop_dists

        def vecs_to_set(vecs):
            ret = set()
            for vec in vecs:
                ret.add(grid.vec_idx(vec))
            return ret
        vts = vecs_to_set

        self.assertEqual(set(gpd(grid, 1, 0)), vts([(1, 0), (1, 1), (1, -1)]))
        self.assertEqual(set(gpd(grid, -1, 0)), vts([(-1, 0), (-1, 1), (-1, -1)]))
        self.assertEqual(set(gpd(grid, 1, 1)), vts([(0, 1), (-1, 1), (1, 1)]))
        self.assertEqual(set(gpd(grid, -1, 1)), vts([(0, -1), (-1, -1), (1, -1)]))

        grid = sym.D3Q19
        self.assertEqual(set(gpd(grid, 1, 0)), vts([(1, 0, 0), (1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1)]))
        self.assertEqual(set(gpd(grid, 1, 2)), vts([(0, 0, 1), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1)]))


if __name__ == '__main__':
    unittest.main()

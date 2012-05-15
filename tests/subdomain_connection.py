import numpy as np
import unittest
from sailfish.config import LBConfig
from sailfish.controller import LBGeometryProcessor
from sailfish.geo import LBGeometry2D
from sailfish.subdomain import Subdomain2D, SubdomainSpec2D, SubdomainSpec3D
from sailfish.sym import D2Q9, D3Q15, D3Q19

vi = lambda x, y: D2Q9.vec_idx([x, y])
vi3 = lambda x, y, z: D3Q19.vec_idx([x, y, z])
vi15 = lambda x, y, z: D3Q15.vec_idx([x, y, z])

def _verify_partial_map(self, conn, expected_map):
    self.assertEqual(set(conn.dst_partial_map.keys()),
            set(expected_map.keys()))
    for key, val in expected_map.iteritems():
        self.assertEqual(
                set([tuple(x) for x in val]),
                set([tuple(x) for x in conn.dst_partial_map[key]]))


class TestBlock3D(unittest.TestCase):
    def test_subdomain_connection_y(self):
        base = SubdomainSpec3D((10, 10, 10), (10, 10, 12), envelope_size=1, id_=0)
        face_hi = SubdomainSpec3D.Y_HIGH

        # exact match
        b1 = SubdomainSpec3D((10, 20, 10), (10, 5, 12), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D3Q19))
        cpair = base.get_connection(face_hi, b1.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(1, 13)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(1, 13)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(1, 13)])
        self.assertEqual(cpair.src.dst_low, [0,0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 9), slice(1, 11)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 9), slice(1, 11)])

        # Order of axes in the connection buffer is: x, z
        # Edges
        l = [(x, 0) for x in range(1, 9)]
        r = [(x, 11) for x in range(1, 9)]
        t = [(9, z) for z in range(1, 11)]
        b = [(0, z) for z in range(1, 11)]

        # Corners
        tl = [(9, 0)]
        tr = [(9, 11)]
        bl = [(0, 0)]
        br = [(0, 11)]

        expected_map = {
                vi3(0,1,0): l + r + t + b + tl + tr + br + bl,
                vi3(1,1,0): l + r + t + tl + tr,
                vi3(-1,1,0): l + r + b + bl + br,
                vi3(0,1,1): r + t + b + tr + br,
                vi3(0,1,-1): l + t + b + tl + bl,
            }
        _verify_partial_map(self, cpair.src, expected_map)

    def test_subdomain_connection_z(self):
        base = SubdomainSpec3D((10, 10, 10), (10, 12, 10), envelope_size=1, id_=0)
        face_hi = SubdomainSpec3D.Z_HIGH

        # exact match
        b1 = SubdomainSpec3D((10, 10, 20), (10, 12, 5), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D3Q19))
        cpair = base.get_connection(face_hi, b1.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(1, 13)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(1, 13)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(1, 13)])
        self.assertEqual(cpair.src.dst_low, [0,0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 9), slice(1, 11)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 9), slice(1, 11)])

        # Order of axes in the connection buffer is: x, y
        # Edges
        l = [(x, 0) for x in range(1, 9)]
        r = [(x, 11) for x in range(1, 9)]
        t = [(9, y) for y in range(1, 11)]
        b = [(0, y) for y in range(1, 11)]

        # Corners
        tl = [(9, 0)]
        tr = [(9, 11)]
        bl = [(0, 0)]
        br = [(0, 11)]

        expected_map = {
                vi3(0,0,1): l + r + t + b + tl + tr + br + bl,
                vi3(1,0,1): l + r + t + tl + tr,
                vi3(-1,0,1): l + r + b + bl + br,
                vi3(0,1,1): r + t + b + tr + br,
                vi3(0,-1,1): l + t + b + tl + bl,
            }
        _verify_partial_map(self, cpair.src, expected_map)

    def test_subdomain_connection_x(self):
        base = SubdomainSpec3D((10, 10, 10), (10, 12, 10), envelope_size=1, id_=0)
        face_hi = SubdomainSpec3D.X_HIGH

        # exact match
        b1 = SubdomainSpec3D((20, 10, 10), (5, 12, 10), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D3Q19))
        cpair = base.get_connection(face_hi, b1.id)
        self.assertEqual(set(cpair.src.dists),
                         set([vi3(1,0,0), vi3(1,1,0), vi3(1,-1,0),
                              vi3(1,0,1), vi3(1,0,-1)]))
        self.assertEqual(cpair.src.src_slice, [slice(1, 13), slice(1, 11)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 13), slice(1, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 13), slice(1, 11)])
        self.assertEqual(cpair.src.dst_low, [0,0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 11), slice(1, 9)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 11), slice(1, 9)])

        # Order of axes in the connection buffer is y, z
        # Edges
        l = [(y, 0) for y in range(1, 11)]
        r = [(y, 9) for y in range(1, 11)]
        t = [(11, z) for z in range(1, 9)]
        b = [(0, z) for z in range(1, 9)]

        # Corners
        tl = [(11, 0)]
        tr = [(11, 9)]
        bl = [(0, 0)]
        br = [(0, 9)]

        expected_map = {
                vi3(1,0,0): l + r + t + b + tl + tr + br + bl,
                vi3(1,1,0): l + r + t + tl + tr,
                vi3(1,-1,0): l + r + b + bl + br,
                vi3(1,0,1): r + t + b + tr + br,
                vi3(1,0,-1): l + t + b + tl + bl,
            }
        _verify_partial_map(self, cpair.src, expected_map)

        base = SubdomainSpec3D((10, 10, 10), (10, 10, 10), envelope_size=1, id_=0)

        # full overlap (2nd subdomain is smaller)
        b2 = SubdomainSpec3D((20, 12, 14), (5, 6, 4), envelope_size=1, id_=2)
        self.assertTrue(base.connect(b2, grid=D3Q19))
        cpair = base.get_connection(face_hi, b2.id)
        self.assertEqual(cpair.src.src_slice, [slice(3, 9), slice(5, 9)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(2, 10), slice(4, 10)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(3, 9), slice(5, 9)])
        self.assertEqual(cpair.src.dst_low, [0,0])
        self.assertEqual(cpair.src.dst_slice, [slice(0, 6), slice(0, 4)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(0, 6), slice(0, 4)])

        # full overlap (2nd subdomain is larger)
        b3 = SubdomainSpec3D((20, 8, 9), (5, 14, 15), envelope_size=1, id_=3)
        self.assertTrue(base.connect(b3, grid=D3Q19))
        cpair = base.get_connection(face_hi, b3.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 12), slice(0, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(1, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(0, 12), slice(0, 12)])
        self.assertEqual(cpair.src.dst_low, [1, 0])
        self.assertEqual(cpair.src.dst_slice, [slice(3, 11), slice(2, 10)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(2, 10), slice(2, 10)])

        # Edges
        l = [(y, 1) for y in range(2, 10)]
        r = [(y, 10) for y in range(2, 10)]
        t = [(10, z) for z in range(2, 10)]
        b = [(1, z) for z in range(2, 10)]

        le = [(y, 0) for y in range(1, 11)]
        re = [(y, 11) for y in range(1, 11)]
        te = [(11, z) for z in range(1, 11)]
        be = [(0, z) for z in range(1, 11)]

        # Corners
        tl = [(10, 1)]
        tr = [(10, 10)]
        bl = [(1, 1)]
        br = [(1, 10)]

        expected_map = {
                vi3(1,0,0): l + r + t + b + tl + tr + br + bl,
                vi3(1,1,0): l + r + t + tl + tr + te,
                vi3(1,-1,0): l + r + b + bl + br + be,
                vi3(1,0,1): r + t + b + tr + br + re,
                vi3(1,0,-1): l + t + b + tl + bl + le,
            }
        _verify_partial_map(self, cpair.src, expected_map)


        # top-left corner match (no connection in D3Q19 topology)
        bf1 = SubdomainSpec3D((20, 20, 5), (5, 5, 5), envelope_size=1)
        self.assertFalse(base.connect(bf1, grid=D3Q19))

        # too far along X axis
        bf2 = SubdomainSpec3D((21, 10, 10), (5, 10, 10), envelope_size=1)
        self.assertFalse(base.connect(bf2, grid=D3Q19))

        # too far along Y axis
        bf3 = SubdomainSpec3D((20, 21, 10), (5, 10, 10), envelope_size=1)
        self.assertFalse(base.connect(bf3, grid=D3Q19))

    def _x_edge_helper(self, face):
        """Tests connections along the edges of one of the faces orthogonal
        to the X axis."""
        base = SubdomainSpec3D((10, 10, 10), (10, 10, 10), envelope_size=1, id_=0)

        if face == SubdomainSpec3D.X_LOW:
            x_low = 5
            x_dir = -1
        else:
            x_low = 20
            x_dir = 1

        # bottom edge match
        b1 = SubdomainSpec3D((x_low, 5, 10), (5, 5, 10), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D3Q19))
        cpair = base.get_connection(face, b1.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 1), slice(1, 11)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 2), slice(1, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(0, 1), slice(1, 11)])
        self.assertEqual(cpair.src.dst_low, [4, 0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])

        expected_map = {
                vi3(x_dir,-1,0): [(0, z) for z in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        # top edge match
        b2 = SubdomainSpec3D((x_low, 20, 10), (5, 5, 10), envelope_size=1, id_=2)
        self.assertTrue(base.connect(b2, grid=D3Q19))
        cpair = base.get_connection(face, b2.id)
        self.assertEqual(cpair.src.src_slice, [slice(11, 12), slice(1, 11)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(10, 11), slice(1, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(11, 12), slice(1, 11)])
        self.assertEqual(cpair.src.dst_low, [0, 0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])

        expected_map = {
                vi3(x_dir,1,0): [(0, z) for z in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        # right edge match
        b3 = SubdomainSpec3D((x_low, 10, 20), (5, 10, 5), envelope_size=1, id_=3)
        self.assertTrue(base.connect(b3, grid=D3Q19))
        cpair = base.get_connection(face, b3.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(11, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(10, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(11, 12)])
        self.assertEqual(cpair.src.dst_low, [0, 0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])

        expected_map = {
                vi3(x_dir,0,1): [(y, 0) for y in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        # left edge match
        b4 = SubdomainSpec3D((x_low, 10, 5), (5, 10, 5), envelope_size=1, id_=4)
        self.assertTrue(base.connect(b4, grid=D3Q19))
        cpair = base.get_connection(face, b4.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(0, 1)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(1, 2)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(0, 1)])
        self.assertEqual(cpair.src.dst_low, [0, 4])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])

        expected_map = {
                vi3(x_dir,0,-1): [(y, 0) for y in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

    def test_subdomain_connection_edge_x_low(self):
        self._x_edge_helper(SubdomainSpec3D.X_LOW)

    def test_subdomain_connection_edge_x_high(self):
        self._x_edge_helper(SubdomainSpec3D.X_HIGH)

    def test_subdomain_connection_edge_non_x(self):
        base = SubdomainSpec3D((10, 10, 10), (10, 10, 10), envelope_size=1, id_=0)
        b1 = SubdomainSpec3D((10, 5, 5), (10, 5, 5), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D3Q19))
        cpair = base.get_connection(SubdomainSpec3D.Y_LOW, b1.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(0, 1)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(1, 2)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(0, 1)])
        self.assertEqual(cpair.src.dst_low, [0, 4])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi3(0, -1, -1): [(x, 0) for x in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        b2 = SubdomainSpec3D((10, 20, 5), (10, 5, 5), envelope_size=1, id_=2)
        self.assertTrue(base.connect(b2, grid=D3Q19))
        cpair = base.get_connection(SubdomainSpec3D.Y_HIGH, b2.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(0, 1)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(1, 2)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(0, 1)])
        self.assertEqual(cpair.src.dst_low, [0, 4])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi3(0, 1, -1): [(x, 0) for x in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        b3 = SubdomainSpec3D((10, 5, 20), (10, 5, 5), envelope_size=1, id_=3)
        self.assertTrue(base.connect(b3, grid=D3Q19))
        cpair = base.get_connection(SubdomainSpec3D.Y_LOW, b3.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(11, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(10, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(11, 12)])
        self.assertEqual(cpair.src.dst_low, [0, 0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi3(0, -1, 1): [(x, 0) for x in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        b4 = SubdomainSpec3D((10, 20, 20), (10, 5, 5), envelope_size=1, id_=4)
        self.assertTrue(base.connect(b4, grid=D3Q19))
        cpair = base.get_connection(SubdomainSpec3D.Y_HIGH, b4.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 11), slice(11, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11), slice(10, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11), slice(11, 12)])
        self.assertEqual(cpair.src.dst_low, [0, 0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi3(0, 1, 1): [(x, 0) for x in range(0, 10)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

    def _corner_helper(self, face):
        base = SubdomainSpec3D((10, 10, 10), (10, 10, 10), envelope_size=1, id_=0)

        if face == SubdomainSpec3D.X_LOW:
            x_low = 5
            x_dir = -1
        else:
            x_low = 20
            x_dir = 1

        b1 = SubdomainSpec3D((x_low, 5, 5), (5, 5, 5), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D3Q15))
        cpair = base.get_connection(face, b1.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 1), slice(0, 1)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 2), slice(1, 2)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(0, 1), slice(0, 1)])
        self.assertEqual(cpair.src.dst_low, [4, 4])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi15(x_dir, -1, -1): [(0, 0)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        b2 = SubdomainSpec3D((x_low, 20, 5), (5, 5, 5), envelope_size=1, id_=2)
        self.assertTrue(base.connect(b2, grid=D3Q15))
        cpair = base.get_connection(face, b2.id)
        self.assertEqual(cpair.src.src_slice, [slice(11, 12), slice(0, 1)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(10, 11), slice(1, 2)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(11, 12), slice(0, 1)])
        self.assertEqual(cpair.src.dst_low, [0, 4])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi15(x_dir, 1, -1): [(0, 0)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        b3 = SubdomainSpec3D((x_low, 20, 20), (5, 5, 5), envelope_size=1, id_=3)
        self.assertTrue(base.connect(b3, grid=D3Q15))
        cpair = base.get_connection(face, b3.id)
        self.assertEqual(cpair.src.src_slice, [slice(11, 12), slice(11, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(10, 11), slice(10, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(11, 12), slice(11, 12)])
        self.assertEqual(cpair.src.dst_low, [0, 0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi15(x_dir, 1, 1): [(0, 0)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

        b4 = SubdomainSpec3D((x_low, 5, 20), (5, 5, 5), envelope_size=1, id_=4)
        self.assertTrue(base.connect(b4, grid=D3Q15))
        cpair = base.get_connection(face, b4.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 1), slice(11, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 2), slice(10, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(0, 1), slice(11, 12)])
        self.assertEqual(cpair.src.dst_low, [4, 0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi15(x_dir, -1, 1): [(0, 0)]
            }
        _verify_partial_map(self, cpair.src, expected_map)

    def test_corner_x_low(self):
        self._corner_helper(SubdomainSpec3D.X_LOW)

    def test_corner_x_high(self):
        self._corner_helper(SubdomainSpec3D.X_HIGH)


class TestBlock2D(unittest.TestCase):

    def _verify_partial_map(self, conn, expected_map):
        self.assertEqual(conn.dst_partial_map.keys(),
                expected_map.keys())
        for key, val in expected_map.iteritems():
            self.assertTrue(np.all(val == conn.dst_partial_map[key]))

    def _test_subdomain_conn(self, axis):
        # All coordinate tuples below are specified for the case
        # in which the two subdomains are connected along the X axis.
        # Y-axis connection requires a swap.
        def f(a, b):
            if axis == 0:
                return (a,b)
            else:
                return (b,a)

        face_hi = SubdomainSpec2D.axis_dir_to_face(axis, 1)
        face_lo = SubdomainSpec2D.axis_dir_to_face(axis, -1)
        base = SubdomainSpec2D(f(10, 10), f(10, 10), envelope_size=1, id_=0)

        # exact match
        b1 = SubdomainSpec2D(f(20, 10), f(5, 10), envelope_size=1, id_=1)
        self.assertTrue(base.connect(b1, grid=D2Q9))
        cpair = base.get_connection(face_hi, b1.id)
        self.assertEqual(set(cpair.src.dists),
                         set([vi(*f(1,0)), vi(*f(1,1)), vi(*f(1,-1))]))
        self.assertEqual(cpair.src.src_slice, [slice(1, 11)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(1, 11)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 9)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 9)])
        expected_map = {vi(*f(1,-1)): np.array([[0]]),
                        vi(*f(1,1)): np.array([[9]]),
                        vi(*f(1,0)): np.array([[0],[9]])}
        _verify_partial_map(self, cpair.src, expected_map)

        self.assertEqual(cpair.src.elements, 30)
        self.assertEqual(cpair.src.transfer_shape, [3, 10])

        # partal overlap
        b2 = SubdomainSpec2D(f(20, 5), f(5, 10), envelope_size=1, id_=2)
        self.assertTrue(base.connect(b2, grid=D2Q9))
        cpair = base.get_connection(face_hi, b2.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 6)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 7)])
        self.assertEqual(cpair.src.dst_macro_slice, [slice(0, 6)])
        self.assertEqual(cpair.src.dst_low, [4])
        self.assertEqual(cpair.src.dst_slice, [slice(6, 10)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(2, 6)])
        expected_map = {vi(*f(1,-1)): np.array([[0], [1]]),
                        vi(*f(1,0)): np.array([[1]])}
        _verify_partial_map(self, cpair.src, expected_map)

        # full overlap (2nd subdomain is smaller)
        b3 = SubdomainSpec2D(f(20, 12), f(5, 7), envelope_size=1, id_=3)
        self.assertTrue(base.connect(b3, grid=D2Q9))
        cpair = base.get_connection(face_hi, b3.id)
        self.assertEqual(cpair.src.src_slice, [slice(3, 10)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(2, 11)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [slice(0, 7)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(0, 7)])
        self.assertFalse(cpair.src.dst_partial_map)

        self.assertEqual(cpair.dst.src_slice, [slice(0, 9)])
        self.assertEqual(cpair.dst.src_macro_slice, [slice(1, 8)])
        self.assertEqual(cpair.dst.dst_low, [1])
        self.assertEqual(cpair.dst.dst_slice, [slice(3, 8)])
        self.assertEqual(cpair.dst.dst_full_buf_slice, [slice(2, 7)])

        # full overlap (2nd subdomain is larger)
        b4 = SubdomainSpec2D(f(20, 8), f(5, 14), envelope_size=1, id_=4)
        self.assertTrue(base.connect(b4, grid=D2Q9))
        cpair = base.get_connection(face_hi, b4.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11)])
        self.assertEqual(cpair.src.dst_low, [1])
        self.assertEqual(cpair.src.dst_slice, [slice(3, 11)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(2, 10)])
        expected_map = {
                vi(*f(1,-1)): np.array([[0], [1]]),
                vi(*f(1,0)): np.array([[1], [10]]),
                vi(*f(1,1)): np.array([[10], [11]])}
        _verify_partial_map(self, cpair.src, expected_map)

        # exact match at the bottom (2nd subdomain is smaller)
        b5 = SubdomainSpec2D(f(20, 10), f(5, 5), envelope_size=1, id_=5)
        self.assertTrue(base.connect(b5, grid=D2Q9))
        cpair = base.get_connection(face_hi, b5.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 6)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 7)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 5)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 5)])
        expected_map = {
                vi(*f(1,-1)): np.array([[0]]),
                vi(*f(1,0)): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)

        # exact match at the bottom (2nd subdomain is larger)
        b6 = SubdomainSpec2D(f(20, 10), f(5, 15), envelope_size=1, id_=6)
        self.assertTrue(base.connect(b6, grid=D2Q9))
        cpair = base.get_connection(face_hi, b6.id)
        self.assertEqual(cpair.src.src_slice, [slice(1, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 11)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [slice(1, 9)])
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 9)])
        expected_map = {
                vi(*f(1,-1)): np.array([[0]]),
                vi(*f(1,0)): np.array([[0], [9]]),
                vi(*f(1,1)): np.array([[9], [10]]),
            }
        _verify_partial_map(self, cpair.src, expected_map)

        # note that the size of the src_slice is different, depending
        # on the direction of the transfer
        self.assertEqual(cpair.dst.src_slice, [slice(1, 11)])
        self.assertEqual(cpair.dst.src_macro_slice, [slice(1, 12)])
        self.assertEqual(cpair.dst.dst_low, [0])
        self.assertEqual(cpair.dst.dst_slice, [slice(1,10)])
        self.assertEqual(cpair.dst.dst_full_buf_slice, [slice(1,10)])

        # disconnected subdomains
        bf1 = SubdomainSpec2D(f(20, 21), f(5, 10), envelope_size=1)
        bf2 = SubdomainSpec2D(f(20, 5),  f(5, 4), envelope_size=1)
        bf3 = SubdomainSpec2D(f(19, 10), f(5, 10), envelope_size=1)
        bf4 = SubdomainSpec2D(f(21, 10), f(5, 10), envelope_size=1)
        self.assertFalse(base.connect(bf1))
        self.assertFalse(base.connect(bf2))
        self.assertFalse(base.connect(bf3))
        self.assertFalse(base.connect(bf4))

    def test_subdomain_connection_x(self):
        self._test_subdomain_conn(0)

    def test_subdomain_connection_y(self):
        self._test_subdomain_conn(1)

    def test_corner_connection(self):
        base = SubdomainSpec2D((10, 10), (10, 10), envelope_size=1, id_=0)
        # corner match (low)
        b6 = SubdomainSpec2D((20, 5), (5, 5), envelope_size=1, id_=6)
        self.assertTrue(base.connect(b6, grid=D2Q9))
        cpair = base.get_connection(SubdomainSpec2D.X_HIGH, b6.id)
        self.assertEqual(cpair.src.src_slice, [slice(0, 1)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(1, 2)])
        self.assertEqual(cpair.src.dst_low, [4])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi(1,-1): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)

        self.assertEqual(cpair.dst.src_slice, [slice(6, 7)])
        self.assertEqual(cpair.dst.src_macro_slice, [slice(5, 6)])
        self.assertEqual(cpair.dst.dst_low, [0])
        self.assertEqual(cpair.dst.dst_slice, [])
        self.assertEqual(cpair.dst.dst_full_buf_slice, [])
        expected_map = {
                vi(-1,1): np.array([[0]])}
        _verify_partial_map(self, cpair.dst, expected_map)

        # corner match (high)
        b7 = SubdomainSpec2D((20, 20), (5, 5), envelope_size=1, id_=7)
        self.assertTrue(base.connect(b7, grid=D2Q9))
        cpair = base.get_connection(SubdomainSpec2D.X_HIGH, b7.id)
        self.assertEqual(cpair.src.src_slice, [slice(11, 12)])
        self.assertEqual(cpair.src.src_macro_slice, [slice(10, 11)])
        self.assertEqual(cpair.src.dst_low, [0])
        self.assertEqual(cpair.src.dst_slice, [])
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {
                vi(1,1): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)

        self.assertEqual(cpair.dst.src_slice, [slice(0, 1)])
        self.assertEqual(cpair.dst.src_macro_slice, [slice(1, 2)])
        self.assertEqual(cpair.dst.dst_low, [9])
        self.assertEqual(cpair.dst.dst_slice, [])
        self.assertEqual(cpair.dst.dst_full_buf_slice, [])
        expected_map = {
                vi(-1,-1): np.array([[0]])}
        _verify_partial_map(self, cpair.dst, expected_map)


class TestBlock2DPeriodic(unittest.TestCase):

    def _check_partial_map(self, cpairs, src_slice, dst_partial_map):
        for cpair in cpairs:
            if cpair.src.src_slice == src_slice:
                _verify_partial_map(self, cpair.src, dst_partial_map)
                return True

        return False


    def test_4subdomains(self):
        config = LBConfig()
        config.lat_nx = 64
        config.lat_ny = 64
        config.periodic_x = True
        config.periodic_y = True
        config.grid = 'D2Q9'

        geo = LBGeometry2D(config)
        b1 = SubdomainSpec2D((0, 0), (32, 32), envelope_size=1, id_=1)
        b2 = SubdomainSpec2D((32, 0), (32, 32), envelope_size=1, id_=2)
        b3 = SubdomainSpec2D((0, 32), (32, 32), envelope_size=1, id_=3)
        b4 = SubdomainSpec2D((32, 32), (32, 32), envelope_size=1, id_=4)

        proc = LBGeometryProcessor([b1, b2, b3, b4], 2, geo)
        proc._connect_subdomains(config)

        ## b1 - b4

        cpairs = b1.get_connections(SubdomainSpec2D.X_LOW, b4.id)
        self.assertEqual(len(cpairs), 2)
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(0, 1)],
                    {vi(-1, -1): np.array([[0]])}))
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(33, 34)],
                    {vi(-1, 1): np.array([[0]])}))

        cpairs = b1.get_connections(SubdomainSpec2D.X_HIGH, b4.id)
        self.assertEqual(len(cpairs), 2)
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(0, 1)],
                    {vi(1, -1): np.array([[0]])}))
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(33, 34)],
                    {vi(1, 1): np.array([[0]])}))

        cpairs = b1.get_connections(SubdomainSpec2D.Y_LOW, b4.id)
        self.assertEqual(len(cpairs), 0)

        cpairs = b1.get_connections(SubdomainSpec2D.Y_HIGH, b4.id)
        self.assertEqual(len(cpairs), 0)

        ### b2 - b3

        cpairs = b2.get_connections(SubdomainSpec2D.X_LOW, b3.id)

        self.assertEqual(len(cpairs), 2)
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(0, 1)],
                    {vi(-1, -1): np.array([[0]])}))
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(33, 34)],
                    {vi(-1, 1): np.array([[0]])}))

        cpairs = b2.get_connections(SubdomainSpec2D.X_HIGH, b3.id)
        self.assertEqual(len(cpairs), 2)
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(0, 1)],
                    {vi(1, -1): np.array([[0]])}))
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(33, 34)],
                    {vi(1, 1): np.array([[0]])}))

        cpairs = b2.get_connections(SubdomainSpec2D.Y_LOW, b3.id)
        self.assertEqual(len(cpairs), 0)

        cpairs = b2.get_connections(SubdomainSpec2D.Y_HIGH, b3.id)
        self.assertEqual(len(cpairs), 0)

    def test_5subdomains(self):
        config = LBConfig()
        config.lat_nx = 50
        config.lat_ny = 75
        config.periodic_x = True
        config.periodic_y = True
        config.grid = 'D2Q9'

        geo = LBGeometry2D(config)
        b1 = SubdomainSpec2D((0, 0), (25, 25), envelope_size=1, id_=1)
        b2 = SubdomainSpec2D((25, 0), (25, 25), envelope_size=1, id_=2)
        b3 = SubdomainSpec2D((0, 25), (50, 25), envelope_size=1, id_=3)
        b4 = SubdomainSpec2D((0, 50), (25, 25), envelope_size=1, id_=4)
        b5 = SubdomainSpec2D((25, 50), (25, 25), envelope_size=1, id_=5)

        proc = LBGeometryProcessor([b1, b2, b3, b4, b5], 2, geo)
        proc._connect_subdomains(config)

        cpair = b1.get_connection(SubdomainSpec2D.X_LOW, b5.id)
        expected_map = {vi(-1, -1): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)

        cpair = b1.get_connection(SubdomainSpec2D.X_HIGH, b5.id)
        expected_map = {vi(1, -1): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)

        cpair = b1.get_connection(SubdomainSpec2D.Y_LOW, b4.id)
        expected_map = {vi(0, -1): np.array([[0], [24]]),
                        vi(1, -1): np.array([[24]]),
                        vi(-1, -1): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)

        cpair = b1.get_connection(SubdomainSpec2D.X_LOW, b2.id)
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 24)])

        cpair = b1.get_connection(SubdomainSpec2D.X_HIGH, b2.id)
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 24)])

        cpair = b1.get_connection(SubdomainSpec2D.Y_HIGH, b3.id)
        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 24)])

        cpair = b1.get_connection(SubdomainSpec2D.X_LOW, b3.id)
        self.assertEqual(cpair.src.dst_full_buf_slice, [])
        expected_map = {vi(-1, 1): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)


    def test_3subdomains(self):
        config = LBConfig()
        config.lat_nx = 64
        config.lat_ny = 64
        config.periodic_x = True
        config.periodic_y = False
        config.grid = 'D2Q9'

        geo = LBGeometry2D(config)
        b1 = SubdomainSpec2D((0, 0), (32, 64), envelope_size=1, id_=1)
        b2 = SubdomainSpec2D((32, 0), (8, 64), envelope_size=1, id_=2)
        b3 = SubdomainSpec2D((40, 0), (24, 64), envelope_size=1, id_=3)

        proc = LBGeometryProcessor([b1, b2, b3], 2, geo)
        proc._connect_subdomains(config)

        cpair = b1.get_connection(SubdomainSpec2D.X_LOW, b3.id)

        self.assertEqual(cpair.src.dst_full_buf_slice, [slice(1, 63)])
        expected_map = {
            vi(-1, 0): np.array([[0], [63]]),
            vi(-1, 1): np.array([[63]]),
            vi(-1, -1): np.array([[0]])}
        _verify_partial_map(self, cpair.src, expected_map)

    def test_2subdomains_partial(self):
        config = LBConfig()
        config.lat_nx = 64
        config.lat_ny = 64
        config.periodic_x = False
        config.periodic_y = True
        config.grid = 'D2Q9'

        geo = LBGeometry2D(config)
        b1 = SubdomainSpec2D((0, 0), (32, 64), envelope_size=1, id_=1)
        b2 = SubdomainSpec2D((32, 0), (32, 64), envelope_size=1, id_=2)

        proc = LBGeometryProcessor([b1, b2], 2, geo)
        proc._connect_subdomains(config)

        cpairs = b1.get_connections(SubdomainSpec2D.X_HIGH, b2.id)

        self.assertEqual(len(cpairs), 3)
        self.assertTrue(
            self._check_partial_map(cpairs, [slice(0, 1)],
                {vi(1, -1): np.array([[0]])}))

        self.assertTrue(
            self._check_partial_map(cpairs, [slice(1, 65)],
                {vi(1, -1): np.array([[0]]),
                 vi(1, 0): np.array([[0], [63]]),
                 vi(1, 1): np.array([[63]])}))

        self.assertTrue(
            self._check_partial_map(cpairs, [slice(65, 66)],
                {vi(1, 1): np.array([[0]])}))

    def test_2subdomains(self):
        config = LBConfig()
        config.lat_nx = 64
        config.lat_ny = 64
        config.periodic_x = True
        config.periodic_y = True
        config.grid = 'D2Q9'

        geo = LBGeometry2D(config)
        b1 = SubdomainSpec2D((0, 0), (32, 64), envelope_size=1, id_=1)
        b2 = SubdomainSpec2D((32, 0), (32, 64), envelope_size=1, id_=2)

        proc = LBGeometryProcessor([b1, b2], 2, geo)
        proc._connect_subdomains(config)

        cpairs = b1.get_connections(SubdomainSpec2D.X_HIGH, b2.id)
        self.assertEqual(len(cpairs), 3)

        cpairs = b1.get_connections(SubdomainSpec2D.X_LOW, b2.id)
        self.assertTrue(
                self._check_partial_map(cpairs, [slice(1, 65)],
                {vi(-1, -1): np.array([[0]]),
                 vi(-1, 0): np.array([[0], [63]]),
                 vi(-1, 1): np.array([[63]])}))

        self.assertTrue(
            self._check_partial_map(cpairs, [slice(0, 1)],
                {vi(-1, -1): np.array([[0]])}))

        self.assertTrue(
            self._check_partial_map(cpairs, [slice(65, 66)],
                {vi(-1, 1): np.array([[0]])}))


if __name__ == '__main__':
    unittest.main()

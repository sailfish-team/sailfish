import unittest

from sailfish.config import MachineSpec
from sailfish import controller
from sailfish.geo_block import SubdomainSpec2D

class TestSubdomainDistribution(unittest.TestCase):
    def test_1_1_mapping(self):
        nodes = [
                MachineSpec('a', 'a', gpus=[0]),
                MachineSpec('b', 'b', gpus=[1])
            ]
        subds = [
                SubdomainSpec2D((0,0), (10, 10), id_=0),
                SubdomainSpec2D((0,10), (10, 10), id_=1)
            ]

        assignments = controller.split_subdomains_between_nodes(nodes, subds)
        self.assertEqual(assignments, [[subds[0]], [subds[1]]])

    def test_1_1_mapping_multigpi(self):
        nodes = [
                MachineSpec('a', 'a', gpus=[0, 1]),
                MachineSpec('b', 'b', gpus=[1])
            ]
        subds = [
                SubdomainSpec2D((0,0), (10, 10), id_=0),
                SubdomainSpec2D((0,10), (10, 10), id_=1),
                SubdomainSpec2D((0,20), (10, 10), id_=2)
            ]

        assignments = controller.split_subdomains_between_nodes(nodes, subds)
        self.assertEqual(assignments, [[subds[0], subds[1]], [subds[2]]])

    def test_more_nodes_than_subdomains(self):
        nodes = [
                MachineSpec('a', 'a', gpus=[0]),
                MachineSpec('b', 'b', gpus=[1])
            ]
        subds = [
                SubdomainSpec2D((0,0), (10, 10), id_=0),
            ]

        assignments = controller.split_subdomains_between_nodes(nodes, subds)
        self.assertEqual(assignments, [[subds[0]]])

    def test_more_gpus_than_subdomains(self):
        nodes = [
                MachineSpec('a', 'a', gpus=[0, 1]),
                MachineSpec('b', 'b', gpus=[1])
            ]
        subds = [
                SubdomainSpec2D((0,0), (10, 10), id_=0),
                SubdomainSpec2D((0,10), (10, 10), id_=1),
            ]

        assignments = controller.split_subdomains_between_nodes(nodes, subds)
        self.assertEqual(assignments, [[subds[0], subds[1]]])

    def test_more_subdomains_than_gpus(self):
        nodes = [
                MachineSpec('a', 'a', gpus=[0, 1]),
                MachineSpec('b', 'b', gpus=[1])
            ]
        subds = [
                SubdomainSpec2D((0,0), (10, 10), id_=0),
                SubdomainSpec2D((0,10), (10, 10), id_=1),
                SubdomainSpec2D((0,20), (10, 10), id_=2),
                SubdomainSpec2D((0,30), (10, 10), id_=3),
            ]

        assignments = controller.split_subdomains_between_nodes(nodes, subds)
        self.assertEqual(assignments, [[subds[0], subds[1], subds[2]], [subds[3]]])


if __name__ == '__main__':
    unittest.main()

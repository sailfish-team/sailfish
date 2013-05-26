/***************************************************************************
 *   Copyright (C) 2008 by BEEKHOF, Fokko                                  *
 *   fpbeekhof@gmail.com                                                   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <iostream>
#include <fstream>
#include <cvmlcpp/base/StringTools>
#include <cvmlcpp/volume/DTree>

int main()
{
	typedef cvmlcpp::DTree<int, 3> OctTreeInt;

	// Initialize the tree with a value
	OctTreeInt tree(7);
	assert(tree.root()() == 7);

	// Assign new value
	tree() = 1; // implicitly assigned to the root node
	tree.root()() = 5; // explicitly assigned to the root node
	assert(tree.root()() == 5);

	// The tree implicitly converts to its root node
	assert(tree.root() == tree);

	// Expanding a node without giving a value copies the current value
	// to all new children.
	tree.root().expand();
	for (unsigned i = 0; i < OctTreeInt::N; ++i)
		assert(tree[i]() == 5);


	// Change one leaf node of the root and expand it to a branch node.
	// The children of that node now all contain "3", the other children of
	// the root node still have their original value
	tree[0]() = 3;
	tree[0].expand();
	for (unsigned i = 1; i < OctTreeInt::N; ++i)
		assert(tree[i]() == 5);
	for (unsigned i = 0; i < OctTreeInt::N; ++i)
		assert(tree[0][i]() == 3);

	// Test if we can find a node by walking its trail of indices from
	// the root node.
	for (std::size_t i = 0; i < 1/*OctTreeInt::N*/; ++i)
	{
		const std::vector<OctTreeInt::index_t>
			trail = tree[0][i].index_trail();
		assert(trail[0] == 0);
		assert(trail[1] == int(i));

		OctTreeInt::DNode node = tree.root();
		for (std::size_t j = 0; j < trail.size(); ++j)
			node = node[ trail[j] ];
		assert(node == tree[0][i]);
	}

	// Collapse the expanded node and assign it a value of 6
	tree[0].collapse(6);
	assert(tree[0]() == 6);

	// Collapse the whole tree and assign it a value of 6
	tree.collapse(4);
	assert(tree.root()() == 4);

	// Collapse nodes with grand-children
	tree.expand();
	tree[0].expand();
	tree[0][0]() = 2;
	tree.collapse(-2);
	assert(tree.root()() == -2);
	// Expand the tree, assign all children different values.
	int values [] = {0, 1, 2, 3, 4, 5, 6, 7};
	tree.expand(values, values+OctTreeInt::N);
	assert(tree[3]() == 3);

	// Grab a node in the tree
	typedef OctTreeInt::DNode DNode8i;
	DNode8i node = tree[3];

	// See if all is consistent
	assert(node != tree[2]);
	assert(node.parent() == node.parent());
	assert(node.parent() == tree.root());
	assert(node != tree.root());

	// Wrong things should throw
	try { // Root node has no index!
		assert(node.parent().index() >= 0);
		assert(false); // previous should throw
	}
	catch(cvmlcpp::NoParentException &e) {
	}

	try { // Root node has no parent
		DNode8i n = node.parent().parent();
		assert(false); // previous should throw
	}
	catch(cvmlcpp::NoParentException &e) {
	}

	// Expand the node and fill with N different values
	assert(node.isLeaf());
	node.expand(values, values+OctTreeInt::N);
	assert(!node.isLeaf());

	// Verify that the i-th child of the node is the i-th child of the
	// 3th child of the root.
	for (unsigned i = 0; i < OctTreeInt::N; ++i)
	{
		// This compares the nodes - NOT the values
		assert(node[i] == tree[3][i]);

		// Although the nodes are different ...
		assert(node[i] != tree[i]);

		// ... the values should be identical, except the expanded node.
		if (i != 3)
		{
			// A value is obtained with operator() ...
			assert(node[i]() == tree[i]());

			// ... or with a static_cast
// 			assert( int(node[i]) == int(tree[i]));
		}
	}

	// Some arithmetic - DTrees and DNodes have reference semantics
	for (unsigned i = 0; i < OctTreeInt::N; ++i)
	{
		node[i]() += 10;
		assert(node[i]() == tree[3][i]());
	}

	// Copying data
	OctTreeInt tree2 = tree.clone(); // Full tree copy
	assert(tree2 != tree);

	OctTreeInt tree3 = tree[3].clone(); // Partial tree copy;
	for (unsigned i = 0; i < OctTreeInt::N; ++i)
	{
		assert(tree [3][i]() == tree2[3][i]()); // all values equal
		assert(tree3[i]() == tree2[3][i]()); // all values equal
		assert(tree3[i].index() == tree2[3][i].index());
		assert(tree3[i].index() == int(i));
		assert(tree [3][i].depth() == 2u);
		assert(tree2[3][i].depth() == 2u);
		assert(tree3[i].depth() == 1u);
	}

	// Nodes can be identified with their ID. This ID indicates a
	// position, it is valid in a copy of the tree.
	for (unsigned i = 0; i < OctTreeInt::N; ++i)
	{
		// Can we find ourselves ?
		assert( tree[3][i] == tree.retrieve(tree[3][i].id()) );

		// Equal in other tree (by value) ?
		assert( tree[3][i]() == tree2.retrieve(tree[3][i].id())() );
	}

// 	std::cout << tree;

	std::ofstream out("/tmp/tree.xml");
	out << tree;
	out.close();

	OctTreeInt tree4;
	std::ifstream in("/tmp/tree.xml");
	in >> tree4;
	in.close();
// 	std::cout << tree4;
	// Equal in other tree (by value) ?
	for (unsigned i = 0; i < OctTreeInt::N; ++i)
	{
		if (i != 3)
			assert( tree[i]() == tree4.retrieve(tree[i].id())() );
		assert( tree[3][i]() == tree4.retrieve(tree[3][i].id())() );
	}

	return 0;
}

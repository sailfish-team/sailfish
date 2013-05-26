/***************************************************************************
 *   Copyright (C) 2005, 2006, 2007 by F. P. Beekhof                       *
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
 *   along with program; if not, write to the                              *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

#include <cassert>
#include <cvmlcpp/base/Meta>

namespace cvmlcpp
{

namespace detail
{

template <template <typename Tm, std::size_t D, typename Aux> class Matrix_t,
	  typename Ti, typename A>
void replaceVoxels(Matrix_t<Ti, 3u, A> &matrix,
			const Ti from, const Ti nb, const Ti to)
{
	const std::size_t mX = matrix.extent(X);
	const std::size_t mY = matrix.extent(Y);
	const std::size_t mZ = matrix.extent(Z);

#ifdef _OPENMP
#pragma omp parallel for
#endif
	for (int i = 0; i < int(mX); ++i)
	{
		const std::size_t im = (i + mX - 1) % mX;
		const std::size_t ip = (i + 1) % mX;

		for (std::size_t j = 0; j < mY; ++j)
		{
			const std::size_t jm = (j + mY - 1) % mY;
			const std::size_t jp = (j + 1) % mY;

			for (std::size_t k = 0; k < mZ; ++k)
			{
				const std::size_t km = (k + mZ - 1) % mZ;
				const std::size_t kp = (k + 1) % mZ;

				if (  (matrix[i][j][k] == from) &&
				      ( (matrix[im][jm][km] == nb) ||
					(matrix[im][jm][k ] == nb) ||
					(matrix[im][jm][kp] == nb) ||
					(matrix[im][j ][km] == nb) ||
					(matrix[im][j ][k ] == nb) ||
					(matrix[im][j ][kp] == nb) ||
					(matrix[im][jp][km] == nb) ||
					(matrix[im][jp][k ] == nb) ||
					(matrix[im][jp][kp] == nb) ||
					(matrix[i ][jm][km] == nb) ||
					(matrix[i ][jm][k ] == nb) ||
					(matrix[i ][jm][kp] == nb) ||
					(matrix[i ][j ][km] == nb) ||
					// skip[ i][ j][ k)
					(matrix[i ][j ][kp] == nb) ||
					(matrix[i ][jp][km] == nb) ||
					(matrix[i ][jp][k ] == nb) ||
					(matrix[i ][jp][kp] == nb) ||
					(matrix[ip][jm][km] == nb) ||
					(matrix[ip][jm][k ] == nb) ||
					(matrix[ip][jm][kp] == nb) ||
					(matrix[ip][j ][km] == nb) ||
					(matrix[ip][j ][k ] == nb) ||
					(matrix[ip][j ][kp] == nb) ||
					(matrix[ip][jp][km] == nb) ||
					(matrix[ip][jp][k ] == nb) ||
					(matrix[ip][jp][kp] == nb) ) )
				{
					matrix[i][j][k] = to;
				}
			}
		}
	}
}

template <std::size_t D>
struct Expander
{
	template <class Slice, typename It, typename T>
	static void fill_block(Slice slice, const It offsets, const std::size_t length, const T value)
	{
//std::cout << "\t\t\tDimension " << D << " length " << length << " offset " << (*offsets) << std::endl;
		for (std::size_t i = 0; i < length; ++i)
			Expander<D-1>::fill_block/*_slice*/
				(slice[ *offsets+i ], offsets+1, length, value);
	}
};

template <>
struct Expander<0> { }; // No bullshit

template <>
struct Expander<1>
{
	template <class Slice, typename It, typename T>
	static void fill_block(Slice slice, const It offsets, const std::size_t length, const T value)
	{
//std::cout << "\t\t\t\t--> Writeout length:" << length << " " << int(value) << std::endl;
		assert(length > 0);
		for (std::size_t i = 0; i < length; ++i)
			slice[ *offsets+i ] = value;
	}
};


template <template <typename Tm, std::size_t Dm, typename Aux> class Matrix_t,
	  typename T, std::size_t D, typename A>
void expandRecursive(const typename DTree<T, D>::DNode & node, Matrix_t<T, D, A> &matrix,
		const std::array<std::size_t, D> offsets,
		const std::size_t length, const std::size_t depth_increase)
{
/*
for (unsigned d = 0; d < node.depth(); ++d)
	std::cout << "\t";
std::cout << "["<<node.depth()<<"] (" << offsets[X] << " " << offsets[Y] << ") " << length << std::endl;
*/
	if (node.isLeaf())
	{
		Expander<D>::fill_block(matrix, offsets.begin(), length << depth_increase, node());
		return;
	}

	const std::size_t sub_length = length / 2u;
	assert(sub_length > 0);
	for (std::size_t index = 0; index < 1u << D; ++index)
	{
		std::array<std::size_t, D> sub_offsets = offsets;
		for (std::size_t d = 0; d < D; ++d)
		{
			assert( ((index & (1u<<d)) >> d) == 0 ||
				((index & (1u<<d)) >> d) == 1 );
			sub_offsets[d] += ((index & (1u<<d)) >> d) * sub_length;
			assert(sub_offsets[d] < offsets[d] + length);
		}
		expandRecursive(node[index], matrix, sub_offsets, sub_length, depth_increase);
	}
}


/*
template <typename T, std::size_t D>
bool block_homogeneous(const std::size_t size, const Matrix<T, D> data,
		   const StaticVector<std::size_t, D> &offset, const unsigned axis)
{
	assert(size > 0);

}

// Forward Declaration
template <typename T, std::size_t D>
void MatrixToDNode_(typename DTree<T, D>::DNode voxels,
		 const double voxelSize, const StaticVector<std::size_t, D> &offset,
		 const Matrix<T, D> voxels,
		 const std::size_t maxLevel, const std::size_t currentLevel,
		 const T padding);
		 
template <typename T, std::size_t D>
void handleSubDNode__(typename DTree<voxel_type, 3u>::DNode voxels,
		 const double voxelSize, const StaticVector<std::size_t, D> &offset,
		 const vector_type &subVoxOffset, const Matrix<T, D> &data,
		 const std::size_t maxLevel, const std::size_t currentLevel,
		 const T &padding, const unsigned nodeIndex)
{
	const std::size_t size = 1u << (maxLevel-currentLevel);

	// Create offset of subnode
	StaticVector<std::size_t, D> os = offset;
	for (unsigned dim = 0; dim < D; ++dim)
		os[dim] += ((nodeIndex & (1u<<dim)) >> dim) * size / 2u;

	MatrixToDNode_<T, D>(voxels[nodeIndex], voxelSize, os, subVoxOffset,
				data, maxLevel, currentLevel+1, padding);
}

template <typename T, std::size_t D>
void MatrixToDNode_(typename DTree<T, D>::DNode voxels,
		 const double voxelSize, const StaticVector<std::size_t, D> &offset,
		 const Matrix<T, D> data,
		 const std::size_t maxLevel, const std::size_t currentLevel,
		 const T &padding)
{
	assert( currentLevel <= maxLevel );
	const std::size_t size = 1u << (maxLevel-currentLevel);

	const bool homogeneous = block_homogeneous( size, data, offset );

	if ( !homogeneous)
	{
		assert(currentLevel < maxLevel);
		// Expand into subnodes to handle different content
		voxels.expand();

		// Voxelize subnodes
#ifdef _OPENMP
		if (omp_get_max_threads() >= 1 << (3*currentLevel))
		{
			#pragma omp parallel for
			for (int i = 0; i < (1<<D); ++i)
				handleSubDNode__<T, D>(voxelTree,
					offset, data, maxLevel,
					currentLevel, padding, i);
		}
		else
#endif
			for (unsigned i = 0u; i < (1<<D); ++i)
				handleSubDNode__<T, D>(voxelTree,
					offset, data, maxLevel,
					currentLevel, padding, i);

		// If all underlying nodes are all leaf nodes with the same value,
		// they can be compacted.
		bool sub_homogeneous = true; // Must be leaf to be homogeneous
		for (unsigned i = 0u; i < (1<<D) && sub_homogeneous; ++i)
			sub_homogeneous = voxels[i].isLeaf() &&
					 (voxels[i]() == voxels[0]());
		if (sub_homogeneous)
			voxels.collapse(voxels[0]());
	}
	else
	{
//std::cout << "Block " << offset[X] << " " << offset[Y] << " " << offset[Z] << " size " << size << " counted " << counted_inside << std::endl;
		// Convert to voxel values by majority voting
		assert(voxels.isLeaf());
		voxels() = ;
	}

	assert(!voxels.isLeaf() || voxels() == inside || voxels() == outside);
}

template <typename T, std::size_t D>
void MatrixToDTree_(DTree<T, D> &voxelTree, const Matrix<T, D> &voxels,	const T &padding)
{
	// Clear to empty space
	if (voxelTree.root().isLeaf())
		voxelTree.root()() = padding;
	else
		voxelTree.collapse(padding);

	const std::size_t dimension = *std::max_element(voxels.extents(), voxels.extents()+D); // all equal
	const std::size_t maxLevels = log2(dimension);
	assert(dimension <= 1u << maxLevels);

	StaticVector<std::size_t, D> offset(0);
	MatrixToDNode_<T, D>(voxelTree.root(), offset, voxels, maxLevels, 0u, padding);
}
*/
} // end namespace detail

/*
template <typename T, std::size_t D>
DTree<T, D> MatrixToDTree(const Matrix<T, D> &data, const T padding)
{
	DTree<T, D> tree;
	const std::size_t dimension = *std::max_element(data.extents(), data.extents()+D); // all equal
	const std::size_t maxLevels = log2(dimension);
	assert(dimension <= 1u << maxLevels);

	StaticVector<std::size_t, D> offset(0);
	MatrixToDNode_<T, D>(tree.root(), offset, data, maxLevels, 0u, padding);
	return tree;
}
*/
template <  template <typename Tm, std::size_t D, typename Aux> class Matrix_t,
	    typename Ti, typename A>
void cover(Matrix_t<Ti, 3u, A> & matrix,
	   const Ti layerValue,   const Ti insideValue,
	   const Ti outsideValue, const bool replaceOutside)
{
	if (replaceOutside)
		detail::replaceVoxels(matrix, outsideValue, insideValue,  layerValue);
	else
		detail::replaceVoxels(matrix, insideValue,  outsideValue, layerValue);
}

template <template <typename Tm, std::size_t Dm, typename Aux> class Matrix_t,
	  typename T, std::size_t D, typename A>
bool expand(const typename DTree<T, D>::DNode & node, Matrix_t<T, D, A> &matrix,
		const std::size_t depth)
{
	const  std::size_t max_depth = node.max_depth();
	if (depth > 0 && depth < max_depth)
		return false;
	assert(depth == 0 || depth >= max_depth);
	const std::size_t depth_increase = (depth > max_depth) ? depth - max_depth : 0;

	std::array<std::size_t, D> offsets;
	std::fill(offsets.begin(), offsets.end(), 0);
	const std::size_t length = 1ul << node.max_depth();

	std::array<std::size_t, D> dimensions;
	std::fill(dimensions.begin(), dimensions.end(), length << depth_increase);
	typedef array_traits<Matrix_t, T, D, A>  Traits;
	Traits::resize(matrix, dimensions.begin());

//	assert(size() == (1ul << D) << tree.max_depth());
	detail::expandRecursive(node, matrix, offsets, length, depth_increase);

	return true;
}

template <template <typename Tm, std::size_t Dm, typename Aux> class Matrix_t,
	  typename T, std::size_t D, typename A>
bool expand(const DTree<T, D> & tree, Matrix_t<T, D, A> &matrix, const std::size_t depth)
{
	return expand(tree.root(), matrix, depth);
}

} // namespace cvmlcpp

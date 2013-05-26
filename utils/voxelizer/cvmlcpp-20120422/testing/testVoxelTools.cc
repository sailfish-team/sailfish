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

#include <cstdlib>
#include <cvmlcpp/base/Matrix>
#include <cvmlcpp/volume/DTree>
#include <cvmlcpp/volume/Geometry>
#include <cvmlcpp/volume/VolumeIO>
#include <cvmlcpp/volume/Voxelizer>
#include <cvmlcpp/volume/VoxelTools>

using namespace cvmlcpp;


/*
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>

#include <boost/lexical_cast.hpp>
*/

/*
void imgs(const cvmlcpp::Matrix<char, 3u> &m, const cvmlcpp::Matrix<char, 3u> &m2)
{
	assert(m.extent(X) <= m2.extent(X));
	assert(m.extent(Y) <= m2.extent(Y));
	assert(m.extent(Z) <= m2.extent(Z));

	using namespace boost::gil;
	gray8_image_t image(m.extent(Z), m.extent(Y));
	gray8_view_t img = view(image);

	assert(img.height() == m.extent(Y));
	assert(img.width () == m.extent(Z));
	for (std::size_t x = 0u; x < m.extent(X); ++x)
	{
		for (std::size_t h = 0u; h < img.height(); ++h)
		for (std::size_t w = 0u; w < img.width (); ++w)
		{
			if (m[x][h][w] == m2[x][h][w])
				img(w, h) = m[x][h][w] ? 255 : 0;
			else if (m[x][h][w])
				img(w, h) = 127+64;
			else
				img(w, h) = 64;
		}
		const std::string name = std::string("tmp/slice_") +
			((x < 10) ? "0" :"") + ((x < 100) ? "0" :"") +
			boost::lexical_cast<std::string>(x) + ".png";
		png_write_view(name, img);
	}

}

void img(const cvmlcpp::Matrix<char, 2u> &m)
{
	using namespace boost::gil;
	gray8_image_t image(m.extent(X), m.extent(Y));
	gray8_view_t img = view(image);

	for (std::size_t h = 0u; h < img.height(); ++h)
	for (std::size_t w = 0u; w < img.width (); ++w)
		img(w, h) = m[h][w] ? 255 : 0;
	png_write_view("quadtree.png", img);
}
*/

int main(int argc, char **argv)
{

	Geometry<float> g;
 	Matrix<char, 3u> m;
	const double voxelSize = 0.1;
	std::size_t elems = 0u;

	assert(readSTL(g, "cube.stl"));
	assert(voxelize(g, m, voxelSize, 2));


	for (std::size_t x = 0u; x < m.extents()[X]; ++x)
	for (std::size_t y = 0u; y < m.extents()[Y]; ++y)
	for (std::size_t z = 0u; z < m.extents()[Z]; ++z)
		if (m[x][y][z] == 1)
			++elems;
//	std::cout << "Elements: " << elems << std::endl;

	assert(elems == 1000u);

	const char two = 2u;
	cover(m, two);

	std::size_t ones  = 0u;
	std::size_t twos = 0u;

	for (std::size_t x = 0u; x < m.extents()[X]; ++x)
	for (std::size_t y = 0u; y < m.extents()[Y]; ++y)
	for (std::size_t z = 0u; z < m.extents()[Z]; ++z)
	{
		if (m[x][y][z] == 1)
			++ones;
		if (m[x][y][z] == 2)
			++twos;
	}

	assert(ones == 1000u);
	assert(twos == (6u * 10u * 10u) + (12u * 10u) + 8u);


/*
	DTree<char, 2> quadtree;
	quadtree.root().expand();
	quadtree[0]() = 1;
	quadtree[0].expand();
	assert(quadtree[0][1]() == 1);
	quadtree[0][1]() = 0;
	assert(quadtree.max_depth() == 2);
// 	std::cout << "Max Depth " << octree.max_depth() << std::endl;

	cvmlcpp::Matrix<char, 2> m2;
	assert(expand(quadtree, m2));
	img(m2);
	for (std::size_t x = 0u; x < m2.extents()[X]; ++x)
	{
		for (std::size_t y = 0u; y < m2.extents()[Y]; ++y)
			std::cout << int(m2[x][y]) << " ";
		std::cout << std::endl;
	}
*/

	assert(readSTL(g, "d4.stl"));
	assert(g.nrPoints() == 4u);
	assert(g.nrFacets() == 4u);
	assert(voxelize(g, m, voxelSize));

	DTree<char, 3> octree;
	assert(voxelize(g, octree, voxelSize));
	Matrix<char, 3u> m2;
	assert(expand(octree, m2, 4));//, logN));

//std::cout << m.extent(X) << " " << m.extent(Y) << " " << m.extent(Z) << " " << std::endl;
//std::cout << m2.extent(X) << " " << m2.extent(Y) << " " << m2.extent(Z) << " " << std::endl;
	assert(m.extent(X) <= m2.extent(X));
	assert(m.extent(Y) <= m2.extent(Y));
	assert(m.extent(Z) <= m2.extent(Z));

//	imgs(m, m2);
	elems = 0;
	for (std::size_t x = 0u; x < m.extent(X); ++x)
	for (std::size_t y = 0u; y < m.extent(Y); ++y)
	for (std::size_t z = 0u; z < m.extent(Z); ++z)
	{
		if (m[x][y][z] != m2[x][y][z])
			std::cout << "Mismatch (" << x << ", " << y << ", " << z << ") "
				<< int(m[x][y][z]) << " vs " << int(m2[x][y][z]) << std::endl;
		assert(m[x][y][z] == m2[x][y][z]);
		assert(m[x][y][z] == 0 || m[x][y][z] == 1);

		elems += m[x][y][z];
	}
	assert(elems > 0);
//	std::cout << "Elements: " << elems << std::endl;

	return 0;
}
